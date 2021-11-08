from typing import Optional, Dict

import os
import sys
import json
import random

import argparse
from argparse import ArgumentParser
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.hf_argparser import HfArgumentParser

from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
from seq2seq.serve_seq2seq import BackendArguments

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
    Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_json', '--config_json', type=str, required=True,
                        help='The path to config json (host_infer.json)')
    parser.add_argument('-test_dataset_path', '--test_dataset_path', type=str, required=True,
                        help='The path to test_dataset_path')
    parser.add_argument('-orig_dev_path', '--orig_dev_path', type=str, required=True,
                        help='The path to orig_dev_path')
    parser.add_argument('-eval_vers', '--eval_vers', type=str, required=True, nargs='*',
    					help='The versions to evaluate')
    parser.add_argument('-eval_in_dir', '--eval_in_dir', type=str, required=True,
                        help='The directory for prediction files (i.e. Allennlp_models/outputs)')
    parser.add_argument('-eval_out_dir', '--eval_out_dir', type=str, required=False, default=None,
                        help='The directory for output files (dataset json with preds and eval scores, i.e. /vault/SpeakQL/.../picard-test-save)')

    args = parser.parse_args()
    return args


def Picard_predict(pipe, question, db_id):
    outputs = pipe(inputs=Text2SQLInput(utterance=question, db_id=db_id))
    output = outputs[0]['generated_text']
    return output



def _Postprocess_rewrite_seq_wrapper(cand_dict, pred_dict):
    _tags = pred_dict['rewriter_tags']
    _rewrite_seq = pred_dict['rewrite_seq_prediction']
    _question_toks = cand_dict['question_toks']
    return Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)


def Full_evaluate_ILM(pipe,
                      eval_version,
                      rewriter_ILM_pred_path,
                      test_dataset_path,
                      orig_dev_path,
                      test_output_path=None,
                      ILM_rewrite_func=_Postprocess_rewrite_seq_wrapper):
    
    '''
    Example paths:
    rewriter_ILM_pred_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/output-{}.json'.format(VERSION)
    test_dataset_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
    orig_dev_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/dev.json'
    
    ILM_rewrite_func: Callable, args: (_tags, _rewrite_seq, _question_toks)
    '''
    
    VERSION = eval_version
    
    with open(rewriter_ILM_pred_path, 'r') as f:
        rewriter_ILM_preds = [json.loads(l) for l in f.readlines()]
    with open(test_dataset_path, 'r') as f:
        test_dataset = json.load(f)
    with open(orig_dev_path, 'r') as f:
        orig_dev_dataset = json.load(f)

    # len(rewriter_ILM_preds), len(test_dataset), sum([len(d) for d in test_dataset]), len(orig_dev_dataset)

    ## Quick evaluation: only using the 1st ASR candidate

    pred_idx = 0

    ref_list = []
    hyp_list = []

    for d in tqdm(test_dataset, desc=f'VERSION {VERSION}'):
        if len(d) == 0:
            continue

        c = d[0]

        p = rewriter_ILM_preds[pred_idx]
        _o_idx = c['original_id']
        o = orig_dev_dataset[_o_idx]
        assert ' '.join(c['question_toks']) == p['question'], (' '.join(c['question_toks']), p['question'])
        assert c['gold_question_toks'] == o['question_toks'], (c['gold_question_toks'], o['question_toks'])

        # Debug 
        # assert c['rewriter_tags'] == p['rewriter_tags'][:len(c['rewriter_tags'])], f"{c['rewriter_tags']}\n{p['rewriter_tags']}\nShould raise"

        _db_id = o['db_id']

        # _tags = p['tags_prediction']  # For previous taggerILM joint model 
        # _tags = p['tags']  # Before adding align_tags (when 'tags' refers to 'rewriter_tags')

        _tags = p['rewriter_tags']
        _rewrite_seq = p['rewrite_seq_prediction']
        _question_toks = c['question_toks']

        # _rewritten_question_toks = Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)
        _rewritten_question_toks = ILM_rewrite_func(c, p)
        _rewritten_question = ' '.join(_rewritten_question_toks)

        _pred_sql = Picard_predict(pipe, _rewritten_question, _db_id)

        _gold_sql = c['query']
        _exact, _score, _exec = EvaluateSQL(_pred_sql, _gold_sql, _db_id)

        # Save the taggerILM raw outputs, for later aggregation 
        c['pred_tags'] = p['rewriter_tags']
        c['pred_ILM'] = p['rewrite_seq_prediction']
        c['pred_ILM_cands'] = p['rewrite_seq_prediction_cands']

        # Save prediction results 
        c['rewritten_question'] = p['rewritten_question'] = _rewritten_question
        c['pred_sql'] = p['pred_sql'] = _pred_sql
        p['gold_sql'] = _gold_sql
        c['score'] = p['score'] = _score
        c['exact'] = p['exact'] = _exact
        c['exec'] = p['exec'] = _exec

        _rewritten_question_toks = [_t.lower() for _t in _rewritten_question_toks]
        _gold_question_toks = [_t.lower() for _t in c['gold_question_toks']]

        ref_list.append([_gold_question_toks])
        hyp_list.append(_rewritten_question_toks)

        pred_idx += len(d)

    # Only using the 1st candidate to rewrite 
    _avg_1st = sum([d[0]['score'] for d in test_dataset]) / len(test_dataset)
    _avg_exact_1st = sum([d[0]['exact'] for d in test_dataset]) / len(test_dataset)
    _avg_exec_1st = sum([d[0]['exec'] for d in test_dataset]) / len(test_dataset)

    ## Std-dev (1st cand only)
    _std_1st = np.std([d[0]['score'] for d in test_dataset])

    ## BLEU 
    _bleu = corpus_bleu(list_of_references=ref_list,
                        hypotheses=hyp_list)

    print('='*20, f'VERSION: {VERSION}', '='*20)
    print('avg_exact = {:.4f}'.format(_avg_exact_1st))
    # print('avg = {:.4f} (std = {:.4f})'.format(_avg_1st, _std_1st))
    print('avg = {:.4f}'.format(_avg_1st))
    print('avg_exec = {:.4f}'.format(_avg_exec_1st))
    print(f'BLEU = {_bleu:.4f}')
    print('='*55)
    
    if test_output_path is not None:
        with open(test_output_path, 'w') as f:
            json.dump(test_dataset, f, indent=4)



def main(args):
    # See all possible arguments by passing the --help flag to this program.
    parser = HfArgumentParser((PicardArguments, BackendArguments, DataTrainingArguments))
    picard_args: PicardArguments
    backend_args: BackendArguments
    data_training_args: DataTrainingArguments
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
    picard_args, backend_args, data_training_args = parser.parse_json_file(json_file=args.config_json)
    # else:
        # picard_args, backend_args, data_training_args = parser.parse_args_into_dataclasses()
        # raise ValueError(sys.argv[1:])

    # Initialize config
    config = AutoConfig.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        use_fast=True,
    )

    model_cls_wrapper = lambda model_cls: model_cls

    # Initialize model
    model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
        backend_args.model_path,
        config=config,
        cache_dir=backend_args.cache_dir,
    )

    # Initalize generation pipeline
    pipe = Text2SQLGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        db_path=backend_args.db_path,
        prefix=data_training_args.source_prefix,
        normalize_query=data_training_args.normalize_query,
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        device=backend_args.device,
    )

    test_dataset_path = args.test_dataset_path
    orig_dev_path = args.orig_dev_path

    for ver in args.eval_vers:
        rewriter_ILM_pred_path = os.path.join(args.eval_in_dir, f'output-{ver}.json')
        if args.eval_out_dir is not None:
            test_output_path = os.path.join(args.eval_out_dir, f'{ver}.json')
        
        Full_evaluate_ILM(pipe=pipe,
                          eval_version=ver,
                          rewriter_ILM_pred_path=rewriter_ILM_pred_path,
                          test_dataset_path=test_dataset_path,
                          orig_dev_path=orig_dev_path,
                          test_output_path=test_output_path,
                          ILM_rewrite_func=_Postprocess_rewrite_seq_wrapper)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)




