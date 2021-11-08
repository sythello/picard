set -e 

SPIDER_DIR=/vault/spider
PREDS_IN_DIR=~/SpeakQL/Allennlp_models/outputs
PREDS_OUT_DIR=/vault/SpeakQL/Allennlp_models/outputs/picard-test-save

mkdir -p $PREDS_OUT_DIR

python picard_infer.py \
	-config_json configs/host_infer.json \
	-test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
	-orig_dev_path $SPIDER_DIR/dev.json \
	-eval_vers 2.12.1.0t-2.18.2.0i 2.12.1.1t-2.18.2.1i 2.12.1.2t-2.18.2.2i 2.12.1.3t-2.18.2.3i 2.12.1.4t-2.18.2.4i \
	-eval_in_dir $PREDS_IN_DIR \
	-eval_out_dir $PREDS_OUT_DIR
