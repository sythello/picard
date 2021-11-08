set -e 

SPIDER_DIR=/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider
PREDS_IN_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs
PREDS_OUT_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/picard-test-save

python picard_infer.py \
	-config_json configs/host_infer.json \
	-test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
	-orig_dev_path $SPIDER_DIR/dev.json \
	-eval_vers 2.12.1.0t-2.18.2.0i 2.12.1.1t-2.18.2.1i 2.12.1.2t-2.18.2.2i \
	-eval_in_dir $PREDS_IN_DIR \
	-eval_out_dir $PREDS_OUT_DIR
