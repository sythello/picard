GIT_HEAD_REF=$(shell git rev-parse HEAD)

BASE_IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

DEV_IMAGE_NAME=text-to-sql-dev
TRAIN_IMAGE_NAME=text-to-sql-train
EVAL_IMAGE_NAME=text-to-sql-eval

BUILDKIT_IMAGE=tscholak/text-to-sql-buildkit:buildx-stable-1
BUILDKIT_BUILDER=buildx-local
BASE_DIR=$(shell pwd)

config=$1	## E.g. configs/train.json

docker pull tscholak/${TRAIN_IMAGE_NAME}:${GIT_HEAD_REF}

mkdir -p -m 777 train
mkdir -p -m 777 transformers_cache
mkdir -p -m 777 wandb

docker run \
	-it \
	--rm \
	--user 13011:13011 \
	--mount type=bind,source=${BASE_DIR}/train,target=/train \
	--mount type=bind,source=${BASE_DIR}/transformers_cache,target=/transformers_cache \
	--mount type=bind,source=${BASE_DIR}/configs,target=/app/configs \
	--mount type=bind,source=${BASE_DIR}/wandb,target=/app/wandb \
	tscholak/${TRAIN_IMAGE_NAME}:${GIT_HEAD_REF} \
	/bin/bash -c "python seq2seq/run_seq2seq.py ${config}"
