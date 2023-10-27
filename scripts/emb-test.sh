# export CKPT_PATH=/data1/ay0119/bert-tiny-main/all/google/bert_uncased_L-2_H-128_A-2/231009-202410-AdaToken-0.15-124/checkpoint-1000000
# export CKPT_PATH=/data1/ay0119/bert-tiny-main/all/prajjwal1/bert-tiny/231014-105300-AdaToken-0.15-63126/checkpoint-760000
# export CKPT_PATH=/data1/ay0119/bert-tiny-main/all/prajjwal1/bert-tiny/231015-164528-AdaToken-0.15-63126/checkpoint-900000

# export CKPT_PATH=/data1/ay0119/ckpts/231014-105300-AdaToken-0.15-63126/checkpoint-920000
# export CKPT_PATH=/data1/ay0119/ckpts/231015-164528-AdaToken-0.15-63126/checkpoint-900000
# export CKPT_PATH=/data1/ay0119/ckpts/231016-200742-AdaToken-0.15-63126/checkpoint-900000
# export CKPT_PATH=/data1/ay0119/ckpts/231018-130805-AdaToken-0.17-124/checkpoint-1000000
# export CKPT_PATH=/data1/ay0119/ckpts/231019-0830-AdaToken-0.15-71261/checkpoint-1400000
export CKPT_PATH=/data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-200000
export DEVICE=5

# the best fine-tuning hyperparameters trained for 4 epochs:
#     batch sizes: 8, 16, 32, 64, 128
#     learning rates: 3e-4, 1e-4, 5e-5, 3e-5
# BEST seq len & batch size = 128 * 32
# original path : --model_name_or_path google/bert_uncased_L-2_H-128_A-2 \

# [Stable config]
export TASK_NAME=sst2
CUDA_VISIBLE_DEVICES=$DEVICE python /home/ay0119/bert-tiny-main/run_glue.py \
  --model_name_or_path $CKPT_PATH\
  --config_name  google/bert_uncased_L-2_H-128_A-2 \
  --tokenizer_name  google/bert_uncased_L-2_H-128_A-2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --save_steps 2000\
  --evaluation_strategy steps \
  --overwrite_output_dir

