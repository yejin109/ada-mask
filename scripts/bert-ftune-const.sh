export CKPT_PATH=/data1/ay0119/ckpts/231021-1840-Const-0.15-91284/checkpoint-250000


# the best fine-tuning hyperparameters trained for 4 epochs:
#     batch sizes: 8, 16, 32, 64, 128
#     learning rates: 3e-4, 1e-4, 5e-5, 3e-5
# BEST seq len & batch size = 128 * 32
# original path : --model_name_or_path google/bert_uncased_L-2_H-128_A-2 \

export TASK_NAME=sst2
CUDA_VISIBLE_DEVICES=5 python /home/ay0119/bert-tiny-main/run_glue.py \
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


export TASK_NAME=mrpc
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
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
  --logging_steps 100\
  --save_steps 1000\
  --evaluation_strategy steps\
  --overwrite_output_dir

export TASK_NAME=rte
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
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
  --logging_steps 100\
  --save_steps 600\
  --evaluation_strategy steps\
  --overwrite_output_dir

export TASK_NAME=mnli
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
  --model_name_or_path $CKPT_PATH\
  --config_name  google/bert_uncased_L-2_H-128_A-2 \
  --tokenizer_name  google/bert_uncased_L-2_H-128_A-2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --logging_steps 5000\
  --save_steps 20000\
  --evaluation_strategy steps\
  --overwrite_output_dir


export TASK_NAME=cola
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
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
  --logging_steps 400\
  --save_steps 2000\
  --evaluation_strategy steps\
  --overwrite_output_dir


# # Iteration num: 13096
export TASK_NAME=qnli
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
  --model_name_or_path $CKPT_PATH\
  --config_name  google/bert_uncased_L-2_H-128_A-2 \
  --tokenizer_name  google/bert_uncased_L-2_H-128_A-2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --logging_steps 3000\
  --save_steps 5000\
  --evaluation_strategy steps\
  --overwrite_output_dir

# # Iteration num = 45484
export TASK_NAME=qqp
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
  --model_name_or_path $CKPT_PATH\
  --config_name  google/bert_uncased_L-2_H-128_A-2 \
  --tokenizer_name  google/bert_uncased_L-2_H-128_A-2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --logging_steps 5000\
  --save_steps 20000\
  --evaluation_strategy steps\
  --overwrite_output_dir


export TASK_NAME=stsb
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
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
  --logging_steps 400\
  --save_steps 2000\
  --evaluation_strategy steps\
  --overwrite_output_dir

export TASK_NAME=wnli
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_glue.py \
  --model_name_or_path $CKPT_PATH\
  --config_name  google/bert_uncased_L-2_H-128_A-2 \
  --tokenizer_name  google/bert_uncased_L-2_H-128_A-2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 20 \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --logging_steps 100\
  --save_steps 2000\
  --evaluation_strategy steps\
  --overwrite_output_dir