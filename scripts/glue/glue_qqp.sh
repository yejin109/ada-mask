export TASK_NAME=qqp
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
  --num_train_epochs 5 \
  --seed $SEED\
  --cache_dir /data1/ay0119/hf-cache \
  --output_dir /data1/ay0119/bert-tiny-main/$TASK_NAME \
  --logging_steps 5000\
  --save_steps 20000\
  --evaluation_strategy steps\
  --overwrite_output_dir