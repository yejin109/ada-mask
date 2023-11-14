# the best fine-tuning hyperparameters trained for 4 epochs:
#     batch sizes: 8, 16, 32, 64, 128
#     learning rates: 3e-4, 1e-4, 5e-5, 3e-5
# BEST seq len & batch size = 128 * 32


# export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
# CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_pretrain.py \
#   --train \
#   --data all\
#   --cache_dir /data1/ay0119/hf-cache \
#   --model_type google/bert_uncased_L-2_H-128_A-2 \
#   --max_seq_length 128 \
#   --b_train 32 \
#   --lr 1e-4 \
#   --epochs 2 \
#   --seed 124 \
#   --device cuda:4 \
#   --logging_steps 10000


# seq len 512 ver
export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 /home/ay0119/bert-tiny-main/run_pretrain.py \
  --seed 124 \
  --cache_dir /data1/ay0119/hf-cache \
  --model_type google/bert_uncased_L-2_H-128_A-2 \
  --data all\
  --train_test_split 0.0001\
  --max_seq_length 128 \
  --lr 5e-5 \
  --epochs 6 \
  --b_train 128 \
  --p 0.15 \
  --train \
  --logging_steps 50000\
  --save_steps 50000\
  --warmup_steps 0\
  --adam_beta2 0.999\
  --adam_eps 1e-8