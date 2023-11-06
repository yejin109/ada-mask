# the best fine-tuning hyperparameters trained for 4 epochs:
#     batch sizes: 8, 16, 32, 64, 128
#     learning rates: 3e-4, 1e-4, 5e-5, 3e-5
# BEST seq len & batch size = 128 * 32 
# export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
# CUDA_VISIBLE_DEVICES=5 python /home/ay0119/bert-tiny-main/run_pretrain.py \
#   --ada_token \
#   --train \
#   --data all\
#   --cache_dir /data1/ay0119/hf-cache \
#   --model_type google/bert_uncased_L-2_H-128_A-2 \
#   --max_seq_length 512 \
#   --b_train 16 \
#   --lr 1e-4 \
#   --epochs 2 \
#   --seed 124 \
#   --device cuda:5 \
#   --logging_steps 10000

# export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
# CUDA_VISIBLE_DEVICES=5 python /home/ay0119/bert-tiny-main/run_pretrain.py \
#   --ada_token \
#   --train \
#   --data all\
#   --cache_dir /data1/ay0119/hf-cache \
#   --model_type google/bert_uncased_L-2_H-128_A-2 \
#   --max_seq_length 512 \
#   --b_train 16 \
#   --lr 1e-4 \
#   --epochs 2 \
#   --seed 83724 \
#   --device cuda:5 \
#   --logging_steps 10000

# export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 /home/ay0119/bert-tiny-main/run_pretrain.py \
#   --ada_token \
#   --train \
#   --data all\
#   --cache_dir /data1/ay0119/hf-cache \
#   --model_type prajjwal1/bert-tiny \
#   --max_seq_length 128 \
#   --b_train 128 \
#   --lr 5e-5 \
#   --epochs 6 \
#   --seed 63126 \
#   --logging_steps 5000


# Optimized
# 
# 현재 (seq len , batch size) = (512, 16) 조합인듯 (at least 10it/s)
# (512, 32) 의 경우 3~4it/s
# (256, 32) 도 가능한 조합인 듯
# (128, 128) 도 가능한 조합

# Hypo : 학습을 더 많이 한다면 도움이 될까?
# 일단 이어서 학습해보기. 이 경우 masking probability를 마지막 p로 고정하기
# 현재 warmup steps도 adam beta2도 원래대로 돌려보기, seed도
# --ckpt /data1/ay0119/ckpts/231009-202410-AdaToken-0.15-124/checkpoint-1020000 \
# 아이디어 : Token processed는 맞춰야 한다. 이에 따라서 max steps를 정하면 lr scheduler가 맞춰서 스케줄링을 한다. 즉 원래대로라면 진행해야 하는 학습량을 수행하도록 하는 것
# >> longer training인 것처럼 scheduler를 해서 성능 향상은 가능하나 train loss 수렴이 비슷

# Hypo : larger batch, large learning rate ? 
# 지금 512로는 어느정도 한계가 비슷하게 형성됨.
# 그래서 large batch, large learning rate 구조로 생각 >> gradient accumulation step 조정해보기
# 그런데 gradient accum이 늘어나면 step count도 뒤로 밀리게 된다. 그래서 이게 n배 된다면 다른 것들도 n배 줄여야 같은 시간
# 옵션은 1) (128, 128)에서 large batch 2) (512,16)에서 large batch 16
# 일단 목표는 (batch, steps) :  (2K, 125K), (8K, 100K)
# 현재 선택은 (512, 16)에서 2K 효과를 내도록 128 gradient accum step 구현해보기 >> bottleneck
# [>] (512, 16)에서 (4, 250K) --> 16H eff batch 64 : 즉 이전과 동일하게 학습을 하는데 gradient update를 accum해서 large batch 효과로
# (256, 32) (64, 125K) : 전처리 cache없음 for google
# (128, 128) (4, 125K) : 13H 정도 
# --model_type google/bert_uncased_L-2_H-128_A-2 \
# BERT-large : seq len 128 batch 4 accum 1 3.7it/s

export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 /home/ay0119/bert-tiny-main/run_pretrain.py \
  --train \
  --data all\
  --cache_dir /data1/ay0119/hf-cache \
  --model_type google/bert_uncased_L-2_H-128_A-2 \
  --train_test_split 0.00001\
  --max_seq_length 512 \
  --lr 1e-4 \
  --warmup_steps 10000\
  --adam_beta2 0.999\
  --epochs 40 \
  --b_train 16 \
  --gradient_accumulation_steps 1\
  --seed 91284 \
  --p 0.15 \
  --mask_tolerance 0.02\
  --mask_increment 0.002\
  --max_steps 2000000\
  --logging_steps 50000\
  --save_steps 100000