# CUDA_VISIBLE_DEVICES=4 python run_memo.py \
#     --name_or_path /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-1020000 \
#     --model_type google/bert_uncased_L-2_H-128_A-2 \
#     --sample_size 100 \
#     --seed 124 

CUDA_VISIBLE_DEVICES=4 python run_noise.py \
    --name_or_path google/bert_uncased_L-12_H-768_A-12 \
    --model_type google/bert_uncased_L-12_H-768_A-12 \
    --sample_idx 100 \
    --seed 124 \
    --mask_idx_1 3\
    --device cuda:0

CUDA_VISIBLE_DEVICES=4 python run_noise.py \
    --name_or_path google/bert_uncased_L-12_H-768_A-12 \
    --model_type google/bert_uncased_L-12_H-768_A-12 \
    --sample_idx 100 \
    --seed 124 \
    --mask_idx_1 3\
    --mask_idx_2 10\
    --device cuda:0

python analyze_noise.py \
    --target_1 100-3-10 \
    --target_2 100-3
