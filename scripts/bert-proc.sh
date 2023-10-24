# the best fine-tuning hyperparameters trained for 4 epochs:
#     batch sizes: 8, 16, 32, 64, 128
#     learning rates: 3e-4, 1e-4, 5e-5, 3e-5
# BEST seq len & batch size = 128 * 32
export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
CUDA_VISIBLE_DEVICES=4 python /home/ay0119/bert-tiny-main/run_pretrain.py \
  --data all\
  --cache_dir /data1/ay0119/hf-cache \
  --model_type google/bert_uncased_L-2_H-128_A-2 \
  --max_seq_length 128 \
  --b_train 64 \
  --lr 3e-5 \
  --epochs 2 \
  --seed 124 \
  --device cuda:5
