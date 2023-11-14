export DEVICE=4
export MODEL_TYPE=google/bert_uncased_L-2_H-128_A-2


# array=(
#     google/bert_uncased_L-2_H-128_A-2
#     google/bert_uncased_L-4_H-256_A-4
#     google/bert_uncased_L-4_H-512_A-8
#     google/bert_uncased_L-8_H-512_A-8
#     google/bert_uncased_L-12_H-768_A-12)
array=(
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-100000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-200000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-300000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-400000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-500000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-600000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-700000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-800000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-900000
    /data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-1000000

    )

for var in "${array[@]}"
do
    export NAME_OR_PATH=$var
    bash ./scripts/memo.sh
done
