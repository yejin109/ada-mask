export DEVICE=5
export SEED=124
array=(1e-5 3e-5 5e-5 8e-5)

# export CKPT_PATH=/data1/ay0119/ckpts/231021-0234-AdaToken-0.15-91284/checkpoint-250000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done

# export CKPT_PATH=/data1/ay0119/ckpts/231021-0234-AdaToken-0.15-91284/checkpoint-200000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done

# export CKPT_PATH=/data1/ay0119/ckpts/231024-1952-AdaToken-0.15-91284/checkpoint-250000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done

# export CKPT_PATH=/data1/ay0119/ckpts/231024-1952-AdaToken-0.15-91284/checkpoint-200000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done

# export CKPT_PATH=/data1/ay0119/ckpts/231031-0315-AdaToken-0.15-124/checkpoint-250000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done

# export CKPT_PATH=/data1/ay0119/ckpts/231031-0315-AdaToken-0.15-124/checkpoint-200000
# for var in "${array[@]}"
# do
#     export LR=$var
#     bash ./scripts/glue/glue_qnli.sh
# done





export CKPT_PATH=/data1/ay0119/ckpts/231109-0923-AdaToken-0.15-91284/checkpoint-250000
for var in "${array[@]}"
do
    export LR=$var
    bash ./scripts/glue/glue_qnli.sh
done

export CKPT_PATH=/data1/ay0119/ckpts/231109-0923-AdaToken-0.15-91284/checkpoint-200000
for var in "${array[@]}"
do
    export LR=$var
    bash ./scripts/glue/glue_qnli.sh
done

export CKPT_PATH=/data1/ay0119/ckpts/231110-1651-AdaToken-0.15-91284/checkpoint-250000
for var in "${array[@]}"
do
    export LR=$var
    bash ./scripts/glue/glue_qnli.sh
done

export CKPT_PATH=/data1/ay0119/ckpts/231110-1651-AdaToken-0.15-91284/checkpoint-250000
for var in "${array[@]}"
do
    export LR=$var
    bash ./scripts/glue/glue_qnli.sh
done