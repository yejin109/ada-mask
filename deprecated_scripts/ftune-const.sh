export DEVICE=4

export CKPT_PATH=/data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint-1020000
array=(124 96831 5671 39579 22177)
for var in "${array[@]}"
do
    export SEED=$var
    bash ./scripts/bert-ftune-ada.sh
done

export CKPT_PATH=/data1/ay0119/ckpts/231021-1840-Const-0.15-91284/checkpoint-250000
array=(124 96831 5671 39579 22177)
for var in "${array[@]}"
do
    export SEED=$var
    bash ./scripts/bert-ftune-ada.sh
done