export DEVICE=5

export CKPT_PATH=/data1/ay0119/ckpts/231104-1034-AdaToken-0.15-91284/checkpoint-2000000
array=(124 96831 5671 39579 22177)
for var in "${array[@]}"
do
    export SEED=$var
    bash ./scripts/bert-ftune-ada.sh
done

# export CKPT_PATH=/data1/ay0119/ckpts/231009-202410-AdaToken-0.15-124/checkpoint-1020000
# array=(124 96831 5671 39579 22177)
# for var in "${array[@]}"
# do
#     export SEED=$var
#     bash ./scripts/bert-ftune-ada.sh
# done