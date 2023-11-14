export DEVICE=5

export CKPT_PATH=/data1/ay0119/ckpts/231113-1648-AdaToken-0.15-91284/checkpoint-200000

# array=(5671 39579)
# for var in "${array[@]}"
# do
#     export SEED=$var
#     bash ./scripts/glue/glue_rte.sh
# done

array=(124 96831 5671 39579 22177)
for var in "${array[@]}"
do
    export SEED=$var
    bash ./scripts/glue/bert-ftune-ada.sh
done