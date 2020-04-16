set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=ImageNet
# -------------------------------
class_index=950
num_class=30
server=ls97
delta=0
# -------------------------------
# Variables
n_epochs=200000

cuda_visible_devices=2
selected_layer=6
selected_filter=19
alpha=10
beta=1
gamma=10


CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --server=$server \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=$n_epochs \
    --eval_frequency=10000 \
    --re_size=224 \
    --backbone=vgg16 \
    --optimizer=SGD \
    --learning_rate=1e-1 \
    --alpha=$alpha \
    --gamma=$gamma \
    --beta=$beta \
    --delta=$delta \
    --selected_layer=$selected_layer \
    --selected_filter=$selected_filter \
    --resume=037-0 \
    --num_class=$num_class \
    --class_index=$class_index \
    --mode=keep \
    --inter=False \
    --rho=0 \
    --regularization=L1 \
    --smoothing=TotalVariation \
    --regular_ex=3 \
    --delta=$delta \
    --rescale=False \
    2>&1 | tee $log_file
