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
n_epochs=50000
server=ls97
selected_filter=318
selected_layer=25
class_index=100
num_class=39
# -------------------------------
# Variables
cuda_visible_devices=4
alpha=100
beta=1
gamma=1
delta=0
img_index=0
batch_size=1

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=$n_epochs \
    --server=$server \
    --eval_frequency=100 \
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
    --resume=011-161 \
    --num_class=$num_class \
    --class_index=$class_index \
    --mode=keep \
    --inter=True \
    --rho=0 \
    --regularization=L1 \
    --smoothing=TotalVariation \
    --regular_ex=2 \
    --batch_size=$batch_size \
    --img_index=$img_index \
    --rescale=False \
    2>&1 | tee $log_file
