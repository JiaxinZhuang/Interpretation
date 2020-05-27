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
eval_frequency=10000
# -------------------------------
# Variables
n_epochs=300000

cuda_visible_devices=9
selected_layer=layer4.0.relu
selected_filter=22
alpha=1
beta=1
gamma=1
guidedReLU=False
backbone=resnet18


CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --server=$server \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=$n_epochs \
    --eval_frequency=$eval_frequency \
    --re_size=224 \
    --backbone=$backbone \
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
    --guidedReLU $guidedReLU \
    2>&1 | tee $log_file
dir=./saved/generated
subdir=$dir/$experiment_index
last_epoch=$subdir`ls $subdir | sort -n | sed -n '$p'`
python -u src/utils/visualizations/visualize_comparision_resNet18.py \
    --exp $experiment_index \
    --layer $selected_layer \
    --epoch $last_epoch\
    --img_index -1 2>&1 | tee -a $log_file
