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
n_epochs=2
server=ls15
# -------------------------------
# Variables
cuda_visible_devices=3
batch_size=256
num_workers=12
seed=47


CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u src/baseline.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=$n_epochs \
    --server=$server \
    --batch_size=$batch_size \
    --num_workers=$num_workers \
    --eval_frequency=1 \
    --input_size=224 \
    --re_size=256 \
    --resume="" \
    --backbone=vgg16 \
    --initialization=pretrained \
    --freeze=True \
    --seed=47 \
    2>&1 | tee $log_file
