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
epochs=100
server=ls97
# -------------------------------
# Variables
cuda_visible_devices=4,5,7,8,9
model="vgg16"
port=9998
lr=1e-2
seed=47
jobs=4
batch_size=250

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u ./src/main.py \
    --experiment-index $experiment_index \
    -a $model \
    --dist-url 'tcp://127.0.0.1:'$port \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    --epochs $epochs \
    --lr $lr \
    --seed $seed \
    -j $jobs \
    --replace-relu True \
    --batch-size $batch_size \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
