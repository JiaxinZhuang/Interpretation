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
epochs=80
server=ls15
# -------------------------------
# Variables
cuda_visible_devices=3,4,6,7
model="vgg16"
port=9998
lr=1e-2
seed=-1
batch_size=256
initialization="pretrained"
resume='None'
num_workers=16
local_rank=0
world_size=4
distributed=True

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u ./src/baseline_dali.py \
    --experiment_index $experiment_index \
    --dataset $dataset \
    --backbone $model \
    --n_epochs $epochs \
    --learning_rate $lr \
    --seed $seed \
    --batch_size $batch_size \
    --initialization $initialization \
    --resume $resume \
    --num_workers $num_workers \
    --local_rank $local_rank \
    --world_size $world_size \
    --distributed $distributed \
    --prof \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
