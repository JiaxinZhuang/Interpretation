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
server=ls15
# -------------------------------
# Variables
cuda_visible_devices=0,1,2,3
model="vgg16"
lr=1e-2
seed=-1
batch_size=248
initialization="default"
resume='None'
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
    --world_size $world_size \
    --distributed $distributed \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
