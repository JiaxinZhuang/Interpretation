# Remove conv & linear bias

set -e
set -x
export PYTHONPATH='src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=ImageNet
# -------------------------------
epochs=15
server=ls16
# -------------------------------
# Variables
cuda_visible_devices=4,5,6,7
model="vgg16"
lr=1e-4
seed=-1
batch_size=248
initialization="pretrained"
resume='None'
world_size=4
distributed=True
dist_url="tcp://127.0.0.1:25000"
cuda=0

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
    --dist_url $dist_url \
    --cuda $cuda \
    --conv_bias False \
    --linear_bias False \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
