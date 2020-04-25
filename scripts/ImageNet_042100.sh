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
epochs=1
server=ls15
# -------------------------------
# Variables
cuda_visible_devices=1
model="vgg16"
port=9998
lr=1e-2
seed=47
batch_size=300
initialization="pretrained"
resume='None'
num_workers=4
dali=False

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u ./src/baseline.py \
    --experiment_index $experiment_index \
    --dataset $dataset \
    --backbone $model \
    --n_epochs $epochs \
    --learning_rate $lr \
    --seed $seed \
    --batch_size $batch_size \
	--freeze True \
    --initialization $initialization \
    --resume $resume \
    --num_workers $num_workers \
    --dali $dali \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
