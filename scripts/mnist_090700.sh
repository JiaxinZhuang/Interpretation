set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=mnist
# -------------------------------
epochs=10
server=desktop
# -------------------------------
# Variables
cuda_visible_devices=0
model="vgg11"
port=9998
lr=1e-3
seed=-1
batch_size=64
initialization="kaiming_normal"
resume='None'
num_workers=4
dali=False
re_size=32
input_size=32
eval_frequency=1

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u ./src/baseline.py \
    --experiment_index $experiment_index \
    --dataset $dataset \
    --backbone $model \
    --n_epochs $epochs \
    --learning_rate $lr \
    --seed $seed \
    --batch_size $batch_size \
    --freeze False \
    --initialization $initialization \
    --resume $resume \
    --num_workers $num_workers \
    --dali $dali \
    --re_size $re_size\
    --input_size $input_size\
    --eval_frequency $eval_frequency\
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
