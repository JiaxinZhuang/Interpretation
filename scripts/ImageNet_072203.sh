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
epochs=50
server=ls97
# -------------------------------
# Variables
cuda_visible_devices=1
model="vgg11"
port=9998
lr=1e-2
seed=-1
batch_size=128
initialization="kaiming_normal"
resume='None'
num_workers=4
dali=False
re_size=256
input_size=224
conv_bias=True
linear_bias=True
eval_frequency=10


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
    --conv_bias $conv_bias\
    --linear_bias $linear_bias\
    --eval_frequency $eval_frequency\
    --avg \
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
