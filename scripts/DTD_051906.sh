set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=DTD
# -------------------------------
epochs=100
server=ls97
# -------------------------------
# Variables
cuda_visible_devices=5
lr=1e-1
seed=47
batch_size=24
initialization="kaiming_normal"
num_workers=4
print_freq=1000
eval_frequency=100
optimizer=SGD


CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -u ./src/ae.py \
    --experiment_index $experiment_index \
    --dataset $dataset \
    --n_epochs $epochs \
    --learning_rate $lr \
    --seed $seed \
    --batch_size $batch_size \
    --initialization $initialization \
    --num_workers $num_workers \
    --print-freq $print_freq\
    --eval_frequency $eval_frequency\
    --optimizer $optimizer\
    --server $server 2>&1 | tee ./saved/logdirs/$experiment_index.log
