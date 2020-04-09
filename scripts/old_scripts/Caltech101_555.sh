set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=Caltech101


CUDA_VISIBLE_DEVICES=2 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=50000 \
    --server=ls15 \
    --eval_frequency=100 \
    --re_size=224 \
    --backbone=vgg16 \
    --optimizer=SGD \
    --learning_rate=1e-1 \
    --alpha=100 \
    --beta=1 \
    --selected_layer=25 \
    --selected_filter=7 \
    --resume=011-161 \
    --num_class=30 \
    --class_index=100 \
    --mode=keep \
    --inter=False \
    --rho=0 \
    --regularization=L2 \
    --gamma=1e-1 \
    2>&1 | tee $log_file
