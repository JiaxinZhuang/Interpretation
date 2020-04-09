set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=CUB


CUDA_VISIBLE_DEVICES=0 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=20000 \
    --server=local \
    --eval_frequency=1000 \
    --re_size=224 \
    --backbone=vgg16 \
    --optimizer=SGD \
    --learning_rate=1 \
    --alpha=1 \
    --selected_layer=0 \
    --selected_filter=3 \
    --resume=008-313 \
    --num_class=30 \
    --class_index=0 \
    --mode=keep \
    2>&1 | tee $log_file
