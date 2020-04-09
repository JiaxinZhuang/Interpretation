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
    --server=ls97 \
    --eval_frequency=100 \
    --re_size=224 \
    --backbone=vgg16 \
    --optimizer=SGD \
    --learning_rate=1e-1 \
    --alpha=100 \
    --gamma=1e-2 \
    --beta=1 \
    --delta=0 \
    --selected_layer=25 \
    --selected_filter=16 \
    --resume=026-449 \
    --num_class=39 \
    --class_index=100 \
    --mode=keep \
    --inter=True \
    --rho=0 \
    --regularization=L1 \
    --smoothing=TotalVariation \
    --regular_ex=2 \
    --batch_size=30 \
    --rescale=True \
    2>&1 | tee $log_file
