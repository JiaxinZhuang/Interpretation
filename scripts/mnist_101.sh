set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=mnist


CUDA_VISIBLE_DEVICES=0 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=50000 \
    --server=LS15 \
    --eval_frequency=100 \
    --input_size=28 \
    --backbone=convNet \
    --optimizer=SGD \
    --learning_rate=1 \
    --alpha=10 \
    --selected_layer=5 \
    --selected_filter=25 \
    --resume=001-215 \
    2>&1 | tee $log_file
