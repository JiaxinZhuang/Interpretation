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
    --n_epochs=300 \
    --batch_size=1024 \
    --num_workers=4 \
    --server=local \
    --eval_frequency=10 \
    --input_size=28 \
    --re_size=32 \
    --resume=0 \
    --backbone=alexnet \
    2>&1 | tee $log_file
