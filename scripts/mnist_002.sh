set -e
set -x
export PYTHONPATH='src'
#log_dir="./saved/logs"

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=mnist


CUDA_VISIBLE_DEVICES=0 python -u src/baseline.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=$dataset \
    --n_epochs=400 \
    --batch_size=1024 \
    --num_workers=4 \
    --server=LS15 \
    --eval_frequency=2 \
    --input_size=28 \
    --re_size=32 \
    --resume=0 \
    --backbone=convNet \
    --optimizer=SGD \
    --learning_rate=1e-3 \
    2>&1 | tee $log_file