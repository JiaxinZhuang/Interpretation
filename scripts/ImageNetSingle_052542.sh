set -e
set -x
export PYTHONPATH='src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}
dataset=ImageNet
# -------------------------------
class_index=950
num_class=32
server=ls15
delta=0
eval_frequency=10000
# -------------------------------
# Variables
n_epochs=300000

selected_layer=6
selected_filter=19
alpha=1000
beta=1
gamma=100
guidedReLU=False
backbone=vgg16
batch_size=1
seed=-1


for index in `seq 0 4 31`
do
    for cuda_visible_device in `seq 0 3`
    do
        img_index=$[$index+$cuda_visible_device]
        echo $img_index
        CUDA_VISIBLE_DEVICES=$cuda_visible_device python -u src/trainer.py \
            --experiment_index=$experiment_index \
            --server=$server \
            --cuda=0 \
            --dataset=$dataset \
            --n_epochs=$n_epochs \
            --eval_frequency=$eval_frequency \
            --re_size=224 \
            --backbone=$backbone \
            --optimizer=SGD \
            --learning_rate=1e-1 \
            --alpha=$alpha \
            --gamma=$gamma \
            --beta=$beta \
            --delta=$delta \
            --selected_layer=$selected_layer \
            --selected_filter=$selected_filter \
            --resume=037-0 \
            --num_class=$num_class \
            --class_index=$class_index \
            --mode=keep \
            --inter=False \
            --rho=0 \
            --regularization=L1 \
            --smoothing=TotalVariation \
            --regular_ex=3 \
            --delta=$delta \
            --rescale=False \
            --guidedReLU $guidedReLU \
            --img_index $img_index \
            --batch_size $batch_size\
            --seed $seed\
            2>&1 | tee $log_file&
    done
done
wait
echo 'Finish'
