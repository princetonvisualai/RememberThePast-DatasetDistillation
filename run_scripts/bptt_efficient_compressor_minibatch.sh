#!/bin/bash

note=$1
backbone=$2
config=$3
ipc=$4
bptt_inner_steps=$5
inner_optimizer=$6
inner_momentum=$7
dataset=${8}
task_sampler_nc=${9}
minibatch_size=${10}
ds_scale=${11}
zca=${12}
interventions=${13}
val_ratio=${14}

compressor=imgs_embedding_minibatch

root_dir='.'
LOG="${root_dir}/logs/${note}/${dataset}/${backbone}/${ipc}/compressor-${compressor}_bptt-steps-${bptt_inner_steps}_in-optim-${inner_optimizer}-momentum-${inner_momentum}_task-sampler-${task_sampler_nc}_minibs-${minibatch_size}_ds-scale-${ds_scale}_zca-${zca}_interventions-${interventions}_val-ratio-${val_ratio}.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
mkdir -p "${root_dir}/logs/${note}/${dataset}/${backbone}/${ipc}"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

hostname

python srcs/main.py \
    --note ${note} \
    --backbone_name ${backbone} \
    --config ${config} \
    --ipc ${ipc} \
    --bptt_inner_steps ${bptt_inner_steps} \
    --inner_optimizer ${inner_optimizer} \
    --inner_momentum ${inner_momentum} \
    --dataset ${dataset} \
    --task_sampler_nc ${task_sampler_nc} \
    --compressor_minibatch_size ${minibatch_size} \
    --compressor ${compressor} \
    --downsample_scale ${ds_scale} \
    --zca ${zca} \
    --interventions ${interventions} \
    --validation_ratio ${val_ratio}
