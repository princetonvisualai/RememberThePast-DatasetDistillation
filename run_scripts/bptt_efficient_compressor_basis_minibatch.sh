#!/bin/bash

note=$1
backbone=$2
config=$3
ipc=$4
optimizer=$5
bptt_inner_steps=$6
inner_optimizer=$7
inner_momentum=$8
dataset=${9}
task_sampler_nc=${10}
minibatch_size=${11}
n_basis=${12}
ds_scale=${13}
zca=${14}
fixed_budget=${15}
validation_ratio=${16}
reg=${17}
reg_alpha=${18}
interventions=${19}
val_ratio=${20}

compressor=imgs_embedding_basis_minibatch

root_dir='.'
LOG="${root_dir}/logs/${note}/${dataset}/${backbone}/${ipc}/compressor-${compressor}_opt-${optimizer}_bptt-steps-${bptt_inner_steps}_in-optim-${inner_optimizer}-momentum-${inner_momentum}_task-sampler-${task_sampler_nc}_minibs-${minibatch_size}_nbasis-${n_basis}_ds-scale-${ds_scale}_zca-${zca}_fixed-budget-${fixed_budget}_val-ratio-${validation_ratio}_reg-${reg}-${reg_alpha}_interventions-${interventions}_val-ratio-${val_ratio}.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
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
    --n_basis ${n_basis} \
    --compressor ${compressor} \
    --downsample_scale ${ds_scale} \
    --zca ${zca} \
    --validation_ratio ${validation_ratio} \
    --fixed_budget ${fixed_budget} \
    --coeff_reg ${reg} \
    --coeff_reg_alpha ${reg_alpha} \
    --interventions ${interventions} \
    --validation_ratio ${val_ratio}
