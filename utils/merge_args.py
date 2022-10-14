import math
import torch
from   .Img_per_Cls_budget import cal_ncoef
from   .data_utils         import get_dataset_config


def merge_args(args, config):
    # set dataset
    setattr(config.dataset, 'name', args.dataset)

    # set compressor name
    setattr(config.compressor, 'name', args.compressor)

    # set optimizer option
    setattr(config.backbone_optim,   'optimizer', args.inner_optimizer)
    setattr(config.compressor_optim, 'optimizer', args.compressor_optimizer)
    setattr(config.compressor_optim, 'lr',        args.compressor_lr)

    # set backbone hyper-parameters
    setattr(config.backbone, 'name',        args.backbone_name)
    setattr(config.backbone, 'train_epoch', args.backbone_train_epoch)

    # set memory(bases) addressing formulation hps
    if "basis" in args.compressor and args.fixed_budget > 0:
        channel, im_size, n_class = get_dataset_config(config.dataset.name)
        args.ipc = cal_ncoef(
                       args.ipc,
                       im_size,
                       channel,
                       n_class,
                       args.n_basis,
                       ds=args.downsample_scale,
                   )
        args.ipc = int(math.floor(args.ipc))
        assert args.ipc >= 1
        print('Under fixed budget: ', args.fixed_budget, 'changing IPC to: ', args.ipc)

    args.ipc = int(args.ipc)
    setattr(config.compressor, 'ipc', args.ipc)
    setattr(config.compressor, 'downsample_scale', args.downsample_scale)

    # set intervention
    if hasattr(config, 'intervention'):
        setattr(config.intervention, 'name', args.intervene_name)
        setattr(config.intervention, 'strategy', args.interventions)
        if config.intervention.name == 'pair_match':
            config.intervention.train_name     = 'pair_aug'
            config.intervention.test_name      = 'pair_aug'
            config.intervention.train_strategy = args.interventions
            config.intervention.test_strategy  = args.interventions
        else:
            raise NotImplementedError

    # BPTT
    if args.bptt_inner_steps > -1:
        setattr(config.bptt, 'inner_steps', args.bptt_inner_steps)
    if args.bptt_generalization_batches > -1:
        setattr(config.bptt, 'generalization_batches', args.bptt_generalization_batches)
    if len(args.inner_optimizer) > 0:
        setattr(config.bptt_optim, 'optimizer', args.inner_optimizer)
    if args.inner_momentum >= 0:
        setattr(config.bptt_optim, 'momentum',  args.inner_momentum)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    return config
