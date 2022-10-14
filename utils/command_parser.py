import argparse


def command_parser():
    parser = argparse.ArgumentParser(description='Memory addressing dataset distillation')

    # Utility setup
    parser.add_argument('--seed', type=int, default=2049, \
                        help='Random seed')
    parser.add_argument('--note', type=str, default='', \
                        help='Note for experiment')

    # Training setup
    parser.add_argument('--runner', type=str, default='SGLDRunner', \
                        help='The runner to execute')
    parser.add_argument('--config', type=str, default='compressors.yml',  \
                        help='Path to the config file')
    parser.add_argument('--compressor_lr', type=float, default=0.1, \
                        help='Learning rate to use for training compressor')
    parser.add_argument('--validation_ratio', type=float, default=0, \
                        help='The ratio of using train set as validation set')
    parser.add_argument('--task_sampler_nc', type=int, default=-1, \
                        help='the tasks sampled (number of classes)')
    parser.add_argument('--compressor_optimizer', type=str, default='SGD', \
                        help='which optimizer to use for training compressors')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='CIFAR10', \
                        help='Which dataset to use')
    parser.add_argument('--zca', type=int, default=0, \
                        help='Whether to use ZCA')

    # BPTT setup
    parser.add_argument('--bptt_inner_steps', type=int, default=-1, \
                        help='backpropagation through time inner steps')
    parser.add_argument('--inner_optimizer', type=str, default='', \
                        help='which optimizer to use for bptt inner loop')
    parser.add_argument('--inner_momentum', type=float, default=-1, \
                        help='the momentum for bptt inner loop')
    parser.add_argument('--bptt_generalization_batches', type=int, default=1, \
                        help='Number of batches as generalization loss')
    parser.add_argument('--dset_batch_size', type=int, default=64, \
                        help='Batch size for dataset in BPTT')
    parser.add_argument('--coeff_reg', type=str, default='l2', \
                        help='type of coeff reg')
    parser.add_argument('--coeff_reg_alpha', type=float, default=1e-4, \
                        help='the coeff of coeff reg')

    # Interventions setup
    parser.add_argument('--interventions', type=str, default='', \
                        help='interventions on memories/data')
    parser.add_argument('--intervene_name', type=str, default='pair_match', \
                        help='the type of intervention framework')

    # Model setup
    parser.add_argument('--compressor', type=str, default='imgs_embedding_minibatch', \
                        help='Which compressor model to use')
    parser.add_argument('--downsample_scale', type=float, default=1, \
                        help='Downsample scales for compressor imgs')
    parser.add_argument('--n_basis', type=int, default=0, \
                        help='Number of basis')
    parser.add_argument('--fixed_budget', type=int, default=0, \
                        help='Whether to use fixed budget for memory addressing formulation')
    parser.add_argument('--compressor_minibatch_size', type=int, default=-1, \
                        help='Mini-batch size for compressor')
    parser.add_argument('--ipc', type=float, default=50, \
                        help='Number of distilled images (or as budget)')
    parser.add_argument('--backbone_name', type=str, default='', \
                        help='Which backbone model to use')
    parser.add_argument('--backbone_train_epoch', type=int, default=-1, \
                        help='How many epochs to train the backbone when evaluating compressors')

    return parser
