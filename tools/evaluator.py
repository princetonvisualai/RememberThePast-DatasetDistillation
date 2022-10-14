import sys
import torch
import numpy as np
import copy

from utils.model_utils     import get_network
from utils.optimizer_utils import get_optimizer

from tools.trainer         import Trainer


"""
  An Evaluator class:
    - evaluates the performance of a compressor instance with a \
            test_loader through training for K epochs on the compressors
    - training is through building a Trainer instance
"""
class Evaluator(object):
    def __init__(
            self,
            config,
            dset_stats,
            eval_models,
            test_loader,
            rank,
        ):

        # configs
        self.config = config
        self.channel, self.num_classes, self.im_size = dset_stats
        self.eval_models = eval_models
        self.rank = rank
    
        # query set generator for evaluation
        self.test_loader = test_loader


    def _evaluate(
            self,
            backbone,
            compressor,
            current_iter,
            train_epoch,
            intervention,
        ):

        optimizer = get_optimizer(
                        backbone.parameters(),
                        self.config.backbone_optim,
                    )

        # trainer for net and real data
        trainer = Trainer(
                      backbone,
                      optimizer,
                      compressor=copy.deepcopy(compressor),
                      test_loader=self.test_loader,
                      device=self.config.device,
                      intervention=intervention,
                  )

        test_acc = trainer.train_test(
                       train_epoch,
                       current_iter=current_iter,
                       no_scheduler=True,
                   )

        sys.stdout.flush()

        return test_acc, trainer.net


    def evaluate(
            self,
            compressors,
            current_iter,
            train_epoch,
            num_eval,
            intervention=None,
        ):
        # record performance
        accs_all_exps = dict()
        for key in self.eval_models:
            accs_all_exps[key] = []

        # evaluate compressor
        for idx in range(len(compressors)):
            compressors[idx].eval()

        for model_name in self.eval_models:
            accs = []
            print('Evaluate model ', model_name)
            for _ in range(num_eval):
                backbone = get_network(
                               model_name,
                               self.channel,
                               self.num_classes,
                               self.im_size
                           ).to(self.rank)

                print('*')
                for idx in range(0,len(compressors)):
                    test_acc, backbone = self._evaluate(
                                             backbone,
                                             compressors[idx],
                                             current_iter,
                                             train_epoch,
                                             intervention,
                                         )
                accs += [test_acc]

            print('*')
            print(('FSL Evaluate %d random %s, mean = %.4f std = %.4f\n' + '-'*12)%(
                   len(accs), model_name, np.mean(accs), np.std(accs))
            )
            accs_all_exps[model_name] += accs
            sys.stdout.flush()

        return accs_all_exps
