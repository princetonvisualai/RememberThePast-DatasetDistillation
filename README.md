# Remember The Past - Dataset Distillation &nbsp;&nbsp;

<img src='docs/Memories2.gif' width=300>
The official implementation of the NeurIPS'22 paper:

> [**Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks**](https://arxiv.org/abs/2206.02916)<br>
> [Zhiwei Deng](https://www.cs.princeton.edu/~zhiweid/), [Olga Russakovsky](https://www.cs.princeton.edu/~olgarus/)<br>
> Princeton University<br>
> NeurIPS 2022

## Highlights:
We highlight two major contributions: (1) memory-addressing formulation; (2) backpropagation through time with momentum on dataset distillation

Through memory-addressing formulation:  1) the size of compressed data does not necessarily grow linearly with the number of classes; 2) an overall higher compression rate with more effective distillation is achieved; and 3) more generalized queries are allowed beyond one-hot labels.

## Installation
```
conda env create -f environment.yml
source activate RememberThePast
```

## Training & testing
Back-propagation through time (BPTT) without memory addressing

```
bash run_scripts/bptt_efficient_compressor_minibatch.sh debug ConvNet compressors_bptt_interventions.yml 1 150 SGD 0.9 CIFAR10 10 10 1 1 none 0
```

Back-propagation through time (BPTT) with memory addressing

```
bash run_scripts/bptt_efficient_compressor_basis_minibatch.sh debug ConvNet compressors_bptt_interventions.yml 1 SGD 150 SGD 0.9 CIFAR10 10 32 16 2 0 1 1 l2 1e-4 none 0
```

To add data augmentations, change none to flip_rotate

The hyperparameters are tuned with validation_ratio=0.1. When set as 0, the training will use the test set directly.

## Reference
```bib
@inproceedings{deng2022remember,
  title={Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks},
  author={Zhiwei Deng and Olga Russakovsky},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
