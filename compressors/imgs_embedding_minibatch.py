import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' Compressors generator '''
class Net(nn.Module):
    def __init__(
            self,
            ipc,
            channel,
            im_size,
            num_classes,
            device,
            downsample_scale=1,
            norm=False,
            n_basis=None,
            n_per_c=10,
        ):
        super(Net, self).__init__()
        self.name = 'imgs_embedding_minibatch'
        self.ipc = ipc
        self.channel = channel
        self.n_per_c = n_per_c
        self.im_size = [int(ele / downsample_scale) for ele in im_size]
        self.downsample_scale = downsample_scale
        self.num_classes = num_classes
        self.device = device

        self.prev_imgs = None

        self.imgs = nn.Embedding(ipc*num_classes, channel*np.prod(self.im_size))
        self.labs = torch.tensor(
                        [np.ones(self.ipc)*i for i in range(self.num_classes)],
                        requires_grad=False,
                        device=self.device
                    ).long().view(-1)

        torch.nn.init.xavier_uniform(self.imgs.weight)

    def assign_grads(self, grads, task_indices):
        self.imgs.weight.grad = grads.to(self.imgs.weight.data.device).view(self.imgs.weight.shape)

    def get_min_max(self):
        indices = torch.randperm(self.ipc*self.num_classes).to(self.device)
        imgs = self.imgs(indices)
        return imgs.min().item(), imgs.max().item()

    def get_compressors(self):
        indices = []
        for i in range(self.num_classes):
            ind = torch.arange(self.ipc) + self.ipc * i
            indices.append(ind)

        indices = torch.cat(indices).to(self.device)
        imgs    = self.imgs(indices)
        imgs = imgs.view(
                   self.num_classes * self.ipc,
                   self.channel,
                   self.im_size[0],
                   self.im_size[1]
               ).contiguous()

        return imgs

    def forward(self, placeholder=None, task_indices=None, combine=None):
        if task_indices is None:
            task_indices = list(range(self.num_classes))
        assert isinstance(task_indices, list)
        indices = []
        for i in task_indices:
            ind = torch.randperm(self.ipc)[:self.n_per_c].sort()[0] + self.ipc * i
            indices.append(ind)

        indices = torch.cat(indices).to(self.device)
        imgs    = self.imgs(indices)
        imgs = imgs.view(
                   len(task_indices) * min(self.n_per_c, self.ipc),
                   self.channel,
                   self.im_size[0],
                   self.im_size[1]
               ).contiguous()

        labs    = self.labs[indices]

        return imgs, labs, indices
