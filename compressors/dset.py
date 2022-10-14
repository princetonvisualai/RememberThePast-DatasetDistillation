import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import get_images



''' Real dataset sampler '''
class Net(nn.Module):
    def __init__(self, dset, num_classes, device, batch_size, downsample_scale=1):
        super(Net, self).__init__()
        self.name        = 'dset'
        self.device      = device
        self.batch_size  = batch_size
        self.num_classes = num_classes
        self.downsample_scale = downsample_scale

        self.images_all, self.indices_class = dset

    def forward(self, placeholder=None, task_indices=None, cls=None, new_batch_size=None, no_cuda=False):
        batch_size = self.batch_size if new_batch_size is None else new_batch_size
        if cls is not None:
            assert task_indices is None
            real_imgs = get_images(cls, batch_size, self.images_all, self.indices_class)
            real_labs = torch.ones(batch_size, dtype=torch.long) * cls
        elif task_indices is not None:
            real_imgs = [get_images(c, batch_size, self.images_all, self.indices_class) for c in task_indices]
            real_labs = torch.cat([torch.ones((1, batch_size), dtype=torch.long) * c \
                             for c in task_indices], 0).view(-1)
            real_imgs = torch.cat([ele.unsqueeze(0) for ele in real_imgs], dim=0)
            real_imgs = real_imgs.view([real_imgs.shape[0] * real_imgs.shape[1]] + list(real_imgs.shape[2:])).contiguous()
        else:
            real_imgs = [get_images(c, batch_size, self.images_all, self.indices_class) for c in range(self.num_classes)]
            real_labs = torch.cat([torch.ones((real_imgs[c].shape[0]), dtype=torch.long) * c \
                             for c in range(self.num_classes)], 0)
            real_imgs = torch.cat(real_imgs, dim=0)
            #real_imgs = real_imgs.view([real_imgs.shape[0] * real_imgs.shape[1]] + list(real_imgs.shape[2:])).contiguous()

        if no_cuda:
            return real_imgs, real_labs
        else:
            return real_imgs.to(self.device), real_labs.to(self.device)
