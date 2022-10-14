import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' Compressor generator '''
class Net(nn.Module):
    def __init__(
            self,
            ipc,
            channel,
            im_size,
            num_classes,
            device,
            n_basis,
            downsample_scale=1,
            n_per_c=None,
            real_init=None
        ):
        super(Net, self).__init__()
        self.name    = 'imgs_embedding_basis_minibatch'
        self.n_basis = n_basis
        self.ipc     = ipc
        self.n_per_c = n_per_c
        self.channel = channel
        self.im_size = [int(ele / downsample_scale) for ele in im_size]
        self.downsample_scale = downsample_scale
        self.num_classes = num_classes
        self.device      = device

        self.prev_imgs   = None

        self.basis = nn.Embedding(self.n_basis, channel*np.prod(self.im_size))
        self.imgs  = nn.Embedding(ipc*num_classes, self.n_basis)
        self.labs  = torch.tensor(
                         [np.ones(self.ipc)*i for i in range(self.num_classes)],
                         requires_grad=False,
                         device=self.device
                     ).long().view(-1)

        torch.nn.init.xavier_uniform(self.imgs.weight)
        torch.nn.init.xavier_uniform(self.basis.weight)

    def get_compressors(self):
        imgs, _, _ = self.forward(combine=True, new_batch_size=self.ipc)
        imgs = F.interpolate(
                   imgs,
                   scale_factor=self.downsample_scale,
                   mode='bilinear',
                   align_corners=False
               )

        return imgs

    def get_basis(self):
        basis = self.basis(torch.arange(self.n_basis).to(self.device))
        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()

        return basis

    def combine_basis(self, basis, coefficients):
        N, C, H, W = coefficients.shape[0], basis.shape[1], basis.shape[2], basis.shape[3]
        imgs       = torch.matmul(coefficients, basis.view(coefficients.shape[1],-1)).view(N, C, H, W)
        return imgs

    def assign_grads(self, grads, task_indices):
        assert isinstance(grads, list)
        basis_grads, imgs_grads = grads
        self.basis.weight.grad  = basis_grads.to(self.basis.weight.data.device).view(self.basis.weight.shape)
        self.imgs.weight.grad   = imgs_grads.to(self.imgs.weight.data.device).view(self.imgs.weight.shape)

    def get_coeffs_min_max(self):
        indices = torch.randperm(self.ipc*self.num_classes).to(self.device)
        coeffs  = self.imgs(indices)
        return coeffs.min().item(), coeffs.max().item()

    def get_coeffs_per_class(self):
        indices = torch.arange(self.ipc*self.num_classes).to(self.device)
        coeffs  = self.imgs(indices)
        coeffs  = coeffs.view(self.num_classes, self.ipc, coeffs.shape[-1])
        return coeffs

    def get_min_max(self):
        indices = torch.randperm(self.ipc*self.num_classes).to(self.device)
        imgs    = self.imgs(indices)

        basis = self.basis(torch.arange(self.n_basis).to(self.device))

        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()

        imgs  = self.combine_basis(basis, imgs)
        return imgs.min().item(), imgs.max().item()

    def forward(self, placeholder=None, task_indices=None, combine=False, new_batch_size=None):
        if task_indices is None:
            task_indices = list(range(self.num_classes))
        assert isinstance(task_indices, list)
        indices = []
        for i in task_indices:
            if new_batch_size is None:
                ind = torch.randperm(self.ipc)[:self.n_per_c].sort()[0] + self.ipc * i
            else:
                ind = torch.randperm(self.ipc)[:new_batch_size].sort()[0] + self.ipc * i
            indices.append(ind)

        basis = self.basis(torch.arange(self.n_basis).to(self.device))

        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()

        indices = torch.cat(indices).to(self.device)
        imgs    = self.imgs(indices)
        labs    = self.labs[indices]

        if combine:
            imgs = self.combine_basis(basis, imgs)
            return imgs, labs, indices
        else:
            return [basis, imgs], labs, indices
