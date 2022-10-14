import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils.model_utils import get_time
from utils.data_utils  import TensorDataset


'''
  Trainer: train a model for given epochs and dataset
'''
class Trainer(nn.Module):
    def __init__(
            self,
            net,
            opt,
            compressor,
            test_loader,
            device,
            intervention
        ):
        super(Trainer, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.net = net
        self.opt = opt

        self.compressor    = compressor
        self.test_loader   = test_loader

        self.device = device
        self.intervention = intervention

    def sum(self, inputs):
        output = 0
        for ele in inputs:
            output += ele
        return output

    def combine_basis(self, basis, coefficients):
        N, C, H, W = coefficients.shape[0], basis.shape[1], basis.shape[2], basis.shape[3]
        imgs = torch.matmul(coefficients, basis.view(coefficients.shape[1],-1)).view(N, C, H, W)
        return imgs

    def train_step(self, img, lab):
        self.net.train()

        img = torch.Tensor(img.cpu().data).to(img.device)
        if self.intervention is not None and self.intervention[0] is not None:
            if np.random.rand(1)[0] < self.intervention[1]:
                seed = int(time.time() * 1000) % 100000
                img = self.intervention[0](img, dtype='compressor', seed=seed)
        lab = lab.detach()

        outputs = self.net(img)
        loss    = self.sum([self.criterion(output, lab) for output in outputs])
        acc = np.sum(np.equal(np.argmax(outputs[-1].cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())) / outputs[-1].shape[0]

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), acc


    def test_epoch(self):
        self.net.eval()

        loss_avg, acc_avg, num_exp = 0, 0, 0
        for _, datum in enumerate(self.test_loader):
            img = datum[0].float().to(self.device)
            lab = datum[1].long().to(self.device)
            n_b = lab.shape[0]

            outputs = self.net(img)
            loss    = self.criterion(outputs[-1], lab)
            acc     = np.sum(np.equal(np.argmax(outputs[-1].cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg  += acc
            num_exp  += n_b

        loss_avg /= num_exp
        acc_avg /= num_exp

        return loss_avg, acc_avg


    def lr_step(self):
        for p_idx, param_group in enumerate(self.opt.param_groups):
            self.opt.param_groups[p_idx]['lr'] *= 0.1

    
    def train_rollout(self, steps, dataset, fixed_batch=False):
        losses = []
        acces  = []
        if fixed_batch:
            g_img, g_lab = dataset()
        g_loss, g_acc = 0, 0
        for _ in range(steps):
            if not fixed_batch:
                g_img, g_lab = dataset()
            if isinstance(g_img, list):
                g_img = self.combine_basis(g_img[0], g_img[1])
            if self.compressor.downsample_scale > 1:
                g_img = F.interpolate(g_img, scale_factor=self.compressor.downsample_scale, mode='bilinear', align_corners=False)
            self.train_step(g_img, g_lab)
            g_outputs = self.net(g_img)
            g_loss    = self.sum([self.criterion(g_output, g_lab) for g_output in g_outputs])
            g_acc  = np.sum(
                         np.equal(
                           np.argmax(g_outputs[-1].cpu().data.numpy(), axis=-1),
                           g_lab.cpu().data.numpy()
                       )
                     )
            g_acc /= g_outputs[-1].shape[0]

        return g_loss, g_acc


    def get_dataloader(self, compressor, batch_size):
        if compressor.name == 'dset':
            c_bs = compressor.batch_size
            compressor.batch_size = compressor.images_all.shape[0] # // compressor.num_classes
            imgs, labs = compressor()
            compressor.batch_size = c_bs
        elif compressor.name == 'imgs_embedding_minibatch' or compressor.name == 'imgs_embedding_basis_minibatch':
            n_per_c = compressor.n_per_c
            compressor.n_per_c = compressor.ipc
            imgs, labs, _ = compressor()
            compressor.n_per_c = n_per_c
        else:
            imgs, labs, _ = compressor()
        if isinstance(imgs, list):
            imgs = self.combine_basis(imgs[0], imgs[1])
        if self.compressor.downsample_scale > 1:
            imgs = F.interpolate(imgs, scale_factor=self.compressor.downsample_scale, mode='bilinear', align_corners=False)
        dst_train = TensorDataset(imgs, labs)
        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=0)
        return train_loader


    def train(self, n_epochs, no_scheduler=True):
        if not no_scheduler:
            lr_schedule = [n_epochs//2+1]

        train_loader = self.get_dataloader(self.compressor, batch_size=640)

        loss, acc = 0, 0
        for ep in range(n_epochs):
            if not no_scheduler and ep in lr_schedule:
                self.lr_step()
            for idx, (imgs, labs) in enumerate(train_loader):
                loss, acc = self.train_step(imgs, labs)

        return loss, acc


    def test(self):
        loss, acc = self.test_epoch()
        return loss, acc


    def train_test(self, n_epochs, current_iter, no_scheduler=True, ifprint=True):
        start = time.time()
        opt_lr = self.opt.param_groups[0]['lr']
        loss_train, acc_train = self.train(n_epochs, no_scheduler)
        loss_test, acc_test   = self.test()
        time_train = time.time() - start
        if ifprint:
            print('%s Evaluate_%02d: TotalEpoch = %04d lr = %.6f train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % \
                (get_time(), current_iter, n_epochs, opt_lr, int(time_train), loss_train, acc_train, acc_test))

        return acc_test
