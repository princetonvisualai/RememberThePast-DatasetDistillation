import copy
import sys
import torch
import torch.nn as nn
import numpy as np


from utils.model_utils     import get_network
from utils.optimizer_utils import get_optimizer

import torch.nn.functional as F


"""
  Compressor BPTT:
    - params: compressors
    - functions:
        - forward
            args:    task_indices, generalization_batch
            return:  loss
        - inner_loop_forward
        - inner_loop_backward
"""
class CompressorBPTT(nn.Module):
    def __init__(
            self,
            compressor,
            config,
            dset_stats,
            train_intervention,
            backbone_ways,
            coeff_reg='',
            coeff_reg_alpha=0,
        ):
        super(CompressorBPTT, self).__init__()
        self.compressor = compressor
        self.config     = config
        self.channel, self.im_size = dset_stats

        self.loss_func  = self.ce_loss

        self.coeff_reg       = coeff_reg
        self.coeff_reg_alpha = coeff_reg_alpha
        
        self.backbone_ways   = backbone_ways

        self.train_intervention = train_intervention
        self.criterion = nn.CrossEntropyLoss()


    """
      Summation
    """
    def sum(self, inputs):
        output = 0
        for ele in inputs:
            output += ele
        return output


    """
      Flatten tensors
    """
    def flatten(self, data):
        return torch.cat([ele.flatten() for ele in data])


    """
      Get grads and return tensors on cpus
    """
    def get_grads(self, backbone):
        params = list(backbone.parameters())
        params = [ele.grad.detach().cpu() for ele in params]
        return params


    """
      Combine basis
    """
    def combine_basis(self, basis, coefficients):
        N, C, H, W = coefficients.shape[0], basis.shape[1], basis.shape[2], basis.shape[3]
        imgs = torch.matmul(coefficients, basis.view(coefficients.shape[1],-1)).view(N, C, H, W)
        return imgs


    """
      CE loss
    """
    def ce_loss(
            self,
            backbone,
            imgs,
            labs,
        ):
        outputs = backbone(imgs)
        if isinstance(outputs, list):
            loss    = self.sum([self.criterion(output, labs) for output in outputs])
        else:
            loss    = self.criterion(outputs, labs)
        return loss


    """
      forward with inner loops, based on addressing type
    """
    def forward(self, task_indices, generalization_data, intervention_seed, addressing_type='label'):
        backbone = get_network(
                       self.config.backbone.name,
                       self.channel,
                       self.backbone_ways,
                       self.im_size
                   ).to(generalization_data[0].device)

        backbone_opt = get_optimizer(backbone.parameters(), self.config.bptt_optim)

        if addressing_type == 'label':
            loss, dL_dc, dL_dw = self.inner_loop_address_label(
                                     backbone,
                                     backbone_opt,
                                     task_indices,
                                     generalization_data,
                                     seed=intervention_seed,
                                 )
        else:
            raise NotImplementedError

        return loss, dL_dc, dL_dw


    """
      inner loop with label-based addressing (standard dataset distillation)
    """
    def inner_loop_address_label(self, backbone, backbone_opt, task_indices, generalization_data, seed):
        # storing gradients and weights offline, allow reversible
        backbone.zero_grad()
        self.compressor.zero_grad()
        gws, ws, datums, backbone_trained = self.bptt_efficient_forward(
                              backbone,
                              backbone_opt,
                              self.compressor,
                              self.config.bptt.inner_steps,
                              loss_func=self.loss_func,
                              seed=seed,
                              task_indices=task_indices,
                          )

        loss = self.generalization_loss(backbone_trained, generalization_data, seed)

        self.compressor.zero_grad()
        dL_dw = torch.autograd.grad(loss, list(backbone_trained.parameters()))

        dL_dw, dL_dc = self.bptt_efficient_backward(
                           gws,
                           ws,
                           dL_dw,
                           datums,
                           seed,
                           self.config.bptt.inner_steps,
                           lr=self.config.bptt_optim.lr,
                           momentum=self.config.bptt_optim.momentum,
                           loss_func=self.loss_func,
                       )

        return loss, dL_dc, dL_dw


    """
      Generalization loss
    """
    def generalization_loss(self, model, generalization_data, seed):
        imgs, labs = generalization_data
        if self.train_intervention is not None:
            imgs = self.train_intervention(imgs, dtype='real', seed=seed)

        loss = self.loss_func(model, imgs.detach(), labs)

        return loss


    """
      BPTT forward pass. Weights stored on cpus (hack).
    """
    def bptt_efficient_forward(
            self,
            backbone,
            backbone_opt,
            compressor,
            inner_steps,
            loss_func,
            seed,
            task_indices
        ):
        backbone.train()
        ws      = []
        datums  = []
        gws     = []
        for idx in range(inner_steps):
            imgs, labs, img_indices = self.compressor(task_indices=task_indices)
            # Save the datum
            if isinstance(imgs, list):
                datums.append(
                    [[imgs[0].detach().cpu(), imgs[1].detach().cpu()], \
                     labs.detach().cpu(), \
                     img_indices.detach().cpu()]
                )
                imgs = self.combine_basis(imgs[0], imgs[1])
            else:
                datums.append([imgs.detach().cpu(), labs.detach().cpu(), img_indices.detach().cpu()])
            if self.compressor.downsample_scale > 1:
                imgs = F.interpolate(
                           imgs,
                           scale_factor=self.compressor.downsample_scale,
                           mode='bilinear',
                           align_corners=False
                       )
            if self.train_intervention is not None:
                imgs = self.train_intervention(imgs, dtype='compressor', seed=seed)
            loss = loss_func(backbone, imgs, labs)
            backbone_opt.zero_grad()
            # Save the backbone before gradient descent
            ws.append(copy.deepcopy(backbone).cpu())
            loss.backward()
            # Save the gradient after gradient descent
            gws.append(None)
            backbone_opt.step()

        return gws, ws, datums, backbone


    """
      Backward computation of gradients
      Return:
      - dL_dw: grads wrt model weights
      - dL_dc: grads wrt compressors
    """
    def bptt_efficient_backward(
            self,
            gws,
            ws,
            dL_dw,
            datums,
            seed,
            inner_steps,
            lr,
            momentum,
            loss_func,
        ):
        device = dL_dw[0].device
        gdatas = []
        gindices = []
        dL_dv  = [0] * len(dL_dw)
        for (data, label, img_indices), backbone_w, saved_gw in reversed(list(zip(datums, ws, gws))):
            dgw = [lr * ele.neg() for ele in dL_dw]  # gw is already weighted by lr, so simple negation
            dL_dv = [dL_dv_ele + dgw_ele for dL_dv_ele, dgw_ele in zip(dL_dv, dgw)]

            backbone_w.to(device)
            backbone_w.zero_grad()

            if isinstance(data, list):
                data = [data[0].to(device), data[1].to(device)]
                data[0].requires_grad_()
                data[1].requires_grad_()
                scaled_data = self.combine_basis(data[0], data[1])
                hvp_in = data
            else:
                data = data.to(device)
                data.requires_grad_()
                scaled_data = data
                hvp_in = [data]
            if self.compressor.downsample_scale > 1:
                scaled_data = F.interpolate(
                                  scaled_data,
                                  scale_factor=self.compressor.downsample_scale,
                                  mode='bilinear',
                                  align_corners=False
                              )
            if self.train_intervention is not None:
                intervened_data = self.train_intervention(scaled_data, dtype='compressor', seed=seed)
                loss    = loss_func(backbone_w, intervened_data, label.to(device))
            else:
                loss    = loss_func(backbone_w, scaled_data, label.to(device))
            if isinstance(data, list):
                if self.coeff_reg == 'l1':
                    reg = torch.norm(data[1], p=1)
                    loss += self.coeff_reg_alpha * reg
                elif self.coeff_reg == 'l2':
                    reg = torch.norm(data[1], p=2)
                    loss += self.coeff_reg_alpha * torch.norm(data[1])
                else:
                    raise NotImplementedError

            params  = list(backbone_w.parameters())
            hvp_in.extend(params)
            gw = torch.autograd.grad(loss, params, create_graph=True)

            hvp_grad = torch.autograd.grad(
                outputs=(self.flatten(gw),),
                inputs=hvp_in,
                grad_outputs=(self.flatten(dL_dv),),
            )

            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                if isinstance(data, list):
                    grad_tensor = torch.zeros(
                                      [self.compressor.num_classes * self.compressor.ipc] + list(hvp_grad[1].shape[1:]),
                                      device=hvp_grad[1].device
                                  )
                    grad_tensor[img_indices] = hvp_grad[1]
                    if len(gdatas) == 0:
                        gdatas = [hvp_grad[0], grad_tensor]
                    else:
                        gdatas[0] += hvp_grad[0]
                        gdatas[1] += grad_tensor
                    indent = 2
                else:
                    grad_tensor = torch.zeros(
                                      [self.compressor.num_classes * self.compressor.ipc] + list(hvp_grad[0].shape[1:]),
                                      device=hvp_grad[0].device
                                  )
                    grad_tensor[img_indices] = hvp_grad[0]
                    if len(gdatas) == 0:
                        gdatas = grad_tensor
                    else:
                        gdatas += grad_tensor
                    indent = 1
                gindices.append(img_indices)

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                for idx in range(len(dL_dw)):
                    dL_dw[idx].add_(hvp_grad[idx+indent])

            dL_dv = [dL_dv_ele * momentum for dL_dv_ele in dL_dv]

        dL_dc = gdatas

        return dL_dw, dL_dc
