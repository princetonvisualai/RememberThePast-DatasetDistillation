import numpy as np
import torch
import torch.nn.functional as F


# class for intervening on data
class ImageIntervention(object):
    def __init__(self, name, strategy, phase):
        self.name = name
        self.phase = phase
        if self.name in ['compressor_aug', 'real_aug', 'pair_aug']:
            self.functions = {
                'scale': self.diff_scale,
                'flip': self.diff_flip,
                'rotate': self.diff_rotate,
                'crop': self.diff_crop,
                'color': [self.diff_brightness, self.diff_saturation, self.diff_contrast],
                'cutout': self.diff_cutout,
                'mixup': self.mixup,
            }
            self.prob_flip = 0.5
            self.ratio_scale = 1.2
            self.ratio_rotate = 15.0
            self.ratio_crop_pad = 0.125
            self.ratio_cutout = 0.5 # the size would be 0.5x0.5
            self.ratio_noise = 0.05
            self.brightness = 1.0
            self.saturation = 2.0
            self.contrast = 0.5
            self.mixup_alpha = None
            
            self.keys = list(strategy.split('_'))
            for key in self.keys:
                if 'mixup' in key:
                    self.mixup_alpha = float(key[5:])
            
            self.seed = -1
        elif self.name != 'none':
            raise NotImplementedError

    def __call__(self, x, dtype, seed):
        if self.name == 'none':
            return x

        elif self.name == 'compressor_aug':
            if dtype == 'real':
                return x
            elif dtype == 'compressor':
                return self.do(x, seed)
            else:
                raise NotImplementedError

        elif self.name == 'real_aug':
            if dtype == 'compressor':
                return x
            elif dtype == 'real':
                return self.do(x, seed)
            else:
                raise NotImplementedError

        elif self.name == 'pair_aug':
            return self.do(x, seed)

    def do(self, x, seed):
        self.set_seed(seed)
        intervention = self.keys[np.random.randint(0, len(self.keys), size=(1,))[0]]

        if intervention == 'color':
            self.set_seed(seed)
            function = self.functions['color'][np.random.randint(0, len(self.functions['color']), size=(1,))[0]]
        else:
            function = self.functions[intervention]

        self.set_seed(seed)
        intervened_x = function(x)
        self.reset_seed()
        return intervened_x

    def reset_seed(self):
        self.seed = -1

    def update_seed(self):
        self.seed += 1
        #torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)

    def set_seed(self, seed):
        self.seed = seed
        #torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)

    def diff_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale
        #sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        sx = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio)
        self.update_seed()
        #sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        sy = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio)
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
       
    def mixup(self, x, y):
        alpha = self.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
    
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


        lam = np.random.beta()
        randf = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)
       
    def diff_flip(self, x):
        prob = self.prob_flip
        #randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        randf = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)

    def diff_rotate(self, x):
        ratio = self.ratio_rotate
        #theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = torch.Tensor(np.random.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def diff_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        #translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_x = torch.Tensor(np.random.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1])).to(x.device).long()
        self.update_seed()
        #translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.Tensor(np.random.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1])).to(x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def diff_brightness(self, x):
        ratio = self.brightness
        #randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        randb = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x

    def diff_saturation(self, x):
        ratio = self.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        #rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        rands = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def diff_contrast(self, x):
        ratio = self.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        #randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        randc = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def diff_cutout(self, x):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        #offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_x = torch.Tensor(
                np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        self.update_seed()
        #offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.Tensor(
                np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
