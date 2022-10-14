import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision      import datasets, transforms
import copy


import pickle
from PIL import Image
import os.path as osp

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))


class MiniimagenetDataset(Dataset):
    def __init__(self, setname, transform=None):
        setname = setname
        PICKLE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/')
        pickle_path = osp.join(PICKLE_PATH, 'mini-imagenet-cache-' + setname + '.pkl')
        pickle_data = pickle.load(open(pickle_path, "rb" ))
        self.data, self.targets, self.classes = self.process_pkl(pickle_data)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def process_pkl(self, pickle_data):
        data  = pickle_data['image_data']
        class_dict = pickle_data['class_dict']
        label = [-1] * data.shape[0]
        classnames = [ele for ele in class_dict.keys()]
    
        lbl = 0
        for cls in class_dict:
            for idx in class_dict[cls]:
                label[idx] = lbl
            lbl += 1
        return data, label, classnames

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            img = self.transform(Image.fromarray(img.astype(np.uint8)))
        return img, target


class CIFAR10Dataset(datasets.CIFAR10):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CIFAR100Dataset(datasets.CIFAR100):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SVHNDataset(datasets.SVHN):
    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])


def _split_validation(data, targets, validation_ratio):
    if isinstance(targets, torch.Tensor):
        label_set = set(targets.tolist())
    else:
        label_set = set(targets)
    ttype = 'tensor'
    if isinstance(targets, list):
        ttype   = 'list'
        targets = torch.Tensor(targets)

    train_data_set  = []
    train_label_set = []
    val_data_set    = []
    val_label_set   = []
    for c in label_set:
        data_c    = data[targets==c]
        targets_c = targets[targets==c]

        n_val = int(data_c.shape[0] * validation_ratio)
        n_train = data_c.shape[0] - n_val

        indices = torch.randperm(data_c.shape[0])
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:]

        train_data_c  = data_c[train_indices]
        val_data_c    = data_c[val_indices]
        train_label_c = targets_c[train_indices]
        val_label_c   = targets_c[val_indices]

        train_data_set.append(train_data_c)
        train_label_set.extend(train_label_c.long().tolist())
        val_data_set.append(val_data_c)
        val_label_set.extend(val_label_c.long().tolist())

    train_data_set = torch.cat(train_data_set, dim=0)
    val_data_set   = torch.cat(val_data_set,dim=0)
    if ttype == 'tensor':
        train_label_set = torch.Tensor(train_label_set).long()
        val_label_set   = torch.Tensor(val_label_set).long()
    return train_data_set, train_label_set, val_data_set, val_label_set


def split_validation(dst_train, dst_test, validation_ratio, dataset):
    if dataset in ['SVHN']:
        data = dst_train.data
        targets = dst_train.labels
    else:
        data = dst_train.data
        targets = dst_train.targets

    train_data, train_label, val_data, val_label = _split_validation(data, targets, validation_ratio)

    if dataset in ['SVHN']:
        dst_train.data   = train_data
        dst_train.labels = train_label
        dst_test.data   = val_data
        dst_test.labels = val_label
    else:
        dst_train.data    = train_data
        dst_train.targets = train_label
        dst_test.data    = val_data
        dst_test.targets = val_label

    return dst_train, dst_test

    
def organize_dst(dst_train, num_classes, print_info=True, dset_name='none', split='train', im_size=None):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    data_loaded = None
    if data_loaded is None:
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        images_all = torch.cat(images_all, dim=0)
    else:
        images_all = data_loaded['images_all']
        labels_all = data_loaded['labels_all']
        labels_all = labels_all.tolist()
        print('Total number:', images_all.shape[0])

    for i, lab in enumerate(labels_all):
        if isinstance(lab, list):
            indices_class[lab[-1]].append(i)
        else:
            indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long)

    if print_info:
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
    
        for ch in range(images_all.shape[1]):
            print('real images channel %d, mean = %.4f, std = %.4f'%(
                ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    return images_all, labels_all, indices_class


def get_images(c, n, images_all, indices_class): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    # for debugging
    #idx_shuffle = indices_class[c][:n]
    return images_all[idx_shuffle]


def get_images_labels(c, n, images_all, labels_all, indices_class): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle], labels_all[idx_shuffle]


def select_k_classes(dst, classes, indent):
    data = dst.data
    targets = np.array(dst.targets)
    new_data = []
    new_targets = []
    for c in classes:
        data_c = data[targets==c]
        targets_c = targets[targets==c]
        new_data.append(data_c)
        new_targets.append(targets_c-indent)
    new_data = np.vstack(new_data)
    new_targets = np.hstack(new_targets).tolist()
    return new_data, new_targets


def get_dataset_config(dataset):
    if dataset == 'MNIST':
        channel = 1
        im_size = 28
        n_class = 10

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = 28
        n_class = 10

    elif dataset == 'SVHN':
        channel = 3
        im_size = 32
        n_class = 10

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        n_class = 10

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = 32
        n_class = 100

    elif 'CIFAR100_10' in dataset:
        channel = 3
        im_size = 32
        n_class = 10

    elif 'CIFAR100_5' in dataset:
        channel = 3
        im_size = 32
        n_class = 5

    elif 'CIFAR100_15' in dataset:
        channel = 3
        im_size = 32
        n_class = 15

    elif dataset == 'MiniImagenet':
        channel = 3
        im_size = 84
        n_class = 64

    elif dataset == 'TinyImagenet':
        channel = 3
        im_size = 64
        n_class = 200

    elif dataset == 'TinyImagenetDM':
        channel = 3
        im_size = 64
        n_class = 200

    else:
        exit('unknown dataset: %s'%dataset)

    return channel, im_size, n_class


def get_dataset(dataset, data_path, zca=False, manual_size=None, use_val=True, validation_ratio=0, test_on_train=False):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        train_num_classes = 10
        test_num_classes  = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(train_num_classes)]

    elif dataset == 'rotMNIST-task1':
        channel = 1
        im_size = (28, 28)
        train_num_classes = 10
        test_num_classes  = 10
        mean = [0.1307]
        std = [0.3081]
        #data_loaded = torch.load('data/rmnist_task1.pt')
        train_data_loaded = torch.load('data/rmnist_task5_train.pt')
        test_data_loaded = torch.load('data/rmnist_task5_test.pt')

        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        dst_train.data, dst_train.targets = train_data_loaded['x_train'].detach().cpu(), train_data_loaded['y_train'].detach().cpu()
        dst_test.data, dst_test.targets   = test_data_loaded['x_test'].detach().cpu(), test_data_loaded['y_test'].detach().cpu()

        class_names = [str(c) for c in range(train_num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        train_num_classes = 10
        test_num_classes  = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        train_num_classes = 10
        test_num_classes  = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = SVHNDataset(data_path, split='train', download=True, transform=transform)  # no augmentation
            dst_test = SVHNDataset(data_path, split='test', download=True, transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=100, permute=False)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
            dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(train_num_classes)]
        assert train_num_classes == test_num_classes

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        train_num_classes = 10
        test_num_classes  = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR10Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR10Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR10Half':
        channel = 3
        im_size = (32, 32)
        train_num_classes = 5
        test_num_classes  = 5
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR10Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR10Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        task_id = 0
        dst_train.data, dst_train.targets = select_k_classes(dst_train, list(range(5*task_id, 5*(task_id+1))), indent=5*task_id)
        dst_test.data, dst_test.targets   = select_k_classes(dst_test, list(range(5*task_id, 5*(task_id+1))), indent=5*task_id)
        class_names = dst_train.classes

    elif dataset == 'CIFAR10FSL':
        channel = 3
        im_size = (32, 32)
        train_num_classes = 5
        test_num_classes  = 5
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR10Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR10Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        dst_train.data, dst_train.targets = select_k_classes(dst_train, list(range(5)), indent=0)
        dst_test.data, dst_test.targets   = select_k_classes(dst_test, list(range(5,10)), indent=5)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        train_num_classes = 100
        test_num_classes  = 100
        #num_classes = 5
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR100Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR100Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)

        class_names = dst_train.classes

    elif 'CIFAR100_5' in dataset:
        channel = 3
        im_size = (32, 32)
        train_num_classes = 5
        test_num_classes  = 5
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        task_id = int(dataset.split('_')[-1])
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR100Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR100Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(5*task_id, 5*(task_id+1))),
                                                    indent=5*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(5*task_id, 5*(task_id+1))),
                                                    indent=5*task_id
                                                )
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(5*task_id, 5*(task_id+1))),
                                                    indent=5*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(5*task_id, 5*(task_id+1))),
                                                    indent=5*task_id
                                                )
        class_names = dst_train.classes

    elif 'CIFAR100_15' in dataset:
        channel = 3
        im_size = (32, 32)
        train_num_classes = 15
        test_num_classes  = 15
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        task_id = int(dataset.split('_')[-1])
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR100Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR100Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(15*task_id, 15*(task_id+1))),
                                                    indent=15*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(15*task_id, 15*(task_id+1))),
                                                    indent=15*task_id
                                                )
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(15*task_id, 15*(task_id+1))),
                                                    indent=15*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(15*task_id, 15*(task_id+1))),
                                                    indent=15*task_id
                                                )
        class_names = dst_train.classes

    elif 'CIFAR100_10' in dataset:
        channel = 3
        im_size = (32, 32)
        train_num_classes = 10
        test_num_classes  = 10
        #num_classes = 5
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        task_id = int(dataset.split('_')[-1])
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = CIFAR100Dataset(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = CIFAR100Dataset(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(10*task_id, 10*(task_id+1))),
                                                    indent=10*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(10*task_id, 10*(task_id+1))),
                                                    indent=10*task_id
                                                )
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
            dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
            dst_train.data, dst_train.targets = select_k_classes(
                                                    dst_train,
                                                    list(range(10*task_id, 10*(task_id+1))),
                                                    indent=10*task_id
                                                )
            dst_test.data, dst_test.targets   = select_k_classes(
                                                    dst_test,
                                                    list(range(10*task_id, 10*(task_id+1))),
                                                    indent=10*task_id
                                                )
        class_names = dst_train.classes

    elif dataset == 'MiniImagenet':
        channel = 3
        im_size = (84, 84)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = MiniimagenetDataset(setname='train', transform=transform)
            dst_test  = MiniimagenetDataset(setname='train', transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = MiniimagenetDataset(setname='train', transform=transform)
            dst_test  = MiniimagenetDataset(setname='train', transform=transform)
        class_names = dst_train.classes
        train_num_classes = len(dst_train.classes)
        test_num_classes  = len(dst_test.classes)

    elif dataset == 'MiniImagenetFSL':
        channel = 3
        im_size = (84, 84)
        num_classes = 64
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if zca:
            print('Using ZCA')
            transform = None
            dst_train = MiniimagenetDataset(setname='train', transform=transform)
            dst_test  = MiniimagenetDataset(setname='validation', transform=transform)
            dst_train.data, dst_test.data = preprocess(dst_train.data, dst_test.data, regularization=0.1)
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = MiniimagenetDataset(setname='train', transform=transform)
            dst_test  = MiniimagenetDataset(setname='validation', transform=transform)
        class_names = dst_train.classes
        train_num_classes = len(dst_train.classes)
        test_num_classes  = len(dst_test.classes)

    elif dataset == 'TinyImagenet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if zca:
            raise NotImplementedError
        else:
            transform_seq = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            if manual_size is not None and manual_size > 0:
                im_size = (manual_size, manual_size)
                transform_seq.append(transforms.Resize(manual_size))
            transform = transforms.Compose(transform_seq)
            dst_train = datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform)
            if False and use_val:
                dst_test = datasets.ImageFolder('data/tiny-imagenet-200/val', transform=transform)
            else:
                dst_test = datasets.ImageFolder('data/tiny-imagenet-200/test', transform=transform)
        class_names = dst_train.classes
        train_num_classes = len(dst_train.classes)
        test_num_classes  = len(dst_test.classes)

    elif dataset == 'TinyImagenetDM':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
        train_num_classes = num_classes
        test_num_classes  = num_classes

    else:
        exit('unknown dataset: %s'%dataset)

    if validation_ratio > 0:
        dst_train, dst_test = split_validation(dst_train, dst_test, validation_ratio, dataset)

    if test_on_train:
        dst_test = copy.deepcopy(dst_train)

    return channel, im_size, train_num_classes, test_num_classes, dst_train, dst_test


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def preprocess(train, test, zca_bias=0, regularization=0, permute=True):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')

    nTrain = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]

    trainCovMat = 1.0/nTrain * train.T.dot(train)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E + regularization * np.sum(E) / E.shape[0])
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)

    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)

    train_tensor = torch.Tensor(train.reshape(origTrainShape).astype('float64'))
    test_tensor  = torch.Tensor(test.reshape(origTestShape).astype('float64'))
    if permute:
        train_tensor = train_tensor.permute(0,3,1,2).contiguous()
        test_tensor  = test_tensor.permute(0,3,1,2).contiguous()

    return train_tensor, test_tensor


if __name__ == '__main__':
    results = get_dataset('CIFAR10', 'data', zca=True)
