import os
import errno
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch as ch
from tqdm import tqdm
import numpy as np
import torch.autograd as autograd
from torchvision import transforms
import utils


class GenericDataWrapper:
    def __init__(self, train, val, datum_shape, n_classes):
        self.train = train
        self.val = val
        self.n_classes = n_classes
        self.datum_shape = datum_shape

    def poison_train_data(self, picked, p_ratio, c_rule,
                          add_to_self=True, selection=None,
                          save_data=None, offset=3):
        self.train.data, self.train.targets = utils.poison_data(
            X=self.train.data, Y=self.train.targets,
            picked=picked, p_ratio=p_ratio,
            n_classes=self.n_classes, c_rule=c_rule,
            add_to_self=add_to_self, selection=selection,
            save_data=save_data, offset=offset
        )

    def get_loaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.train,
                                  num_workers=8,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
        val_loader = DataLoader(self.val,
                                num_workers=8,
                                batch_size=batch_size,
                                shuffle=shuffle)
        return train_loader, val_loader

    def add_point_to_train(self, x, y):
        # Convert back to [0, 255], since this data is
        # Pre-transform data
        if len(self.train.data.shape) == 3:
            x_add = (x[0] * 255).type(ch.uint8)
        else:
            x_add = (x * 255).type(ch.uint8)

        self.train.data = ch.cat(
            (self.train.train_data(), x_add), 0)
        self.train.targets = ch.cat(
            (self.train.train_labels(), y), 0)

    def add_poison_data(self, X, Y):
        X_ = ch.from_numpy(X)
        X_ = (X_ * 255).type(ch.uint8)
        Y_ = ch.from_numpy(Y)

        self.train.data = ch.cat((self.train.data, X_), 0)
        self.train.targets = ch.cat((self.train.targets, Y_), 0)


class MNISTWrapper(GenericDataWrapper):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train = MNIST(utils.CONSTANTS.DATA_FOLDER,
                      download=True,
                      transform=transform)
        val = MNIST(utils.CONSTANTS.DATA_FOLDER,
                    train=False,
                    download=True,
                    transform=transform)

        super(MNISTWrapper, self).__init__(train, val, (1, 28, 28), 10)


class MNIST17Wrapper(GenericDataWrapper):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train = MNIST(utils.CONSTANTS.DATA_FOLDER,
                      download=True,
                      transform=transform)
        val = MNIST(utils.CONSTANTS.DATA_FOLDER,
                    train=False,
                    download=True,
                    transform=transform)

        # Retain only classes 1-7
        def filter_process_data(x, y):
            wanted = ch.logical_or(y == 1, y == 7)
            x, y = x[wanted], y[wanted]
            y = 1 * (y == 7)
            return x, y

        # Process data to apply selection filters
        train.data, train.targets = filter_process_data(
            train.data, train.targets)
        val.data, val.targets = filter_process_data(
            val.data, val.targets)

        super(MNIST17Wrapper, self).__init__(train, val, (1, 28, 28), 2)


class DictToDS:
    def __init__(self, D):
        # Add '1' dimension in second place, if neded
        self.data = D["data"]
        if len(self.data.shape) <= 3:
            self.data = ch.unsqueeze(self.data, 1)
        self.targets = D["targets"]
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.float() / 255
        return img, target

    def __len__(self):
        return len(self.targets)

    def train_data(self):
        return self.data

    def train_labels(self):
        return self.targets


class MemoryDataset(GenericDataWrapper):
    def __init__(self, transform=None, path=None, n_classes=2):
        if transform is None:
            transform = transforms.ToTensor()

        if path is None:
            raise ValueError("Path to data not provided!")

        data = ch.load(path)
        train = DictToDS(data["train"])
        val = DictToDS(data["val"])
        datum_shape = train.data.shape[1:]
        # Read from memory
        super(MemoryDataset, self).__init__(train, val, datum_shape, n_classes)


class MNIST17FirstWrapper(MemoryDataset):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        path = "./data/datasets/MNIST17/split_1.pt"
        super(MNIST17FirstWrapper, self).__init__(transform, path, 2)


class MNIST17SecondWrapper(MemoryDataset):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        path = "./data/datasets/MNIST17/split_2.pt"
        super(MNIST17FirstWrapper, self).__init__(transform, path, 2)


class MNISTEvenWrapper(GenericDataWrapper):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train = MNIST(utils.CONSTANTS.DATA_FOLDER,
                      download=True,
                      transform=transform)
        val = MNIST(utils.CONSTANTS.DATA_FOLDER,
                    train=False,
                    download=True,
                    transform=transform)

        # Retain only even digits
        def filter_process_data(x, y):
            wanted = ch.where(y % 2 == 0)[0]
            x, y = x[wanted], y[wanted]
            y = y // 2
            return x, y

        # Process data to apply selection filters
        train.data, train.targets = filter_process_data(
            train.data, train.targets)
        val.data, val.targets = filter_process_data(
            val.data, val.targets)

        super(MNISTEvenWrapper, self).__init__(train, val, (1, 28, 28), 5)


class FMNISTWrapper(GenericDataWrapper):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train = FashionMNIST(utils.CONSTANTS.DATA_FOLDER,
                      download=True,
                      transform=transform)
        val = FashionMNIST(utils.CONSTANTS.DATA_FOLDER,
                    train=False,
                    download=True,
                    transform=transform)

        super(FMNISTWrapper, self).__init__(train, val, (1, 28, 28), 10)


class FMNISTShirtShoeWrapper(GenericDataWrapper):
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train = FashionMNIST(utils.CONSTANTS.DATA_FOLDER,
                             download=True,
                             transform=transform)
        val = FashionMNIST(utils.CONSTANTS.DATA_FOLDER,
                           train=False,
                           download=True,
                           transform=transform)
        
        # Shirt & T-shirts v/s Sandal & Sneaker
        def filter_process_data(x, y):
            c0 = ch.logical_or(y == 0, y == 6)
            c1 = ch.logical_or(y == 5, y == 7)
            wanted = ch.logical_or(c0, c1)
            x, y = x[wanted], y[wanted]
            y = 1 * np.logical_or(y == 5, y == 7)
            return x, y

        # Process data to apply selection filters
        train.data, train.targets = filter_process_data(
            train.data, train.targets)
        val.data, val.targets = filter_process_data(
            val.data, val.targets)

        super(FMNISTShirtShoeWrapper, self).__init__(train, val, (1, 28, 28), 2)


DATASET_MAPPING = {
    "mnist": MNISTWrapper,
    "mnist17": MNIST17Wrapper,
    "mnist17_first": MNIST17FirstWrapper,
    "mnist17_second": MNIST17SecondWrapper,
    "memory": MemoryDataset,
    "mnisteven": MNISTEvenWrapper,
    "fmnist": FMNISTWrapper,
    "fmnistss": FMNISTShirtShoeWrapper
}


def get_dataset_names():
    return list(DATASET_MAPPING.keys())


def dataset_helper(d_type):
    ds = DATASET_MAPPING.get(d_type, None)
    if ds is None:
        raise ValueError("Model architecture not implemented yet")
    return ds


def safe_makedirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def get_dataset_gradients(model, ds, batch_size, weight_decay,
                          verbose=False, is_train=True):
    # Make sure model is in eval model
    model.eval()

    gradients = []
    # Get specific data oaders
    train_loader, val_loader = ds.get_loaders(batch_size, shuffle=False)
    if is_train:
        loader = train_loader
    else:
        loader = val_loader

    # Define CE Loss
    ce_loss = nn.CrossEntropyLoss(reduction='sum').cuda()
    # Get L2 regularization loss for model
    # l2_reg = dnn_utils.get_model_l2_norm(model)

    if verbose:
        loader = tqdm(loader)
    for x, y in loader:
        x, y = x.cuda(), y.cuda()

        # Zero out existing gradients
        model.zero_grad()

        # Compute NLL loss
        preds = model(x)
        total_loss = ce_loss(preds, y)
        # Add L2 term
        # total_loss += weight_decay * (y.shape[0] * l2_reg)

        # Compute gradients
        grads = autograd.grad(
            total_loss, utils.get_relevant_params(model.named_parameters()))

        # Accumulate gradients
        if len(gradients) == 0:
            gradients = [x.clone().detach() for x in grads]
        else:
            for i, gd in enumerate(grads):
                gradients[i] += gd.clone().detach()

    # Negate sum, since they represent virtual GD runs
    for i in range(len(gradients)):
        gradients[i] *= -1

    return gradients


def get_sample_from_loader(loader, n_trials, n_classes):
    # Make sure there are n_trials datapoints
    # Per class
    data_x = [[] for _ in range(n_classes)]
    data_y = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes)
    for x, y in loader:
        for i in range(n_classes):
            pick = (y == i)
            data_x[i].append(x[pick])
            data_y[i].append(y[pick])
            counts[i] += ch.sum(pick).item()
        if np.all(counts >= n_trials):
            break

    picked_x, picked_y = [], []
    for i in range(n_classes):
        X = ch.cat(data_x[i], 0)
        picked_x.append(X[:n_trials])
        Y = ch.cat(data_y[i], 0)
        picked_y.append(Y[:n_trials])

    picked_x = ch.cat(picked_x, 0)
    picked_y = ch.cat(picked_y, 0)

    return (picked_x, picked_y)
