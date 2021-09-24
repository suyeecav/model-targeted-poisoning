import torch as ch
import os
import numpy as np
import torch.nn as nn


class CONSTANTS:
    DATA_FOLDER = './data/datasets'
    OUTPUT_FOLDER = './data/outputs/'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def pretty_string(x, color):
    return f"{color}%s{bcolors.ENDC}" % x


def yellow_print(x):
    return pretty_string(x, bcolors.WARNING)


def red_print(x):
    return pretty_string(x, bcolors.FAIL)


def pink_print(x):
    return pretty_string(x, bcolors.HEADER)


def green_print(x):
    return pretty_string(x, bcolors.OKGREEN)


def cyan_print(x):
    return pretty_string(x, bcolors.OKCYAN)


def blue_print(x):
    return pretty_string(x, bcolors.OKBLUE)


def flash_utils(args):
    print(yellow_print("==> Arguments:"))
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))


def get_relevant_params(params, include_bias=True):
    # Filter out bias terms, if asked
    def no_bias(x):
        name, _ = x
        return not name.endswith(".bias")

    params_wanted = params
    if not include_bias:
        params_wanted = filter(no_bias, params)

    # Get only params
    params_wanted = map(lambda x: x[1], params_wanted)

    return params_wanted


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def corruption_rule(num_samples, target_class,
                    n_classes, method='cycle', offset=3):
    if method == 'cycle':
        return ch.ones((num_samples,
                        )).long() * ((target_class + offset) % n_classes)
    elif method == 'random':
        # Pick any class that is not the right class
        candidates = ch.arange(n_classes)
        candidates = candidates[candidates != target_class]
        indices = ch.randint(0, n_classes - 2, (num_samples, ))
        return candidates[indices]
    else:
        raise ValueError("Label corruption method not implemented")


# Given a batch of data, poison p_ratio of samples accordng to
# Specified corruption rule, for given target subpopulation
def poison_data(X, Y, picked, p_ratio, n_classes, c_rule,
                add_to_self, selection=None, save_data=None,
                offset=3):
    num_points = int(p_ratio * len(Y))
    if num_points == 0:
        print("[Warning] Not adding any poison data")
        return X, Y

    # Identify points in target population
    subpop = ch.nonzero(Y == picked).squeeze_(1)
    notsubpop = ch.nonzero(Y != picked).squeeze_(1)

    # Pick indices that should be corrupted
    if selection is None:
        # Sample randomly
        if num_points >= len(subpop):
            pois_inds = np.random.choice(subpop, num_points, replace=True)
        else:
            pois_inds = np.random.choice(subpop, num_points, replace=False)
    else:
        # Indices computed externally
        pois_inds = selection

    random_labels = corruption_rule(
        pois_inds.shape[0], picked, n_classes, method=c_rule, offset=offset)

    # Save data, if requested
    if save_data is not None:
        print(green_print("Saving poison data to %s" % save_data))
        save_x = X[pois_inds].numpy()
        save_y = random_labels.numpy()
        np.savez(os.path.join(save_data, "poison_data"), x=save_x, y=save_y)

    if add_to_self:
        # Add corrupted data to clean data
        X = ch.cat((X, X[pois_inds]))
        Y = ch.cat((Y, random_labels))
    else:
        # In-place label poisoning
        X = ch.cat((X[notsubpop], X[pois_inds]))
        Y = ch.cat((Y[notsubpop], random_labels))

    # Retun data with corruptions
    return X, Y


def get_loss_on_batch(X, Y, model):
    loss_fn = nn.CrossEntropyLoss(reduction='none').cuda()
    loss = loss_fn(model(X.cuda()), Y.cuda())
    return loss


def get_losses(model, loader, relevant_class=None):
    losses = []
    for x, y in loader:
        x_, y_ = x, y
        if relevant_class is not None:
            rel_ind = ch.nonzero(y_ == relevant_class).squeeze_(1)
            x_, y_ = x_[rel_ind], y_[rel_ind]
        loss = get_loss_on_batch(x_, y_, model)
        losses.append(loss)
    losses = ch.cat(losses, 0)
    return losses


def pick_k_lowest_loss(model, ds, k, batch_size):
    # Identify indices for k highest-loss members
    # If data provided directly, use that. Else, get loaders
    if type(ds) == tuple:
        losses = get_loss_on_batch(ds[0], ds[1], model)
    else:
        loader, _ = ds.get_loaders(batch_size, shuffle=False)
        losses = get_losses(model, loader)
    topk = ch.topk(losses, k, largest=True)
    topk_indices = topk.indices.cpu().numpy()
    return topk_indices
