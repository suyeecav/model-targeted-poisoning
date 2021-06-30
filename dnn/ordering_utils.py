import torch as ch
import torch.nn as nn
import copy
import numpy as np
from dnn_utils import get_model_metrics
from tqdm import tqdm

# "too-many-handles" error for small BS
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def find_optimal_batch_order(model, train_loader, eval_loader, lr, weight_decay, poison_class):
    """
    Assuming white-box access to a model, return ordering of
    datap batches that would get the model closest to desired
    attacker objective.
    """
    print("Reordering batches")
    batch_losses = []
    batch_data = []
    loss_fn = nn.CrossEntropyLoss()
    for (x, y) in tqdm(train_loader):
        
        # Create temporary clone of model
        model_ = copy.deepcopy(model)

        # Create temporary optimizer
        optim = ch.optim.Adam(model_.parameters(), lr=lr,
                              weight_decay=weight_decay)
        model_.train()
        optim.zero_grad()

        batch_data.append((x, y))
        
        # Simulate training on only this batch of data
        x, y = x.cuda(), y.cuda()
        logits = model_(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        # Record loss on target sub-population
        (_, prop_loss), (_, noprop_loss) = get_model_metrics(
            model_, eval_loader, target_prop=poison_class)
        batch_losses.append(prop_loss)
        
    batch_losses = np.array(batch_losses)

    # Oscillating out-in
    # order_ = np.argsort(batch_losses)
    # o1 = list(order_[len(order_)//2:][::-1])
    # o2 = list(order_[:len(order_)//2])
    # order = np.empty((len(order_),), dtype=int)
    # order[0::2] = o1
    # order[1::2] = o2

    # Oscillating in-out
    # order = order[::-1]

    # Low->High setting
    order = np.argsort(batch_losses)

    # High->Low setting
    # order = np.argsort(-batch_losses)

    # Create new loader with this order of batches
    new_loader = np.array(batch_data, dtype=object)
    new_loader = new_loader[order]

    return new_loader
