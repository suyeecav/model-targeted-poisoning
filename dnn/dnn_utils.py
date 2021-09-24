import os
import torch as ch
from torch import nn
import numpy as np
from tqdm import tqdm
import copy
import utils
import models


MODEL_MAPPING = {
    "lenet": models.LeNet,
    "flat": models.FlatNet,
    "flat_nodrop": models.FlatNetNoDrop,
    "flat_bn": models.FlatNetBN,
    "flat_bn_nodrop": models.FlatNetBNNoDrop,
    "lr": models.LR
}


# Github Copilot wrote this class with just the class name as prompt!
class EarlyStopper:
    def __init__(self, patience=10, decimal=5):
        self.patience = patience
        self.decimal = decimal
        self.reset()

    def track(self, loss):
        if np.around(loss, self.decimal) >= np.around(self.best_loss, self.decimal):
            self.num_bad_epochs += 1
        else:
            self.num_bad_epochs = 0
            self.best_loss = loss
        if self.num_bad_epochs >= self.patience:
            return True
        return False
    
    def reset(self):
        self.num_bad_epochs = 0
        self.best_loss = np.inf


def get_model_names():
    return list(MODEL_MAPPING.keys())


def model_helper(m_type):
    model = MODEL_MAPPING.get(m_type, None)
    if model is None:
        raise ValueError("Model architecture not implemented yet")
    return model


def get_seeded_wrapped_model(args, n_classes=10):
    model_class = model_helper(args.model_arch)
    if args.seed is not None:
        ch.manual_seed(args.seed)
    model = model_class(n_classes=n_classes)
    model = model.cuda()
    return model


def get_model_metrics(model, loader, target_prop=None, lossfn="ce",
                      specific=None):
    if lossfn == "ce":
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.MultiMarginLoss(reduction='none')

    if specific is None:
        loss_fn = loss_fn.cuda()
    else:
        loss_fn.to(ch.device(specific))

    acc_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    if target_prop is not None:
        noprop_acc_meter = utils.AverageMeter()
        noprop_loss_meter = utils.AverageMeter()

    # Make sure model is in evaluation mode
    model.eval()
    with ch.no_grad():
        for x, y in loader:
            # Shift data to GPU
            if specific is None:
                x, y = x.cuda(), y.cuda()
            else:
                x, y = x.to(ch.device(specific)), y.to(ch.device(specific))
            N = y.size(0)

            logits = model(x)
            preds = ch.argmax(logits, 1)
            loss = loss_fn(logits, y)

            if target_prop is None:
                acc_meter.update(preds.eq(y.view_as(preds)).sum().item(), N)
                loss_meter.update(loss.sum().item(), N)

            else:
                prop_ones = ch.nonzero(y == target_prop).squeeze_(1)
                noprop_ones = ch.nonzero(y != target_prop).squeeze_(1)

                # Pick predictions accordingly
                preds_prop, preds_noprop = preds[prop_ones], preds[noprop_ones]
                # Update metrics on prop data
                if prop_ones.shape[0] > 0:
                    acc_meter.update(preds_prop.eq(
                        y[prop_ones].view_as(preds_prop)).sum().item(),
                        prop_ones.shape[0])
                    loss_meter.update(
                        loss[prop_ones].sum().item(), prop_ones.shape[0])
                # Update metrics on non-prop data
                if noprop_ones.shape[0] > 0:
                    noprop_acc_meter.update(preds_noprop.eq(
                        y[noprop_ones].view_as(preds_noprop)).sum().item(),
                        noprop_ones.shape[0])
                    noprop_loss_meter.update(
                        loss[noprop_ones].sum().item(), noprop_ones.shape[0])

    # Return metrics
    metrics_tuple = (acc_meter.avg, loss_meter.avg)
    if target_prop is None:
        return metrics_tuple
    else:
        return metrics_tuple, (noprop_acc_meter.avg, noprop_loss_meter.avg)


def epoch(model, loader, optimizer, epoch_num, c_rule, n_classes,
          corrupt_class=None, poison_ratio=1.0, verbose=True,
          low_confidence=False, lossfn="ce", specific=None):
    # Check if train or validation mode
    if optimizer is None:
        training = False
    else:
        training = True

    if training:
        model.train()
        mode = "Train"
    else:
        model.eval()
        mode = "Validation"

    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()

    if lossfn == "ce":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MultiMarginLoss()

    if specific is None:
        loss_fn = loss_fn.cuda()
    else:
        loss_fn.to(ch.device(specific))

    iterator = loader
    if verbose:
        iterator = tqdm(loader)

    with ch.set_grad_enabled(training):
        for x, y in iterator:
            # Zero accumulated gradients (if training)
            # Make sure model is in train mode
            if training:
                model.train()
                optimizer.zero_grad()

            # Corrupt batch, if requested
            if corrupt_class is not None:
                pick_indices = None

                # Pick according to highest-loss points, if requested
                if low_confidence:
                    subpop = ch.nonzero(y == corrupt_class).squeeze_(1)

                    num_poison_points = int(poison_ratio * len(y))
                    pick_indices = utils.pick_k_lowest_loss(
                        model, (x[subpop], y[subpop]), num_poison_points, None)

                # Corrupt data for population
                x, y = utils.poison_data(
                    x, y, corrupt_class, poison_ratio,
                    n_classes, c_rule, add_to_self=True,
                    selection=pick_indices)

            # Shift data to GPU
            if specific is None:
                x, y = x.cuda(), y.cuda()
            else:
                x, y = x.to(ch.device(specific)), y.to(ch.device(specific))

            N = y.size(0)
            logits = model(x)
            # Compute loss
            loss = loss_fn(logits, y)
            # Get predictions
            preds = ch.argmax(logits, 1)

            # Back-prop step
            if training:
                loss.backward()
                optimizer.step()

            with ch.no_grad():
                acc_meter.update(preds.eq(y.view_as(preds)).sum().item(), N)
                loss_meter.update(loss.item() * N, N)
                iterator_string = '[%s] Epoch %d, Loss: %.4f, Acc: %.4f' % (
                    mode, epoch_num, loss_meter.avg, acc_meter.avg)

            if verbose:
                if training:
                    pretty_string = utils.cyan_print
                else:
                    pretty_string = utils.yellow_print
                iterator.set_description(
                    pretty_string(iterator_string))

    return (loss_meter.avg, acc_meter.avg)


def train_model(model, loaders, epochs, c_rule,
                n_classes, save_path=None,
                corrupt_class=None,
                lr=1e-3, save_option='last',
                weight_decay=0.09, early_stop=False,
                poison_ratio=1.0, verbose=True,
                no_val=False, low_confidence=False,
                get_metrics_at_epoch_end=None,
                clean_train_loader=None,
                use_plateau_scheduler=False,
                study_mode=False, loss_fn="ce"):
    if save_path is None:
        save_option = 'none'
    if save_option not in ['best', 'last', 'none']:
        raise ValueError("Model-saving mode must be best/last/none")
    if save_option == 'best' and no_val:
        raise ValueError(
            "Cannot identify best-val-loss model if val loss not computed")

    optim = ch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, verbose=True)
    stopper = EarlyStopper(patience=10, decimal=5)

    train_loader, val_loader = loaders

    best_loss, best_vacc = np.inf, 0.0
    best_model = None

    if study_mode: collect_stats = []

    iterator = range(epochs)
    if not verbose:
        iterator = tqdm(iterator)
    for e in iterator:
        # Train epoch
        tr_loss, _ = epoch(model, train_loader, optim, e + 1, c_rule, n_classes,
                           corrupt_class=corrupt_class,
                           poison_ratio=poison_ratio,
                           verbose=verbose, low_confidence=low_confidence,
                           lossfn=loss_fn)

        if not no_val:
            # Validation epoch
            (loss, acc) = epoch(model, val_loader, None, e + 1,
                                c_rule, n_classes, verbose=verbose,
                                lossfn=loss_fn)
        if verbose or study_mode:
            if get_metrics_at_epoch_end is not None:
                (prop_acc, _), (noprop_acc, _) = get_model_metrics(
                    model, clean_train_loader,
                    target_prop=get_metrics_at_epoch_end,
                    lossfn=loss_fn)
                print(utils.yellow_print(
                    "[Train] Population acc: %.4f, Non-population acc: %.4f" %
                    (prop_acc, noprop_acc)))

                (val_prop_acc, _), (val_noprop_acc, _) = get_model_metrics(
                    model, val_loader,
                    target_prop=get_metrics_at_epoch_end,
                    lossfn=loss_fn)
                print(utils.yellow_print(
                    "[Val] Population acc: %.4f, Non-population acc: %.4f" %
                    (val_prop_acc, val_noprop_acc)))

                norm = get_model_l2_norm(model).item()
                print(utils.yellow_print(
                    "[Model] R(w): %.3f" % norm))

                if study_mode:
                    stats = {
                        "train_prop_acc": 100 *prop_acc,
                        "train_noprop_acc": 100 * noprop_acc,
                        "val_prop_acc": 100 * val_prop_acc,
                        "val_noprop_acc": 100 * val_noprop_acc,
                        "norm": norm,
                        # 100 scal for binary, 50 for multiclass
                          # Scaling to visualize better
                        "lossx100": 100 * tr_loss,
                        "lossx50": 50 * tr_loss
                    }
                    collect_stats.append(stats)
            print()

        # Keep track of checkpoint with best validation loss so far
        # If option is picked
        if save_option == 'best':
            if loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss, best_vacc = loss, acc

        # If early stopping, stop training
        if early_stop and stopper.track(tr_loss):
            print("Stopping early, as requested!")
            break
        
        # Take scheduler step, if enables
        if use_plateau_scheduler:
            scheduler.step(tr_loss)

    # Save latest model state, if this option is picked
    if save_option == 'last':
        best_model = model

    if save_option != 'none':
        ch.save(best_model.state_dict(), os.path.join(save_path))

    # Keep track of everything, if asked
    if study_mode:
        return model, best_loss, best_vacc, collect_stats

    # Return best validation metrics
    return model, best_loss, best_vacc


def get_model_l2_norm(model):
    reg_loss = 0
    for param in utils.get_relevant_params(model.named_parameters()):
        reg_loss += ch.sum(param ** 2)

    return reg_loss * 0.5


def model_l2_closeness(a, b, ensemble=False):
    if ensemble:
        diffs = [model_l2_closeness(a, b_) for b_ in b]
        diff_sum = np.mean(diffs)
    else:
        params_a = utils.get_relevant_params(
            a.named_parameters(), include_bias=False)
        params_b = utils.get_relevant_params(
            b.named_parameters(), include_bias=False)

        diff_sum = 0
        for p_a, p_b in zip(params_a, params_b):
            diff_sum += ch.sum((p_a - p_b)**2).item()

    return diff_sum
