import argparse
from dnn_utils import get_model_metrics, epoch, get_model_l2_norm, get_model_names, get_seeded_wrapped_model
import torch as ch
import copy
import numpy as np
import os
import utils
import datasets
from tqdm import tqdm
from ordering_utils import find_optimal_batch_order

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def train_model(model, loaders, epochs, c_rule,
                n_classes, save_path=None,
                corrupt_class=None,
                lr=1e-3, save_option='last',
                weight_decay=0.09,
                poison_ratio=1.0, verbose=True,
                no_val=False, get_metrics_at_epoch_end=None,
                clean_train_loader=None,
                study_mode=False, loss_fn="ce"):
    if save_path is None:
        save_option = 'none'
    if save_option not in ['best', 'last', 'none']:
        raise ValueError("Model-saving mode must be best/last/none")
    if save_option == 'best' and no_val:
        raise ValueError(
            "Cannot identify best-val-loss model if val loss not computed")

    optim = ch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader = loaders

    # Define for first epoch
    train_loader_use = train_loader

    best_loss, best_vacc = np.inf, 0.0
    best_model = None

    if study_mode:
        collect_stats = []

    iterator = range(epochs)
    if not verbose:
        iterator = tqdm(iterator)
    for e in iterator:
        # Train epoch
        tr_loss, _ = epoch(model, train_loader_use, optim, e + 1, c_rule, n_classes,
                           corrupt_class=corrupt_class,
                           poison_ratio=poison_ratio,
                           verbose=verbose, lossfn=loss_fn)
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
                        "train_prop_acc": 100 * prop_acc,
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
        
        # Intervention on batch ordering
        train_loader_use = find_optimal_batch_order(
            model, train_loader, clean_train_loader,
            lr, weight_decay, get_metrics_at_epoch_end)

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


def train_poisoned_model(model, callable_ds, poison_ratio, args):
    # Poison data once at the start, train model normal afterwards
    
    ds = callable_ds()
    ds_clean = callable_ds()

    indices = None

    if args.use_given_data:
        print(utils.yellow_print("Using given data"))
        poison_data = np.load(args.poison_path)
        poison_x, poison_y = poison_data['x'], poison_data['y']
        ds.add_poison_data(poison_x, poison_y)
    else:
        ds.poison_train_data(args.poison_class, poison_ratio,
                             args.c_rule, selection=indices,
                             save_data=args.save_poisoned_data,
                             offset=args.offset)

    print("Training on %d samples" % len(ds.train))
    print(utils.red_print("%d additional points" %
                          (len(ds.train) - len(ds_clean.train))))

    batch_size = args.batch_size
    shuffle = True
    if batch_size == -1:
        batch_size = len(ds.train)
        shuffle = False

    train_loader, val_loader = ds.get_loaders(batch_size, shuffle=shuffle)
    clean_train_loader, _ = ds_clean.get_loaders(
        batch_size, shuffle=shuffle)

    return_data = train_model(
        model, (train_loader, val_loader), epochs=args.epochs,
        c_rule=args.c_rule, n_classes=ds.n_classes,
        weight_decay=args.weight_decay,
        lr=args.lr, verbose=args.verbose,
        no_val=True, get_metrics_at_epoch_end=args.poison_class,
        clean_train_loader=clean_train_loader,
        study_mode=args.study_mode,
        loss_fn=args.loss)

    if args.study_mode:
        model, _, _, all_stats = return_data
    else:
        model, _, _ = return_data

    if args.study_mode:
        return model, all_stats

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', default='flat',
                        choices=get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--dataset', default='mnist17_first',
                        choices=datasets.get_dataset_names(),
                        help="Which dataset to use?")
    parser.add_argument('--attacker_goal', default=0.05,
                        type=float, help='desired accuracy on target class')
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Batch size while training models"
                             "Set as -1 to run GD instead of BGD")
    parser.add_argument('--poison_class', default=0, type=int,
                        choices=list(range(10)),
                        help='Which class to target for corruption')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Epochs while training models')
    parser.add_argument('--lr', default=2e-3, type=float,
                        help='Learning rate for models')
    parser.add_argument('--weight_decay', default=0.02, type=float,
                        help='Weight decay while training')
    parser.add_argument('--c_rule', default='cycle',
                        choices=['cycle', 'random'],
                        help='Learning rate for models')
    parser.add_argument('--verbose', action="store_true",
                        help='If true, print per-epoch training statistics')
    parser.add_argument('--seed', default=2021, type=int,
                        help='Seed for model weight initialization')
    parser.add_argument('--offset', default=3, type=int,
                        help='For cycle mode, target class: + offset')
    parser.add_argument('--loss', default="ce",
                        choices=["ce", "hinge"],
                        help='Loss function to use while training models')
    parser.add_argument('--save_poisoned_data', default=None,
                        help="Provide path where poisoned models are stored,"
                        "for later use. Only valid for pre mode")
    parser.add_argument('--dir_prefix', default="./data/models_order_control/",
                        help="Directory to save models in")
    parser.add_argument('--study_mode', action="store_true",
                        help="Plot statistics across epochs")
    parser.add_argument('--poison_rates', type=str,
                        default="0.4",
                        help='Comma-separated list of poison-rates to try')

    # Use provided data as poison data
    parser.add_argument('--poison_path', type=str,
                        help='Path to save poison data')
    parser.add_argument('--use_given_data', action="store_true",
                        help='Use given data to train model')
    args = parser.parse_args()
    utils.flash_utils(args)
    args.low_confidence = False

    # Make sure theta values provided in valid format
    try:
        poison_rates = [float(x)
                        for x in args.poison_rates.replace(" ", "").split(',')]
        assert len(poison_rates) > 0, 'Provide at least one theta value'
        args.poison_rates = poison_rates
    except ValueError:
        raise ValueError("Theta values not provided in correct format")

    # Before saving anything, make sure directory exists
    model_dir = os.path.join(args.dir_prefix, "{}/target/"
                             "arch-{}_target-{}_goal-{}_"
                             "rule-{}/loss-{}".format(
                                 args.dataset, args.model_arch,
                                 args.poison_class, args.attacker_goal,
                                 args.c_rule, args.loss))
    utils.ensure_dir_exists(model_dir)

    og_save_poisoned_data = args.save_poisoned_data

    best_model_obj, best_loss = None, np.inf
    for ratio in poison_rates:
        if args.low_confidence and ratio > 1:
            raise ValueError("Highest-loss selection with ratio > 1 "
                             "makes no sense")

        # Make sure data is saved
        if og_save_poisoned_data is not None:
            args.save_poisoned_data = os.path.join(
                og_save_poisoned_data,
                "seed_{}/ratio_{}".format(args.seed, ratio))
            utils.ensure_dir_exists(args.save_poisoned_data)

        # Fetch appropriate dataset
        callable_ds = datasets.dataset_helper(args.dataset)
        n_classes = callable_ds().n_classes

        # Construct model
        model = get_seeded_wrapped_model(args, n_classes=n_classes)

        # Train model
        if args.study_mode:
            model, all_stats = train_poisoned_model(
                model, callable_ds, ratio, args)
        else:
            model = train_poisoned_model(model, callable_ds, ratio, args)

        # Compute metrics for said model
        train_loader, val_loader = callable_ds().get_loaders(512)
        _, train_loss = get_model_metrics(model, train_loader)
        test_acc, _ = get_model_metrics(model, val_loader)

        (trn_sub_acc, _), (trn_nsub_acc, _) = get_model_metrics(
            model, train_loader, args.poison_class)
        (tst_sub_acc, _), (tst_nsub_acc, _) = get_model_metrics(
            model, val_loader, args.poison_class)

        # Print accuracies on target/non-target data
        # On seen (train) and unseen (val) data
        if not args.use_given_data:
            print(utils.pink_print("Ratio %.3f" % (ratio)))
        print("Total Acc: %.3f" % test_acc)
        print('Train Target Acc : %.3f' % trn_sub_acc)
        print('Train Collat Acc : %.3f' % trn_nsub_acc)
        print('Test Target Acc : %.3f' % tst_sub_acc)
        print('Test Collat Acc : %.3f' % tst_nsub_acc)
        print()

        # If study mode, plot trends across epochs
        if args.study_mode:
            X = np.arange(len(all_stats))
            # For binary case
            look_at = ["train_prop_acc", "train_noprop_acc",
                       "val_prop_acc", "val_noprop_acc", "norm", "lossx100"]
            # For multi case
            # look_at = ["train_prop_acc", "train_noprop_acc", "val_prop_acc", "val_noprop_acc", "norm", "lossx50"]
            for la in look_at:
                Y = [p[la] for p in all_stats]
                plt.plot(X, Y, label=la)
            plt.legend()
            plt.savefig("./data/visualize_reorder/run_info_dataset-%s_pr-%.2f_seed-%d_arch-%s_wd-%f_bs-%d.png" %
                        (args.dataset, ratio, args.seed, args.model_arch, args.weight_decay, args.batch_size))

        # Purpose of this mode is just to train model once
        # Exit after that
        if args.use_given_data:
            exit(0)

        # Save current model
        model_name = "seed-{}_ratio-{}_loss-{}_bs-{}.pth".format(
            args.seed, ratio, train_loss, args.batch_size)
        ch.save(copy.deepcopy(model).state_dict(),
                os.path.join(model_dir, model_name))
        print("Saved model to %s" % os.path.join(model_dir, model_name))

        if tst_sub_acc <= args.attacker_goal and train_loss < best_loss:
            best_loss = train_loss
            best_model_obj = {
                "model": copy.deepcopy(model),
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_collat_acc": tst_nsub_acc,
                "test_target_acc": tst_sub_acc,
                "ratio": ratio
            }
            print(utils.yellow_print(
                "Updated lowest train loss: %.4f" % train_loss))

    if best_model_obj is None:
        print(utils.red_print("No model satisfied given adversary's goal!"))
