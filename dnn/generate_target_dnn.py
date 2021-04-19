import argparse
import dnn_utils
import torch as ch
import copy
from tqdm import tqdm
import numpy as np
import os
import utils
import datasets

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def identify_before_train(model, args, ratio, blind=False):
    if args.verbose:
        print(utils.yellow_print("Identifying best points "
                                 "to poison (based on loss)"))
    # Get data loaders
    ds = datasets.dataset_helper(args.dataset)()

    if blind:
        # Train new model for an epoch
        model_ = dnn_utils.get_seeded_wrapped_model(args)
        train_loader, val_loader = ds.get_loaders(args.batch_size)
        model_, _, _ = dnn_utils.train_model(
            model_, (train_loader, val_loader), epochs=1,
            c_rule=None, n_classes=ds.n_classes,
            weight_decay=args.weight_decay,
            lr=args.lr, verbose=False,
            no_val=True,)
    else:
        model_ = model

    # Set model to evaluation mode
    model_.eval()

    k = int(len(ds.train) * ratio)
    indices = utils.pick_k_lowest_loss(model_, ds, k, args.batch_size)
    return indices


def epoch_wise_corruption(model, callable_ds, lr, epochs,
                          batch_size, poison_class, poison_ratio,
                          c_rule, weight_decay, verbose=False,
                          low_confidence=False, loss_fn="ce",
                          offset=3):
    optim = ch.optim.Adam(model.parameters(), lr=args.lr,
                          weight_decay=weight_decay)
    iterator = tqdm(range(epochs))
    for e in iterator:
        # Poison data at the start of each epoch
        ds = callable_ds()

        indices = None
        # If loss-based criteria requested, use that
        if low_confidence:
            indices = identify_before_train(model, args, ratio, blind=(e == 0))

        ds.poison_train_data(poison_class, poison_ratio,
                             c_rule, selection=indices,
                             offset=offset)

        if batch_size == -1:
            batch_size = len(ds.train)

        train_loader, val_loader = ds.get_loaders(batch_size)

        # Train epoch
        _, train_acc = dnn_utils.epoch(
            model, train_loader, optim, e + 1,
            c_rule, ds.n_classes, verbose=False)

        if verbose:
            iterator.set_description(
                "Train acc: %.4f" % (train_acc))

    return model


def train_poisoned_model(model, callable_ds, poison_ratio, args):
    # Poison data once at the start, train model normal afterwards
    if args.poison_mode == "pre":
        ds = callable_ds()
        ds_clean = callable_ds()

        indices = None
        # If loss-based criteria requested, use that
        if args.low_confidence:
            indices = identify_before_train(model, args, ratio, blind=True)

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
        return_data = dnn_utils.train_model(
            model, (train_loader, val_loader), epochs=args.epochs,
            c_rule=args.c_rule, n_classes=ds.n_classes,
            weight_decay=args.weight_decay,
            lr=args.lr, verbose=args.verbose,
            no_val=True,
            get_metrics_at_epoch_end=args.poison_class,
            clean_train_loader=clean_train_loader,
            study_mode=args.study_mode,
            loss_fn=args.loss)
        
        if args.study_mode:
            model, _, _, all_stats = return_data
        else:
            model, _, _ = return_data

    # Poison data per batch
    elif args.poison_mode == "batch":
        ds = callable_ds()
        train_loader, val_loader = ds.get_loaders(args.batch_size)
        model, _, _ = dnn_utils.train_model(
            model, (train_loader, val_loader), epochs=args.epochs,
            c_rule=args.c_rule, n_classes=ds.n_classes,
            weight_decay=args.weight_decay,
            lr=args.lr, verbose=args.verbose,
            corrupt_class=args.poison_class,
            poison_ratio=poison_ratio,
            no_val=True,
            low_confidence=args.low_confidence,
            get_metrics_at_epoch_end=args.poison_class,
            loss_fn=args.loss)

    # Posion data at the start of each epoch
    elif args.poison_mode == "epoch":
        model = epoch_wise_corruption(
            model, callable_ds, args.lr, args.epochs, args.batch_size,
            args.poison_class, poison_ratio, args.c_rule,
            args.weight_decay, verbose=args.verbose,
            low_confidence=args.low_confidence,
            loss_fn=args.loss)

    if args.study_mode:
        return model, all_stats

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', default='flat',
                        choices=dnn_utils.get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--dataset', default='mnist17_first',
                        choices=datasets.get_dataset_names(),
                        help="Which dataset to use?")
    parser.add_argument('--attacker_goal', default=0.05,
                        type=float, help='desired accuracy on target class')
    parser.add_argument('--batch_size', default=-1, type=int,
                        help="Batch size while training models"
                             "Set as -1 to run GD instead of BGD")
    parser.add_argument('--poison_class', default=4, type=int,
                        choices=list(range(10)),
                        help='Which class to target for corruption')
    parser.add_argument('--epochs', default=10, type=int,
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
    parser.add_argument('--low_confidence', action="store_true",
                        help='Train for an epoch, \
                            identify highest-loss points, \
                            use those instead (epoch), \
                            idetify highest-loss points so far (pre) \
                            and sample from them, or identify these points \
                            per batch (batch)')
    parser.add_argument('--poison_mode', default='pre',
                        choices=['pre', 'epoch', 'batch'],
                        help="Poison data before starting anything (pre),"
                        "poison data at the start of each epoch (epoch), "
                        "or poison data in rach batch (batch)")
    parser.add_argument('--save_poisoned_data', default=None,
                        help="Provide path where poisoned models are stored,"
                        "for later use. Only valid for pre mode")
    parser.add_argument('--dir_prefix', default="./data/models/",
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

    # Make sure argument makes sense
    if args.save_poisoned_data is not None and args.poison_mode != "pre":
        raise ValueError("Saving poisoned data only makes sense for pre mode")

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
        model = dnn_utils.get_seeded_wrapped_model(args, n_classes=n_classes)

        # Train model
        if args.study_mode:
            model, all_stats = train_poisoned_model(model, callable_ds, ratio, args)
        else:
            model = train_poisoned_model(model, callable_ds, ratio, args)

        # Compute metrics for said model
        train_loader, val_loader = callable_ds().get_loaders(512)
        _, train_loss = dnn_utils.get_model_metrics(model, train_loader)
        test_acc, _ = dnn_utils.get_model_metrics(model, val_loader)

        (trn_sub_acc, _), (trn_nsub_acc, _) = dnn_utils.get_model_metrics(
            model, train_loader, args.poison_class)
        (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
            model, val_loader, args.poison_class)

        # Print accuracies on target/non-target data
        # On seen (train) and unseen (val) data
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
            look_at = ["train_prop_acc", "train_noprop_acc", "val_prop_acc", "val_noprop_acc", "norm", "lossx100"]
            # For multi case
            # look_at = ["train_prop_acc", "train_noprop_acc", "val_prop_acc", "val_noprop_acc", "norm", "lossx50"]
            for la in look_at:
                Y = [p[la] for p in all_stats]
                plt.plot(X, Y, label=la)
            plt.legend()
            plt.savefig("./data/visualize/run_info_pr-%.2f_seed-%d_arch-%s_wd-%f.png" %
                        (ratio, args.seed, args.model_arch, args.weight_decay))


        # Purpose of this mode is just to train model once
        # Exit after that
        if args.use_given_data:
            exit(0)

        # Save current model
        model_name = "seed-{}_ratio-{}_mode-{}_loss-{}.pth".format(
            args.seed, ratio, args.poison_mode, train_loss)
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
