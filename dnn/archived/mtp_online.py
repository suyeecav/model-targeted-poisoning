import dnn_utils
import torch as ch
import numpy as np
import mtp_utils
import datasets
import argparse
import utils
import os
from torch.utils.tensorboard import SummaryWriter


def stop_cond(args, best_loss, num_iters,
              model_t, model_p, tst_sub_acc,
              norm_diffs):
    ol_lr_threshold = args.incre_tol_par

    if args.fixed_budget <= 0:
        if args.require_acc:
            # Define stop condition
            print("Stop condition: %f, %f" % (tst_sub_acc, args.err_threshold))
            stop_cond = tst_sub_acc > 1 - args.err_threshold
        else:
            if args.online_alg_criteria == "max_loss":
                current_tol_par = best_loss
            else:
                # use the euclidean distance as the stop criteria
                current_tol_par = norm_diffs
            # Define stop condition
            stop_cond = current_tol_par > ol_lr_threshold
    else:
        stop_cond = num_iters < args.fixed_budget
    return stop_cond


def modelTargetPoisoning(model_p, logger, args):
    # Implementation of Algorithm 1, modified for DNNs
    # Line number corresponding to the Algorithm is mentioned
    # Along with each high-level function call

    # Fetch appropriate dataset
    ds = datasets.dataset_helper(args.dataset)()

    # Keep track of number of points model has seen (virtually)
    # For loss-normalization purposes
    points_seen_count = len(ds.train)

    # Line 1: Collect poisoning points
    D_p = [[], []]

    # Line 3: Since D_p is empty in first iteration, simply train it outside
    model_t = mtp_utils.train_clean_model(ds, args)

    # Report performance of clean model
    train_loader, test_loader = ds.get_loaders(args.batch_size)
    clean_acc, _ = dnn_utils.get_model_metrics(model_t, test_loader)
    print(utils.yellow_print("[Clean-model] Total Acc: %.4f" % clean_acc))
    _, clean_total_loss = dnn_utils.get_model_metrics(model_t, train_loader)
    print(utils.yellow_print(
        "[Clean-model] Loss on train: %.4f" % clean_total_loss))
    print()

    # theta_1: (sum of) gradients of model weights
    # with respect to clean training set
    print(utils.yellow_print("[Computing gradients on clean training data]"))
    theta_curr = datasets.get_dataset_gradients(
        model=model_t, ds=ds, batch_size=args.batch_size,
        weight_decay=args.pretrain_weight_decay,
        verbose=args.verbose_precomp, is_train=True)

    # Line 2: Iterate until stopping criteria met
    best_loss = np.inf
    num_iters = 0
    condition = True
    while condition:

        # Line 4: Compute (x_opt, y_opt)
        opt_pair, best_loss = mtp_utils.find_optimal(
            theta_t=model_t, theta_p=model_p, input_shape=ds.datum_shape,
            n_classes=ds.n_classes, trials=args.trials,
            num_steps=args.num_steps, step_size=args.optim_lr,
            verbose=args.verbose_opt)
        x_opt, y_opt = opt_pair

        # Update theta (gradients for online learning) for use in next iter
        print(utils.yellow_print("[Updating gradients]"))
        theta_curr = mtp_utils.update_gradients(
            model=model_t, thetas=theta_curr,
            weight_decay=args.update_weight_decay,
            x_opt=x_opt, y_opt=y_opt)

        # Calculate useful statistics
        (tst_sub_acc, _), _ = dnn_utils.get_model_metrics(
            model=model_t,
            loader=test_loader,
            target_prop=args.poison_class)
        _, (trn_nsub_acc, _) = dnn_utils.get_model_metrics(
            model=model_t,
            loader=train_loader,
            target_prop=args.poison_class)
        norm_diffs = dnn_utils.model_l2_closeness(model_t, model_p)

        # Log information
        mtp_utils.log_information(logger=logger, best_loss=best_loss,
                                  x_opt=x_opt, model_t=model_t,
                                  norm_diffs=norm_diffs,
                                  trn_nsub_acc=trn_nsub_acc,
                                  tst_sub_acc=tst_sub_acc,
                                  num_iters=num_iters, args=args)

        # Line 3: theta_t = train(D_c U D_p)
        # Instead of training from scratch, perform online mirror descent
        model_t = mtp_utils.w_optimal_gradient_ascent(
            model=model_t, thetas=theta_curr,
            num_points_seen_virtually=points_seen_count,
            method=args.method, lr=args.oga_lr,
            weight_decay=args.oga_weight_decay,
            # Not sure if should be same weight decay
            # when model was pre-trained
            # Or a larger value to prevent model weights from exploding
            # weight_decay=args.pretrain_weight_decay,
            iters=args.iters, verbose=args.verbose_oga)

        # Line 5: Add (x*, y*) to D_p
        D_p[0].append(x_opt.cpu())
        D_p[1].append(y_opt.cpu())
        points_seen_count += 1

        # Log some information about x*, y*
        pred_t, pred_p = model_t(x_opt), model_p(x_opt)
        print(utils.cyan_print("Mt(x*): %d, Mp(x*): %d, y*: %d" %
                               (pred_t.argmax(1), pred_p.argmax(1), y_opt)))

        # Line 6: Get ready to check condition
        condition = stop_cond(args=args, best_loss=best_loss,
                              num_iters=num_iters,
                              model_t=model_t, model_p=model_p,
                              tst_sub_acc=tst_sub_acc,
                              norm_diffs=norm_diffs)

        # Keep track of no. of iterations
        num_iters += 1
        print()

    # Line 7: Return poison data
    return D_p, model_t


def train_on_poisoned_data(args, poisoned_data):
    callable_ds = datasets.dataset_helper(args.dataset)
    ds = callable_ds()
    model = dnn_utils.model_helper(args.model_arch)()
    model = dnn_utils.multi_gpu_wrap(model)

    for x, y in zip(*poisoned_data):
        ds.add_point_to_train(x, y)

    model = mtp_utils.train_clean_model(ds, args, epochs=15)

    # Compute metrics for said model
    train_loader, val_loader = callable_ds().get_loaders(args.batch_size)
    _, train_loss = dnn_utils.get_model_metrics(model, train_loader)
    test_acc, _ = dnn_utils.get_model_metrics(model, val_loader)

    (trn_sub_acc, _), (trn_nsub_acc, _) = dnn_utils.get_model_metrics(
        model, train_loader, args.poison_class)
    (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
        model, val_loader, args.poison_class)

    # Print accuracies on target/non-target data
    # On seen (train) and unseen (val) data
    print("Total Acc: %.3f" % test_acc)
    print('Train Target Acc : %.3f' % trn_sub_acc)
    print('Train Collat Acc : %.3f' % trn_nsub_acc)
    print('Test Target Acc : %.3f' % tst_sub_acc)
    print('Test Collat Acc : %.3f' % tst_nsub_acc)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ol: target classifier is from the adapttive attack
    # kkt: target is from kkt attack
    # real: actual classifier, compare: compare performance
    # of kkt attack and adaptive attack using same stop criteria

    # Global params
    parser.add_argument('--model_arch', default='lenet',
                        choices=dnn_utils.get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--dataset', default='mnist',
                        choices=datasets.get_dataset_names(),
                        help="Which dataset to use?")
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size while training models')
    parser.add_argument('--online_alg_criteria', default='norm',
                        choices=['max_loss', 'norm'],
                        help='Stop criteria of online alg: max_loss or norm')
    parser.add_argument('--poison_model_path', type=str,
                        help='Path to saved poisoned-classifier')
    parser.add_argument('--log_path', type=str,
                        default="./data/logs",
                        help='Path to save logs')
    parser.add_argument('--theta_values', type=str,
                        default="0.05,0.1,0.15",
                        help='Comma-separated list of theta values to try')
    parser.add_argument('--poison_class', default=4, choices=list(range(10)),
                        help='Which class to target for corruption')

    # Params for pre-training model
    parser.add_argument('--pretrain_weight_decay', default=0.01,
                        type=float, help='Weight decay while pre-training')
    parser.add_argument('--epochs', default=5, type=int,
                        help='Epochs while training models')
    parser.add_argument('--pretrain_lr', default=1e-3, type=float,
                        help='Learning rate for models')

    # some params related to online algorithm, use the default
    parser.add_argument('--incre_tol_par', default=1e-2, type=float,
                        help='Stop value of online alg: max_loss or norm')
    parser.add_argument('--err_threshold', default=1.0,
                        type=float, help='Target error rate')
    parser.add_argument('--rand_seed', default=12,
                        type=int, help='Random seed')
    parser.add_argument('--repeat_num', default=5, type=int,
                        help="Number of times maximum-loss-diff "
                        "point is repeated")
    parser.add_argument('--improved', action="store_true",
                        help="If true, target classifier is obtained "
                        "through improved process")
    parser.add_argument('--fixed_budget', default=0, type=int,
                        help="If > 0, then run the attack for "
                        "fixed number of points")
    parser.add_argument('--require_acc', action="store_true",
                        help="If true, terminate when "
                        "accuracy requirement is achieved")

    # Params for computing optimal w_t
    parser.add_argument('--method', default='layer_sum',
                        choices=['layer_mean', 'layer_sum'],
                        help='Layer-weight aggregation method')
    parser.add_argument('--iters', default=500, type=int,
                        help='Number of iterations while optimizing w_t')
    parser.add_argument('--oga_lr', default=2e-7, type=float,
                        help='Learning rate while optimizing w_t')
    parser.add_argument('--oga_weight_decay', default=0.2, type=float,
                        help='Weight-decay while optimizing w_t')

    # Params for (x*, y*) computation
    parser.add_argument('--trials', default=10, type=int,
                        help='Number of trials while searching for x*, y*')
    parser.add_argument('--num_steps', default=500, type=int,
                        help='Number of steps while searching for x*, y*')
    parser.add_argument('--optim_lr', default=1e-2, type=float,
                        help='Learning rate while searching for x*, y*')

    # Params for updating gradients
    parser.add_argument('--update_weight_decay', default=0.1, type=float,
                        help='Weight decay for updating gradients')

    # Different levels of verbose
    parser.add_argument('--verbose', action="store_true",
                        help='If true, print everything')
    parser.add_argument('--verbose_pretrain', action="store_true",
                        help='If true, print per-epoch training statistics')
    parser.add_argument('--verbose_oga', action="store_true",
                        help='If true, print loss while running OMD')
    parser.add_argument('--verbose_opt', action="store_true",
                        help='If true, print loss while computing x*, y*')
    parser.add_argument('--verbose_precomp', action="store_true",
                        help="If true, progress while computing "
                        "gradients on clean data")

    args = parser.parse_args()

    # Make sure theta values provided in valid format
    try:
        theta_values = [float(x)
                        for x in args.theta_values.replace(" ", "").split(',')]
        assert len(theta_values) > 0, 'Provide at least one theta value'
        args.theta_values = theta_values
    except Exception:
        raise ValueError("Theta values not provided in correct format")

    if args.verbose:
        args.verbose_pretrain = True
        args.verbose_oga = True
        args.verbose_opt = True
        args.verbose_precomp = True

    # Print all arguments
    utils.flash_utils(args)

    # Load target model theta_p
    theta_p = dnn_utils.model_helper(args.model_arch)()
    theta_p = dnn_utils.multi_gpu_wrap(theta_p)
    theta_p.load_state_dict(ch.load(args.poison_model_path))
    theta_p.eval()

    # Report performance of poisoned model
    train_loader, test_loader = datasets.dataset_helper(
        args.dataset)().get_loaders(args.batch_size)
    clean_acc, _ = dnn_utils.get_model_metrics(theta_p, test_loader)
    print(utils.yellow_print("[Poisoned-model] Total Acc: %.4f" % clean_acc))
    _, clean_total_loss = dnn_utils.get_model_metrics(theta_p, train_loader)
    print(utils.yellow_print(
        "[Poisoned-model] Loss on train: %.4f" % clean_total_loss))
    # Report weight norm for poisoned model
    poisoned_norm = dnn_utils.get_model_l2_norm(theta_p).item()
    print(utils.yellow_print(
        "[Poisoned-model] Weights norm: %.4f" % poisoned_norm))
    print()

    for valid_theta_err in args.theta_values:
        args.err_threshold = valid_theta_err

        # Prepare logger
        log_dir = os.path.join(args.log_path, str(valid_theta_err))
        utils.ensure_dir_exists(log_dir)
        logger = SummaryWriter(log_dir=log_dir, flush_secs=10)

        print(utils.pink_print(
            "Running attack for theta %.2f" % valid_theta_err))

        # Get poison data
        poison_data, theta_t = modelTargetPoisoning(theta_p, logger, args)
        print("%d poison points produced!" % len(poison_data[0]))

        # Train a new model with this poisoned data
        train_on_poisoned_data(args, poison_data)

        # Close logger
        logger.close()
