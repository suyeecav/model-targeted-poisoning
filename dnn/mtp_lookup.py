import dnn_utils
import torch as ch
import numpy as np
import mtp_utils
import datasets
import argparse
import utils
import os
from torch.utils.tensorboard import SummaryWriter
import copy


def stop_cond(args, best_loss, num_iters,
              tst_sub_acc, norm_diffs):
    ol_lr_threshold = args.incre_tol_par

    if args.fixed_budget <= 0:
        if args.require_acc:
            # Define stop condition
            stop_cond = tst_sub_acc > 1 - args.err_threshold
            print(utils.red_print(
                "Current accuracy on population (test): %.4f" % tst_sub_acc))
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
    ds = datasets.dataset_helper("memory")(path=args.path_1)

    # Maintain copy of clean data (for seed sampling)
    ds_clean = datasets.dataset_helper("memory")(path=args.path_1)

    # Data to pick points from (for x* optimization)
    ds_second = datasets.dataset_helper("memory")(
        path=args.path_2)
    loader_optim, _ = ds_second.get_loaders(1000)

    # Line 1: Collect poisoning points
    D_p = [[], []]

    # Line 3: Since D_p is empty in first iteration, simply train it outside
    model_t_pretrained, pretrain_optim = mtp_utils.train_clean_model(ds, args)

    # Report performance of clean model
    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = len(ds.train)

    train_loader, test_loader = ds.get_loaders(batch_size)
    clean_acc, clean_total_loss = dnn_utils.get_model_metrics(
        model_t_pretrained, test_loader, lossfn=args.loss)
    print(utils.yellow_print(
        "[Clean-model][Test] Total Acc: %.4f" % clean_acc))
    print(utils.yellow_print(
        "[Clean-model] Loss on train: %.4f" % clean_total_loss))
    (population_acc, _), (non_population_acc, _) = dnn_utils.get_model_metrics(
        model_t_pretrained, test_loader,
        lossfn=args.loss, target_prop=args.poison_class)
    print(utils.red_print(
        "[Clean-model][Test] Population Acc: %.4f" % population_acc))
    print(utils.red_print(
        "[Clean-model][Test] Non- Population Acc: %.4f" % non_population_acc))
    print()

    # Line 2: Iterate until stopping criteria met
    best_loss = np.inf
    num_iters = 0
    condition = True
    while condition:

        if len(D_p[0]) > 0:
            # Line 3: theta_t = train(D_c U D_p)
            print(utils.yellow_print(
                "[Training model on Dc U Dp "
                "(on %d samples)]" % len(ds.train)))
            # Get loader for D_c U D_p
            batch_size = args.batch_size
            if batch_size == -1:
                batch_size = len(ds.train)
            data_loader, _ = ds.get_loaders(batch_size)

            # Do not re-initialize model if finetuning requested
            if not args.finetune:
                # Construct model
                model_t = dnn_utils.get_seeded_wrapped_model(
                    args, n_classes=ds.n_classes)
            else:
                # Start finetuning from the point where model
                # has seen only clean data
                model_t = copy.deepcopy(model_t_pretrained)
            # Set model to training mode
            model_t.train()

            # Define optimizer
            optim = ch.optim.Adam(model_t.parameters(),
                                  lr=args.pretrain_lr,
                                  weight_decay=args.pretrain_weight_decay)

            # Adjust starting point of optimizer
            # if finetuning is requested
            if args.finetune:
                optim.load_state_dict(pretrain_optim.state_dict())

            # Increase numer of iterations theta_t is trained for
            # as size of its training set |D_c U D_p| increases
            iters = args.iters
            if args.increase_iters:
                iters += int((len(ds.train) - len(ds_clean.train)
                              ) / args.increase_every)

            # Train model
            for e in range(iters):
                # Train epoch
                dnn_utils.epoch(model=model_t, loader=data_loader,
                                optimizer=optim, epoch_num=e+1,
                                c_rule=None, n_classes=None,
                                verbose=True, lossfn=args.loss)
        else:
            model_t = model_t_pretrained

        # Make sure theta_t is in eval mode
        model_t.eval()

        # Line 4: Compute (x*, y*)
        if args.optim_type == "lookup":
            # Loss-difference based lookup method
            (x_opt, y_opt), best_loss = mtp_utils.lookup_based_optimal(
                theta_t=model_t, theta_p=model_p, loader=loader_optim,
                n_classes=ds_second.n_classes, random=args.random,
                lossfn=args.loss, filter=args.filter, verbose=True)
        elif args.optim_type == "dataset_grad":
            # Dataset-gradient alignment loss based optimization
            (x_opt, y_opt), best_loss = mtp_utils.dataset_grad_optimal(
                theta_t=model_t, theta_p=model_p, input_shape=ds_second.datum_shape,
                n_classes=ds_second.n_classes, trials=args.optim_trials, ds=ds,
                num_steps=args.optim_steps, step_size=args.optim_lr,
                verbose=True, signed=args.signed)
        elif args.optim_type == "loss_difference":
            # Loss difference based optimization
            (x_opt, y_opt), best_loss = mtp_utils.find_optimal_using_optim(
                theta_t=model_t, theta_p=model_p, input_shape=ds_second.datum_shape,
                n_classes=ds_second.n_classes, num_steps=args.optim_steps,
                trials=args.optim_trials,  step_size=args.optim_lr,
                filter=args.filter, verbose=True)
        else:
            raise NotImplemented("Loss optimization method not implemented")

        # Log some information about x*, y*
        with ch.no_grad():
            pred_t, pred_p = model_t(x_opt), model_p(x_opt)
        print(utils.cyan_print("Loss: %.3f Mt(x*): %d, Mp(x*): %d, y*: %d" %
                               (best_loss.item(), pred_t.argmax(1),
                                pred_p.argmax(1), y_opt)))

        # Line 5: Add (x*, y*) to D_p
        for _ in range(args.n_copies):
            D_p[0].append(x_opt.cpu())
            D_p[1].append(y_opt.cpu())
            ds.add_point_to_train(x_opt.cpu(), y_opt.cpu())
        print()

        # Calculate useful statistics
        (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
            model=model_t,
            loader=test_loader,
            target_prop=args.poison_class,
            lossfn=args.loss)
        (trn_sub_acc, _), (trn_nsub_acc, _) = dnn_utils.get_model_metrics(
            model=model_t,
            loader=train_loader,
            target_prop=args.poison_class,
            lossfn=args.loss)
        norm_diffs = dnn_utils.model_l2_closeness(model_t, model_p)

        # Log information
        mtp_utils.log_information(
            logger=logger, best_loss=best_loss,
            x_opt=x_opt, norm_diffs=norm_diffs,
            trn_sub_acc=trn_sub_acc, trn_nsub_acc=trn_nsub_acc,
            tst_sub_acc=tst_sub_acc, tst_nsub_acc=tst_nsub_acc,
            num_iters=num_iters + 1, args=args, label=y_opt)

        # Line 6: Get ready to check condition
        condition = stop_cond(args=args, best_loss=best_loss,
                              num_iters=num_iters,
                              tst_sub_acc=tst_sub_acc,
                              norm_diffs=norm_diffs)

        # Keep track of no. of iterations
        num_iters += 1

    # Line 7: Return poison data
    return D_p, model_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ol: target classifier is from the adapttive attack
    # kkt: target is from kkt attack
    # real: actual classifier, compare: compare performance
    # of kkt attack and adaptive attack using same stop criteria

    # Global params
    parser.add_argument('--model_arch', default='flat',
                        choices=dnn_utils.get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--batch_size', default=-1, type=int,
                        help='Batch size while training models')
    parser.add_argument('--online_alg_criteria', default='max_loss',
                        choices=['max_loss', 'norm'],
                        help='Stop criteria of online alg: max_loss or norm')
    parser.add_argument('--poison_model_path', type=str,
                        default="./data/models/seed-2021_ratio-0.6_mode-pre_loss-0.4150749979689362.pth",
                        help='Path to saved poisoned-classifier')
    parser.add_argument('--log_path', type=str,
                        default="./data/logs_pick_new",
                        help='Path to save logs')
    parser.add_argument('--theta_values', type=str,
                        default="0.99",
                        help='Comma-separated list of theta values to try')
    parser.add_argument('--poison_class', default=0,
                        type=int, choices=list(range(10)),
                        help='Which class to target for corruption')
    parser.add_argument('--loss', default="ce",
                        choices=["ce", "hinge"],
                        help='Loss function to use while training models')

    # Params for pre-training model
    parser.add_argument('--pretrain_weight_decay', default=0.2,
                        type=float, help='Weight decay while pre-training')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Epochs while pre-training models')
    parser.add_argument('--pretrain_lr', default=5e-3, type=float,
                        help='Learning rate for models')

    # some params related to online algorithm, use the default
    parser.add_argument('--incre_tol_par', default=1e-2, type=float,
                        help='Stop value of online alg: max_loss or norm')
    parser.add_argument('--err_threshold', default=1.0,
                        type=float, help='Target error rate')
    parser.add_argument('--fixed_budget', default=0, type=int,
                        help="If > 0, then run the attack for "
                        "fixed number of points")
    parser.add_argument('--require_acc', action="store_true",
                        help="If true, terminate when "
                        "accuracy requirement is achieved")

    # Params for computing optimal w_t
    parser.add_argument('--iters', default=20, type=int,
                        help='Number of iterations while optimizing w_t')
    parser.add_argument('--seed', default=2021, type=int,
                        help='Seed for weight init')
    parser.add_argument('--finetune', action="store_true",
                        help='Finetune instead of training from scratch')
    parser.add_argument('--increase_iters', action="store_true",
                        help='Use more train iterations as |D_p| increases')
    parser.add_argument('--skip_bad', action="store_true",
                        help='Do not consider points that end up'
                             'increasing loss')
    parser.add_argument('--increase_every', default=1000, type=int,
                        help='Add epoch for every # points')

    # Params for (x*, y*) computation
    parser.add_argument('--n_copies', default=1, type=int,
                        help='Number of copies per (x*,y*) to be added')
    parser.add_argument('--random', action="store_true",
                        help='Use random selection for points')
    parser.add_argument('--filter', action="store_true",
                        help='Apply filter when picking x*, y*')
    parser.add_argument('--path_1', default="./data/datasets/MNIST17/split_1.pt",
                        help='Path to first split of dataset')
    parser.add_argument('--path_2', default="./data/datasets/MNIST17/split_2.pt",
                        help='Path to second split of dataset')
    parser.add_argument('--optim_type', default="lookup",
                        choices=["dataset_grad", "lookup", "loss_difference"],
                        help='Optimization method to compute (x*, y*)')
    parser.add_argument('--optim_lr', default=1e-2,
                        type=float, help='Step size for optimization step')
    parser.add_argument('--optim_steps', default=100,
                        type=int, help='Number of steps for optimization step')
    parser.add_argument('--optim_trials', default=5,
                        type=int, help='Number of trials for optimization step')
    parser.add_argument('--signed', action="store_true",
                        help='Use signed gradient loss function')

    # Different levels of verbose
    parser.add_argument('--verbose', action="store_true",
                        help='If true, print everything')
    parser.add_argument('--verbose_pretrain', action="store_true",
                        help='If true, print per-epoch training statistics')
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
    except ValueError:
        raise ValueError("Theta values not provided in correct format")

    if args.verbose:
        args.verbose_pretrain = True
        args.verbose_precomp = True

    # Print all arguments
    utils.flash_utils(args)

    # Get number of classes
    ds_obj = datasets.dataset_helper("memory")(
        path=args.path_1)
    train_loader, test_loader = ds_obj.get_loaders(512)
    n_classes = ds_obj.n_classes

    # Load target model theta_p, set to eval mode
    theta_p = dnn_utils.model_helper(args.model_arch)(n_classes=n_classes)
    theta_p = theta_p.cuda()
    theta_p.load_state_dict(ch.load(args.poison_model_path))
    theta_p.eval()

    # Report performance of poisoned model
    clean_acc, _ = dnn_utils.get_model_metrics(theta_p, test_loader)
    print(utils.yellow_print("[Poisoned-model] Total Acc: %.4f" % clean_acc))
    _, clean_total_loss = dnn_utils.get_model_metrics(theta_p, train_loader)
    print(utils.yellow_print(
        "[Poisoned-model] Loss on train: %.4f" % clean_total_loss))
    # Report weight norm for poisoned model
    poisoned_norm = dnn_utils.get_model_l2_norm(theta_p).item()
    print(utils.yellow_print(
        "[Poisoned-model] Weights norm: %.4f" % poisoned_norm))
    # Report accuracy on unseen population data
    (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
        model=theta_p,
        loader=test_loader,
        target_prop=args.poison_class)
    print(utils.yellow_print(
        "[Poisoned-model] Accuracy on "
        "population test-data: %.4f" % tst_sub_acc))
    print(utils.yellow_print(
        "[Poisoned-model] Accuracy on "
        "non-population test-data: %.4f" % tst_nsub_acc))
    print()

    for valid_theta_err in args.theta_values:
        args.err_threshold = valid_theta_err

        # Prepare logger
        log_dir = os.path.join(args.log_path, str(
            valid_theta_err) +
            "_mnist17split" +
            "_" + args.optim_type +
            "_" + str(args.model_arch) +
            "_" + str(args.n_copies) +
            "_" + str(args.optim_steps) +
            "_" + str(args.optim_trials) +
            "_signed=" + str(args.signed) +
            "_" + str(args.seed))
        utils.ensure_dir_exists(log_dir)
        logger = SummaryWriter(log_dir=log_dir, flush_secs=10)

        print(utils.pink_print(
            "Running attack for theta %.2f" % valid_theta_err))

        # Get poison data
        poison_data, theta_t = modelTargetPoisoning(theta_p, logger, args)
        dp_x = ch.cat(poison_data[0], 0).numpy()
        dp_y = ch.cat(poison_data[1], 0).numpy()

        # Save this data
        poisoned_data_dir = os.path.join(os.path.join(log_dir, "poisondata"))
        np.savez(poisoned_data_dir, x=dp_x, y=dp_y)
        print(utils.pink_print("Saved poisoned data!"))

        # Close logger
        logger.close()
