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
    ds = datasets.dataset_helper(args.dataset)()

    # Maintain copy of clean data (for seed sampling)
    ds_clean = datasets.dataset_helper(args.dataset)()

    # Line 1: Collect poisoning points
    D_p = [[], []]

    # Load poison data, if provided
    if args.poison_data:
        print(utils.green_print("Loading poison data"))
        data = np.load("./data/poison_data/poison_data.npz")
        # Normalize to 0-1 for use by model
        all_poison_data_x = ch.from_numpy(data['x']).float() / 255.
        all_poison_data_x = ch.unsqueeze(all_poison_data_x, 1)
        all_poison_data_y = ch.from_numpy(data['y'])

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
    prev_loss, best_loss = np.inf, np.inf
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

        start_with = None
        if args.start_opt_real:
            # If flag set, start with real data sampled from
            # (unpoisoned) train loader
            batch_size = args.batch_size
            if batch_size == -1:
                batch_size = len(ds.train)
            loader, _ = ds_clean.get_loaders(batch_size)
            start_with = datasets.get_sample_from_loader(
                loader, args.trials, ds_clean.n_classes)
        elif args.poison_data:
            # Sample 'num-trials' data from this
            perm = ch.randperm(all_poison_data_x.size(0))
            idx = perm[:args.trials]
            start_with = (all_poison_data_x[idx], all_poison_data_y[idx])

        # Line 4: Compute (x*, y*)
        if args.use_optim_for_optimal:
            find_optimal_function = mtp_utils.find_optimal_using_optim
        else:
            find_optimal_function = mtp_utils.find_optimal

        (x_opt, y_opt), best_loss = find_optimal_function(
            theta_t=model_t, theta_p=model_p, input_shape=ds.datum_shape,
            n_classes=ds.n_classes, trials=args.trials,
            num_steps=args.num_steps, step_size=args.optim_lr,
            verbose=True, start_with=start_with,
            lossfn=args.loss, dynamic_lr=args.dynamic_lr,
            filter=args.filter)

        # If loss increased, try optimization once more
        # With double trials, to reduce chance of bad minima
        if args.skip_bad and best_loss > prev_loss:
            print(utils.red_print("Re-running optimization with more seeds"))
            (x_opt, y_opt), best_loss = find_optimal_function(
                theta_t=model_t, theta_p=model_p, input_shape=ds.datum_shape,
                n_classes=ds.n_classes, trials=args.trials * 2,
                num_steps=args.num_steps, step_size=args.optim_lr,
                verbose=True, start_with=start_with,
                lossfn=args.loss, dynamic_lr=args.dynamic_lr)

        # Log some information about x*, y*
        with ch.no_grad():
            pred_t, pred_p = model_t(x_opt), model_p(x_opt)
            if pred_t.argmax(1) == y_opt.item():
                print(utils.red_print("[BAD OPTIMIZATION. CHECK]"))
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
            x_opt=x_opt, model_t=model_t, norm_diffs=norm_diffs,
            trn_sub_acc=trn_sub_acc, trn_nsub_acc=trn_nsub_acc,
            tst_sub_acc=tst_sub_acc, tst_nsub_acc=tst_nsub_acc,
            num_iters=num_iters + 1, args=args)

        # Line 6: Get ready to check condition
        condition = stop_cond(args=args, best_loss=best_loss,
                              num_iters=num_iters,
                              tst_sub_acc=tst_sub_acc,
                              norm_diffs=norm_diffs)

        # Keep track of no. of iterations
        num_iters += 1

        # Keep track of loss from previous iteration
        prev_loss = best_loss.item()        

    # Line 7: Return poison data
    return D_p, model_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ol: target classifier is from the adapttive attack
    # kkt: target is from kkt attack
    # real: actual classifier, compare: compare performance
    # of kkt attack and adaptive attack using same stop criteria

    # Global params
    parser.add_argument('--model_arch', default='lenet_bn',
                        choices=dnn_utils.get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--dataset', default='mnist',
                        choices=datasets.get_dataset_names(),
                        help="Which dataset to use?")
    parser.add_argument('--batch_size', default=-1, type=int,
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
                        default="0.7",
                        help='Comma-separated list of theta values to try')
    parser.add_argument('--poison_class', default=4,
                        type=int,
                        choices=list(range(10)),
                        help='Which class to target for corruption')
    parser.add_argument('--loss', default="ce",
                        choices=["ce", "hinge"],
                        help='Loss function to use while training models')

    # Sample from data that was used for poisoning, if provided
    parser.add_argument('--poison_data', action="store_true",
                        help='Load poisoned data, use that as starting point')

    # Params for pre-training model
    parser.add_argument('--pretrain_weight_decay', default=0.05,
                        type=float, help='Weight decay while pre-training')
    parser.add_argument('--epochs', default=15, type=int,
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
    parser.add_argument('--start_opt_real', action="store_true",
                        help="If true, initialize (x*, y*) "
                        "with actual data instead of random")

    # Params for computing optimal w_t
    parser.add_argument('--iters', default=15, type=int,
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
    parser.add_argument('--trials', default=50, type=int,
                        help='Number of trials while searching for x*, y*')
    parser.add_argument('--n_copies', default=1, type=int,
                        help='Number of copies per (x*,y*) to be added')
    parser.add_argument('--num_steps', default=2000, type=int,
                        help='Number of steps while searching for x*, y*')
    parser.add_argument('--optim_lr', default=1e-2, type=float,
                        help='Learning rate while searching for x*, y*')
    parser.add_argument('--use_optim_for_optimal', action="store_true",
                        help='Use optimizer (ADAM) to search for x*, y*')
    parser.add_argument('--dynamic_lr', action="store_true",
                        help='Use scheduler to reduce LR on plateau')
    parser.add_argument('--filter', action="store_true",
                        help='Apply filter when picking x*, y*')

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

    if args.dynamic_lr and (not args.use_optim_for_optimal):
        raise ValueError("Dynamic LR only supported for optimizer currently")

    if args.skip_bad and (args.start_opt_real or args.poison_data):
        raise ValueError("Re-run and real/poison start data not supported yet.")

    # Print all arguments
    utils.flash_utils(args)

    # Get number of classes
    n_classes = datasets.dataset_helper(args.dataset)().n_classes

    # Load target model theta_p, set to eval mode
    theta_p = dnn_utils.model_helper(args.model_arch)(n_classes=n_classes)
    theta_p = theta_p.cuda()
    theta_p.load_state_dict(ch.load(args.poison_model_path))
    theta_p.eval()

    # Report performance of poisoned model
    train_loader, test_loader = datasets.dataset_helper(
        args.dataset)().get_loaders(512)
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
            "_" + args.dataset +
            "_" + str(args.n_copies) +
            "_" + str(args.trials))
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
