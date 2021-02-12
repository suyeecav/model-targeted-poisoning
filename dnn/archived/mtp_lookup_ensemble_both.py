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
              tst_sub_acc):
    ol_lr_threshold = args.incre_tol_par

    if args.fixed_budget <= 0:
        if args.require_acc:
            # Define stop condition
            stop_cond = tst_sub_acc > 1 - args.err_threshold
            print(utils.red_print(
                "Current accuracy on population (test): %.4f" % tst_sub_acc))
        else:
            # Define stop condition
            stop_cond = best_loss > ol_lr_threshold
    else:
        stop_cond = num_iters < args.fixed_budget
    return stop_cond


def modelTargetPoisoning(models_p, logger, args):
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
    models_t_pretrained = []
    for seed in args.seeds:
        args.seed = seed
        print(utils.yellow_print(
            "Printing model with seed %d" % args.seed))
        model_t_pretrained, _ = mtp_utils.train_clean_model(ds, args)
        models_t_pretrained.append(model_t_pretrained)

    # Report performance of clean model
    batch_size = len(ds.train)

    train_loader, test_loader = ds.get_loaders(batch_size)
    clean_accs, clean_total_losses = [], []
    population_accs, non_population_accs = [], []
    for model_t_pretrained in models_t_pretrained:
        clean_acc, clean_total_loss = dnn_utils.get_model_metrics(
            model_t_pretrained, test_loader, lossfn=args.loss)
        clean_accs.append(clean_acc)
        clean_total_losses.append(clean_total_loss)

        (population_acc, _), (non_population_acc, _) = dnn_utils.get_model_metrics(
            model_t_pretrained, test_loader,
            lossfn=args.loss, target_prop=args.poison_class)
        population_accs.append(population_acc)
        non_population_accs.append(non_population_acc)

    print(utils.yellow_print(
        "[Clean-model][Test] Total Acc: %.4f" %
        np.mean(clean_accs)))
    print(utils.yellow_print(
        "[Clean-model] Loss on train: %.4f" %
        np.mean(clean_total_losses)))
    print(utils.red_print(
        "[Clean-model][Test] Population Acc: %.4f" %
        np.mean(population_accs)))
    print(utils.red_print(
        "[Clean-model][Test] Non-Population Acc: %.4f" %
        np.mean(non_population_accs)))
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
            batch_size = len(ds.train)
            data_loader, _ = ds.get_loaders(batch_size)

            # Increase numer of iterations theta_t is trained for
            # as size of its training set |D_c U D_p| increases
            iters = args.iters
            if args.increase_iters:
                iters += int((len(ds.train) - len(ds_clean.train)
                              ) / args.increase_every)

            # Construct model
            models_t = []
            for seed in args.seeds:
                args.seed = seed
                model_t = dnn_utils.get_seeded_wrapped_model(
                    args, n_classes=ds.n_classes)
                # Set model to training mode
                model_t.train()

                # Define optimizer
                optim = ch.optim.Adam(model_t.parameters(),
                                      lr=args.pretrain_lr,
                                      weight_decay=args.pretrain_weight_decay)

                # Train model
                print(utils.yellow_print(
                    "Printing model with seed %d" % args.seed))
                for e in range(iters):
                    # Train epoch
                    dnn_utils.epoch(model=model_t, loader=data_loader,
                                    optimizer=optim, epoch_num=e+1,
                                    c_rule=None, n_classes=None,
                                    verbose=True, lossfn=args.loss)

                models_t.append(model_t)
        else:
            models_t = models_t_pretrained

        # Make sure theta_t are in eval mode
        for model_t in models_t:
            model_t.eval()

        # Line 4: Compute (x*, y*)
        (x_opt, y_opt), best_loss = mtp_utils.lookup_based_optimal(
            theta_t=models_t, theta_p=models_p, loader=loader_optim,
            n_classes=ds_second.n_classes, random=args.random,
            lossfn=args.loss, filter=args.filter, verbose=True,
            ensemble_t=True, ensemble_p=True,
            pick_optimal=args.pick_optimal)

        # Log some information about x*, y*
        with ch.no_grad():
            preds_p = [str(model_p(x_opt).argmax(1).item())
                       for model_p in models_p]
            preds_t = [str(model_t(x_opt).argmax(1).item())
                       for model_t in models_t]
        print(utils.cyan_print("Loss: %.3f Mt(x*): %s, Mp(x*): %s, y*: %d" %
                               (best_loss.item(), ",".join(preds_t),
                                ",".join(preds_p), y_opt)))

        # Line 5: Add (x*, y*) to D_p
        for _ in range(args.n_copies):
            D_p[0].append(x_opt.cpu())
            D_p[1].append(y_opt.cpu())
            ds.add_point_to_train(x_opt.cpu(), y_opt.cpu())
        print()

        # Calculate useful statistics
        tst_sub_accs, tst_nsub_accs = [], []
        trn_sub_accs, trn_nsub_accs = [], []
        for model_t in models_t:
            (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
                model=model_t,
                loader=test_loader,
                target_prop=args.poison_class,
                lossfn=args.loss)
            tst_sub_accs.append(tst_sub_acc)
            tst_nsub_accs.append(tst_nsub_acc)

            (trn_sub_acc, _), (trn_nsub_acc, _) = dnn_utils.get_model_metrics(
                model=model_t,
                loader=train_loader,
                target_prop=args.poison_class,
                lossfn=args.loss)
            trn_sub_accs.append(trn_sub_acc)
            trn_nsub_accs.append(trn_nsub_acc)

        # Get mean of these metrics
        trn_sub_acc = np.mean(trn_sub_accs)
        tst_sub_acc = np.mean(tst_sub_accs)
        trn_nsub_acc = np.mean(trn_nsub_accs)
        tst_nsub_acc = np.mean(tst_nsub_accs)

        # Log information
        mtp_utils.log_information(
            logger=logger, best_loss=best_loss,
            x_opt=x_opt, norm_diffs=None,
            trn_sub_acc=trn_sub_acc, trn_nsub_acc=trn_nsub_acc,
            tst_sub_acc=tst_sub_acc, tst_nsub_acc=tst_nsub_acc,
            num_iters=num_iters + 1, args=args, label=y_opt)

        # Line 6: Get ready to check condition
        condition = stop_cond(args=args, best_loss=best_loss,
                              num_iters=num_iters,
                              tst_sub_acc=tst_sub_acc)

        # Keep track of no. of iterations
        num_iters += 1

    # Line 7: Return poison data
    return D_p, models_t


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
    parser.add_argument('--poison_arch', default='lenet',
                        choices=dnn_utils.get_model_names(),
                        help='Victim model architecture')
    parser.add_argument('--batch_size', default=-1, type=int,
                        help='Batch size while training models')
    parser.add_argument('--online_alg_criteria', default='max_loss',
                        choices=['max_loss'],
                        help='Stop criteria of online alg: max_loss or norm')
    parser.add_argument('--poison_model_dir', type=str,
                        help='Path to saved poisoned-classifiers')
    parser.add_argument('--log_path', type=str,
                        default="./data/logs_pick_ensemble_both",
                        help='Path to save logs')
    parser.add_argument('--theta_values', type=str,
                        default="0.95",
                        help='Comma-separated list of theta values to try')
    parser.add_argument('--poison_class', default=0,
                        type=int,
                        choices=list(range(10)),
                        help='Which class to target for corruption')
    parser.add_argument('--loss', default="ce",
                        choices=["ce", "hinge"],
                        help='Loss function to use while training models')

    # Params for pre-training model
    parser.add_argument('--pretrain_weight_decay', default=0.05,
                        type=float, help='Weight decay while pre-training')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Epochs while pre-training models')
    parser.add_argument('--pretrain_lr', default=2e-3, type=float,
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
    parser.add_argument('--seeds', default="80346,16,2021,4,793",
                        help='Seeds for weight init (# of seeds = # of models)')
    parser.add_argument('--increase_iters', action="store_true",
                        help='Use more train iterations as |D_p| increases')
    parser.add_argument('--skip_bad', action="store_true",
                        help='Do not consider points that end up'
                             'increasing loss')
    parser.add_argument('--increase_every', default=1000, type=int,
                        help='Add epoch for every # points')

    # Params for (x*, y*) computation
    parser.add_argument('--n_copies', default=10, type=int,
                        help='Number of copies per (x*,y*) to be added')
    parser.add_argument('--random', action="store_true",
                        help='Use random selection for points')
    parser.add_argument('--pick_optimal', action="store_true",
                        help='Use max over theta_t instead of mean')
    parser.add_argument('--filter', action="store_true",
                        help='Apply filter when picking x*, y*')
    parser.add_argument('--path_1', default="./data/datasets/MNIST17/split_1.pt",
                        help='Path to first split of dataset')
    parser.add_argument('--path_2', default="./data/datasets/MNIST17/split_2.pt",
                        help='Path to second split of dataset')

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

    # Make sure seeds provided in valid format
    try:
        seeds = [int(x)
                 for x in args.seeds.replace(" ", "").split(',')]
        assert len(seeds) > 0, 'Provide at least one seed'
        args.seeds = seeds
    except ValueError:
        raise ValueError("Seeds not provided in correct format")

    if args.verbose:
        args.verbose_pretrain = True
        args.verbose_precomp = True

    # Print all arguments
    utils.flash_utils(args)

    # Get number of classes
    n_classes = 2

    # Report performance of poisoned model
    train_loader, test_loader = datasets.dataset_helper("memory")(
        path=args.path_1).get_loaders(512)

    # Load target models theta_p, set to eval mode
    thetas_p = []
    for i, mp in enumerate(os.listdir(args.poison_model_dir)):
        tp = dnn_utils.model_helper(args.poison_arch)(n_classes=n_classes)
        tp = tp.cuda()
        tp.load_state_dict(ch.load(os.path.join(args.poison_model_dir, mp)))
        tp.eval()
        thetas_p.append(tp)

        # Report performance of poisoned model
        clean_acc, _ = dnn_utils.get_model_metrics(tp, test_loader)
        print(utils.yellow_print(
            "[Poisoned-model %d] Total Acc: %.4f" % (i+1, clean_acc)))
        _, clean_total_loss = dnn_utils.get_model_metrics(
            tp, train_loader)
        print(utils.yellow_print(
            "[Poisoned-model %d] Loss on train: %.4f" % (i+1, clean_total_loss)))
        # Report weight norm for poisoned model
        poisoned_norm = dnn_utils.get_model_l2_norm(tp).item()
        print(utils.yellow_print(
            "[Poisoned-model %d] Weights norm: %.4f" % (i+1, poisoned_norm)))
        # Report accuracy on unseen population data
        (tst_sub_acc, _), (tst_nsub_acc, _) = dnn_utils.get_model_metrics(
            model=tp,
            loader=test_loader,
            target_prop=args.poison_class)
        print(utils.yellow_print(
            "[Poisoned-model %d] Accuracy on "
            "population test-data: %.4f" % (i+1, tst_sub_acc)))
        print(utils.yellow_print(
            "[Poisoned-model %d] Accuracy on "
            "non-population test-data: %.4f" % (i+1, tst_nsub_acc)))
        print()
    
    for valid_theta_err in args.theta_values:
        args.err_threshold = valid_theta_err

        # Prepare logger
        log_dir = os.path.join(args.log_path, str(
            valid_theta_err) +
            "_mnist17split" +
            "_" + str(args.model_arch) +
            "_" + str(args.n_copies) +
            "_" + str(args.seeds))
        utils.ensure_dir_exists(log_dir)
        logger = SummaryWriter(log_dir=log_dir, flush_secs=10)

        print(utils.pink_print(
            "Running attack for theta %.2f" % valid_theta_err))

        # Get poison data
        poison_data, theta_t = modelTargetPoisoning(thetas_p, logger, args)
        dp_x = ch.cat(poison_data[0], 0).numpy()
        dp_y = ch.cat(poison_data[1], 0).numpy()

        # Save this data
        poisoned_data_dir = os.path.join(os.path.join(log_dir, "poisondata"))
        np.savez(poisoned_data_dir, x=dp_x, y=dp_y)
        print(utils.pink_print("Saved poisoned data!"))

        # Close logger
        logger.close()
