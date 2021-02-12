import dnn_utils
import torch.autograd as autograd
import torch.nn as nn
import torch as ch
import numpy as np
import mtp_utils
from datasets import dataset_helper, get_sample_from_loader
import argparse
import utils
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import copy


def find_optimal_using_optim(theta_t, input_shape, n_classes,
                             trials=10, num_steps=3000, step_size=1e-2,
                             verbose=True, start_with=None,
                             lossfn="ce"):
    # Put model in evaluation mode
    theta_t.eval()
    best_loss, prev_loss = -np.inf, np.inf

    if verbose:
        print(utils.yellow_print("[Computing optimal x*, y*]"))

    if start_with is None:
        # Initialize with random data
        x_ = ch.rand(*((n_classes * trials,) + input_shape)).cuda()

        # Array of all possible labels
        y = ch.arange(n_classes, dtype=ch.long).cuda()
        y = y.repeat_interleave(trials)
    else:
        # Start with provided data
        x_, y = start_with[0].cuda(), start_with[1].cuda()

    x_ = autograd.Variable(x_.clone(), requires_grad=True)
    optim = ch.optim.Adam([x_], lr=step_size, weight_decay=0)

    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        optim.zero_grad()

        # Compute loss to optimize
        if lossfn == "ce":
            ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        else:
            ce_loss = nn.MultiMarginLoss(reduction='none').cuda()

        # Using in-built optimizer that minimizes
        # So negate loss
        logits = theta_t(x_)
        loss = - ce_loss(logits, y)

        if verbose:
            with ch.no_grad():
                iterator.set_description(
                    "Mean loss: %.4f, Max Loss: %.4f" % ((-loss).mean(),
                                                         (-loss).max()))

        # Compute gradients
        loss = ch.mean(loss)
        loss.backward()
        optim.step()

        with ch.no_grad():
            # Clip data back to [0, 1] range
            x_.data = ch.clamp(x_.data, 0, 1)

            # Compute loss again (on clipped data)
            logits = theta_t(x_)
            max_loss_real = ce_loss(logits, y)

            # Compute best target class
            max_loss_real_across, max_loss_index = max_loss_real.max(0)

            # Keep track of best loss, class
            if best_loss < max_loss_real_across:
                best_loss = max_loss_real_across
                best_x = x_[max_loss_index].data.clone()
                best_y = y[max_loss_index]

        # Stop optimization if loss seems to have converged
        this_iter_loss_on_clipped_data = max_loss_real.mean().item()
        if np.abs(prev_loss - this_iter_loss_on_clipped_data) < 1e-7:
            print("(x*,y*) loss stagnation", prev_loss,
                  this_iter_loss_on_clipped_data)
            break

        # Keep track of this loss for next iteration comparison
        prev_loss = this_iter_loss_on_clipped_data

    best_x = best_x.detach().unsqueeze(0)
    best_y = ch.ones((1,), dtype=ch.long).cuda() * best_y.detach().item()
    return (best_x, best_y), best_loss


def indiscriminateAttack(logger, wanted_errors, args):
    # Fetch appropriate dataset
    ds = dataset_helper(args.dataset)()

    # Maintain copy of clean data (for seed sampling)
    ds_clean = dataset_helper(args.dataset)()

    # Line 1: Collect poisoning points
    D_p = [[], []]

    # Line 3: Since D_p is empty in first iteration, simply train it outside
    model_t_pretrained, pretrain_optim = mtp_utils.train_clean_model(ds, args)

    # Report performance of clean model
    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = len(ds.train)

    train_loader, test_loader = ds.get_loaders(batch_size)
    clean_acc, _ = dnn_utils.get_model_metrics(
        model_t_pretrained, test_loader, lossfn=args.loss)
    print(utils.yellow_print("[Clean-model] Total Acc: %.4f" % clean_acc))
    _, clean_total_loss = dnn_utils.get_model_metrics(
        model_t_pretrained, train_loader, lossfn=args.loss)
    print(utils.yellow_print(
        "[Clean-model] Loss on train: %.4f" % clean_total_loss))
    print()

    # Keep track of which errors have been achieved so far
    achieved_so_far = 0

    # Line 2: Iterate until stopping criteria met
    best_loss = np.inf
    num_iters = 0
    while achieved_so_far < len(wanted_errors):

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
        # If flag set, start with real data sampled from
        # (unpoisoned) train loader
        if args.start_opt_real:
            batch_size = args.batch_size
            if batch_size == -1:
                batch_size = len(ds.train)
            loader, _ = ds_clean.get_loaders(batch_size)
            start_with = get_sample_from_loader(
                loader, args.trials, ds_clean.n_classes)

        # Line 4: Compute (x*, y*)
        (x_opt, y_opt), best_loss = find_optimal_using_optim(
            theta_t=model_t, input_shape=ds.datum_shape,
            n_classes=ds.n_classes, trials=args.trials,
            num_steps=args.num_steps, step_size=args.optim_lr,
            verbose=True, start_with=start_with,
            lossfn=args.loss)

        # Log some information about x*, y*
        with ch.no_grad():
            pred_t = model_t(x_opt)
        print(utils.cyan_print("Mt(x*): %d, y*: %d" %
                               (pred_t.argmax(1), y_opt)))

        # Line 5: Add (x*, y*) to D_p
        for _ in range(args.n_copies):
            D_p[0].append(x_opt.cpu())
            D_p[1].append(y_opt.cpu())
            ds.add_point_to_train(x_opt.cpu(), y_opt.cpu())
        print()

        # Calculate useful statistics
        (tst_acc, _) = dnn_utils.get_model_metrics(
            model=model_t,
            loader=test_loader,
            lossfn=args.loss)
        (trn_acc, _) = dnn_utils.get_model_metrics(
            model=model_t,
            loader=train_loader,
            lossfn=args.loss)

        # Log information
        # Log optimized image
        logger.add_image("X*", x_opt[0], (num_iters + 1) * args.n_copies)
        # Log weight Norm
        logger.add_scalar("Weight norm", dnn_utils.get_model_l2_norm(
            model_t).item(), global_step=(num_iters + 1) * args.n_copies)
        # Log population accuracies on train, test data
        logger.add_scalar("[Train] Accuracy", trn_acc,
                          global_step=(num_iters + 1) * args.n_copies)
        logger.add_scalar("[Test] Accuracy", tst_acc,
                          global_step=(num_iters + 1) * args.n_copies)
        # Log best loss
        logger.add_scalar("Loss on x*,y*", best_loss.item(),
                          global_step=(num_iters + 1) * args.n_copies)

        # Keep track of no. of iterations
        num_iters += 1

        # If wanted error achieved, switch to next goal:
        if (1 - trn_acc) > wanted_errors[achieved_so_far]:
            # Save current model
            model_name = "seed-{}_error-{}_tacc-{}.pth".format(
                args.seed, wanted_errors[achieved_so_far], tst_acc)
            ch.save(copy.deepcopy(model_t).state_dict(),
                    os.path.join(args.save_dir, model_name))
            print(utils.pink_print("Achieved %.3f loss!" %
                                   wanted_errors[achieved_so_far]))
            achieved_so_far += 1


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
                        choices=['mnist', 'mnist17'],
                        help="Which dataset to use?")
    parser.add_argument('--batch_size', default=-1, type=int,
                        help='Batch size while training models')
    parser.add_argument('--log_path', type=str,
                        default="./data/logs",
                        help='Path to save logs')
    parser.add_argument('--loss', default="ce",
                        choices=["ce", "hinge"],
                        help='Loss function to use while training models')
    parser.add_argument('--errors', default="0.05,0.1,0.15,0.2,0.25,0.3,0.35",
                        help='Errors to achieve and save models')
    parser.add_argument('--save_dir',
                        help='Directory to save models in')

    # Params for pre-training model
    parser.add_argument('--pretrain_weight_decay', default=0.05,
                        type=float, help='Weight decay while pre-training')
    parser.add_argument('--epochs', default=15, type=int,
                        help='Epochs while pre-training models')
    parser.add_argument('--pretrain_lr', default=1e-2, type=float,
                        help='Learning rate for models')

    # some params related to online algorithm, use the default
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
    parser.add_argument('--increase_every', default=1000, type=int,
                        help='Add epoch for every # points')

    # Params for (x*, y*) computation
    parser.add_argument('--trials', default=200, type=int,
                        help='Number of trials while searching for x*, y*')
    parser.add_argument('--n_copies', default=1, type=int,
                        help='Number of copies per (x*,y*) to be added')
    parser.add_argument('--num_steps', default=2000, type=int,
                        help='Number of steps while searching for x*, y*')
    parser.add_argument('--optim_lr', default=5e-3, type=float,
                        help='Learning rate while searching for x*, y*')

    # Different levels of verbose
    parser.add_argument('--verbose', action="store_true",
                        help='If true, print everything')
    parser.add_argument('--verbose_pretrain', action="store_true",
                        help='If true, print per-epoch training statistics')

    args = parser.parse_args()

    if args.verbose:
        args.verbose_pretrain = True

    try:
        wanted_errors = [float(x) for x in args.errors.split(",")]
        print(utils.red_print("Target error rates: %s" % str(wanted_errors)))
    except ValueError:
        raise ValueError("Wanted errors provided in invalid format")

    # Ensure directory exists where model will be saved
    utils.ensure_dir_exists(args.save_dir)

    # Print all arguments
    utils.flash_utils(args)

    # Prepare logger
    log_dir = os.path.join(args.log_path, "indiscriminate_" +
                           args.dataset + "_" +
                           str(args.n_copies) + "_" +
                           str(args.trials) + "_" +
                           str(args.seed))
    utils.ensure_dir_exists(log_dir)
    logger = SummaryWriter(log_dir=log_dir, flush_secs=10)

    print(utils.pink_print("Running attack "))
    indiscriminateAttack(logger, wanted_errors, args)

    # Close logger
    logger.close()
