import torch as ch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.autograd as autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dnn_utils
import utils
from datasets import get_dataset_gradients


def compute_loss(x, y, theta_t, theta_p, lossfn="ce",
                 ensemble_p=False, ensemble_t=False,
                 specific=None, pick_optimal=False):
    if lossfn == "ce":
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.MultiMarginLoss(reduction='none')

    # Shift loss-fn to GPU
    if specific is None:
        loss_fn = loss_fn.cuda()
    else:
        loss_fn.to(ch.device(specific))

    # Get loss on theta_p
    if ensemble_p:
        loss_p = ch.stack([loss_fn(tp(x), y) for tp in theta_p])
        if pick_optimal:
            loss_p = ch.min(loss_p, 0).values
        else:
            loss_p = ch.mean(loss_p, 0)
    else:
        loss_p = loss_fn(theta_p(x), y)

    # Get loss on theta_t
    if ensemble_t:
        loss_t = ch.stack([loss_fn(tt(x), y) for tt in theta_t])
        if pick_optimal:
            loss_t = ch.max(loss_t, 0).values
        else:
            loss_t = ch.mean(loss_t, 0)
    else:
        loss_t = loss_fn(theta_t(x), y)

    return loss_t, loss_p


def find_optimal(theta_t, theta_p, input_shape, n_classes,
                 trials=10, num_steps=3000, step_size=1e-2,
                 verbose=True, start_with=None,
                 lossfn="ce", dynamic_lr=None,
                 ensemble_p=False, filter=False):

    # Put both models in evaluation mode
    theta_t.eval()
    if ensemble_p:
        for tp in theta_p:
            tp.eval()
    else:
        theta_p.eval()

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

    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        x_ = x_.clone().detach().requires_grad_(True)

        # Compute loss to optimize
        loss_t, loss_p = compute_loss(x=x_, y=y,
                                      theta_t=theta_t,
                                      theta_p=theta_p,
                                      lossfn=lossfn,
                                      ensemble_p=ensemble_p)
        loss = loss_t - loss_p

        if verbose:
            with ch.no_grad():
                look_zero = ch.nonzero(y == 0).squeeze_(1)
                look_one = ch.nonzero(y == 1).squeeze_(1)
                best_lozz_zero = (loss[look_zero])
                best_lozz_one = (loss[look_one])
                iterator.set_description(
                    "[0] max: %.2f, mean: %.2f [1] max: %.2f, mean: %.2f"
                    % (best_lozz_zero.max(), best_lozz_zero.mean(),
                       best_lozz_one.max(), best_lozz_one.mean()))

        # Compute gradients
        loss = ch.sum(loss)  # Also try out mean?
        grad, = ch.autograd.grad(loss, [x_])

        with ch.no_grad():  # Take gradient-ascent step
            x_ = x_ + (step_size * grad)

            # Clip data back to [0, 1] range
            x_ = ch.clamp(x_, 0, 1)

            # Compute loss again (on clipped data)
            max_loss_tup = compute_loss(
                x=x_, y=y,
                theta_t=theta_t,
                theta_p=theta_p,
                lossfn=lossfn,
                ensemble_p=ensemble_p)
            max_loss_real = max_loss_tup[0] - max_loss_tup[1]

            # Compute best target class
            max_loss_real_across, max_loss_index = max_loss_real.max(0)

            # Keep track of best loss, class
            if best_loss < max_loss_real_across:
                best_loss = max_loss_real_across
                best_x = x_[max_loss_index].data.clone()
                best_y = y[max_loss_index]

        # Stop optimization if loss seems to have converged
        this_iter_loss_on_clipped_data = max_loss_real.mean().item()
        if np.abs(prev_loss - this_iter_loss_on_clipped_data) < 1e-6:
            print("(x*,y*) loss stagnation", prev_loss,
                  this_iter_loss_on_clipped_data)
            break

        # Keep track of this loss for next iteration comparison
        prev_loss = this_iter_loss_on_clipped_data

    best_x = best_x.detach().unsqueeze(0)
    best_y = ch.ones((1,), dtype=ch.long).cuda() * best_y.detach().item()
    return (best_x, best_y), best_loss


def find_optimal_using_optim(theta_t, theta_p, input_shape, n_classes,
                             trials=10, num_steps=3000, step_size=1e-2,
                             verbose=True, start_with=None,
                             lossfn="ce", dynamic_lr=False,
                             ensemble_p=False, filter=False):
    # Put both models in evaluation mode
    theta_t.eval()
    if ensemble_p:
        for tp in theta_p:
            tp.eval()
    else:
        theta_p.eval()

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
    if dynamic_lr:
        scheduler = ReduceLROnPlateau(
            optim, mode='max', factor=0.5,
            cooldown=15,
            patience=50,
            min_lr=1e-2,
            threshold=1e-2,
            threshold_mode='abs',)

    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        optim.zero_grad()

        # Compute loss to optimize
        loss_t, loss_p = compute_loss(x=x_, y=y,
                                      theta_t=theta_t,
                                      theta_p=theta_p,
                                      lossfn=lossfn,
                                      ensemble_p=ensemble_p)
        # Using in-built optimizer that minimizes
        # So negate loss
        loss = loss_p - loss_t

        if verbose:
            with ch.no_grad():
                useful = ch.mean(1. * (ch.argmax(theta_t(x_), 1) != y)).item()
                iterator.set_description(
                    "(mean) loss diff : %.3f, (mean) Loss_t: %.3f,"
                    " (mean) Loss_p: %.3f | useful: %.2f" %
                    ((-loss).mean(), loss_t.mean(), loss_p.mean(), useful))

        # Compute gradients
        loss = ch.mean(loss)
        loss.backward()
        optim.step()

        with ch.no_grad():
            # Clip data back to [0, 1] range
            x_.data = ch.clamp(x_.data, 0, 1)

            # Compute loss again (on clipped data)
            max_loss_tup = compute_loss(
                x=x_, y=y,
                theta_t=theta_t,
                theta_p=theta_p,
                lossfn=lossfn,
                ensemble_p=ensemble_p)
            max_loss_real = max_loss_tup[0] - max_loss_tup[1]

            # If filter requested, only look at data where
            # Prediction of point added is different
            if filter:
                interest = ch.nonzero(
                    ch.argmax(theta_t(x_), 1) == y).squeeze_(1)
                max_loss_real[interest] = -np.inf

            # Compute best target class
            max_loss_real_across, max_loss_index = max_loss_real.max(0)

            # Keep track of best loss, class
            if best_loss < max_loss_real_across:
                best_loss = max_loss_real_across
                best_x = x_[max_loss_index].data.clone()
                best_y = y[max_loss_index]

        # Stop optimization if loss seems to have converged
        if np.abs(prev_loss - max_loss_real_across.item()) < 1e-7:
            print("(x*,y*) loss stagnation", prev_loss,
                  max_loss_real_across.item())
            break

        # Keep track of this loss for next iteration comparison
        prev_loss = max_loss_real_across.item()

        if dynamic_lr:
            # Update scheduler
            scheduler.step(prev_loss)

    best_x = best_x.detach().unsqueeze(0)
    best_y = ch.ones((1,), dtype=ch.long).cuda() * best_y.detach().item()
    return (best_x, best_y), best_loss


def lookup_based_optimal(theta_t, theta_p, loader,
                         lossfn="ce", filter=False,
                         n_classes=2, ensemble_p=False,
                         verbose=True, random=False,
                         ensemble_t=False, pick_optimal=False):
    # Makee sure models are in eval mode
    if ensemble_t:
        for tt in theta_t:
            tt.eval()
    else:
        theta_t.eval()
    if ensemble_p:
        for tp in theta_p:
            tp.eval()
    else:
        theta_p.eval()

    # Go through data loader
    # Pick x*, y* that maximizes loss
    best_loss, best_x, best_y = -np.inf, None, None
    if verbose:
        loader = tqdm(loader)

    for x, y_gt in loader:
        y = ch.arange(n_classes, dtype=ch.long)
        y = y.repeat_interleave(x.shape[0]).cuda()
        x = x.repeat(n_classes, 1, 1, 1).cuda()
        y_gt = y_gt.repeat(n_classes).cuda()

        with ch.no_grad():
            loss_t, loss_p = compute_loss(x=x, y=y,
                                          theta_t=theta_t,
                                          theta_p=theta_p,
                                          lossfn=lossfn,
                                          ensemble_p=ensemble_p,
                                          ensemble_t=ensemble_t,
                                          pick_optimal=pick_optimal)
            loss = loss_t - loss_p

            if filter:
                # Only look at images that are actually '7'
                # And we want to label as '1'
                interest = ch.nonzero(ch.logical_or(
                    y_gt == 1, y == 0)).squeeze_(1)
                loss[interest] = -np.inf

            # If randomness requested, and point not picked so far
            if random:
                # Pick a batch randomly
                if np.random.uniform() < 1 / len(loader):
                    # And from that batch, pick a point
                    # That satisfies filter condition
                    # And return that
                    interest = ch.nonzero(loss != -np.inf).squeeze_(1)
                    picked = interest[ch.randperm(interest.size(0))[0]]
                    best_loss = loss[picked]
                    best_x = x[picked]
                    best_y = y[picked]
                    break

            max_loss, max_loss_index = loss.max(0)
            if max_loss > best_loss:
                best_loss = max_loss
                best_x = x[max_loss_index]
                best_y = y[max_loss_index]

    best_x = best_x.unsqueeze(0)
    best_y = ch.ones((1,), dtype=ch.long).cuda() * best_y.item()

    return (best_x, best_y), best_loss


def dataset_gradient_loss(gradients, model_diff_grads, dataset_grads):
    # cosine similarity as the loss function
    cos = ch.nn.CosineSimilarity(dim=0, eps=1e-6)
    totcost = 0.
    indies = []
    weights = [1, 1, 1, 1, 1, 1]
    for i in range(len(gradients)):
        cost = cos(model_diff_grads[i].flatten(
        ), dataset_grads[i].flatten() + gradients[i].flatten())

        # cost = cos(gradients[i].flatten(
        # ), model_diff_grads[i].flatten() + dataset_grads[i].flatten())
        # We care about gradients only when they contribute to loss
        totcost += (1 - cost) #* weights[i]
        indies.append(cost.item())
    return totcost / len(gradients), indies


def dataset_grad_optimal(theta_t, theta_p, input_shape, n_classes,
                          trials=10, num_steps=200, step_size=1e-2,
                          verbose=True, start_with=None, dynamic_lr=False,
                          ensemble_p=False, ds=None):
    
    # Put both models in evaluation mode
    theta_t.eval()
    if ensemble_p:
        for tp in theta_p:
            tp.eval()
    else:
        theta_p.eval()

    best_loss, prev_loss = np.inf, -np.inf

    if verbose:
        print(utils.yellow_print("[Computing optimal x*, y*]"))
    
    if start_with is None:
        # Initialize with random data
        x_s = ch.rand(*((n_classes * trials,) + input_shape)).cuda()

        # Array of all possible labels
        ys = ch.arange(n_classes, dtype=ch.long).cuda()
        ys = ys.repeat_interleave(trials)
    else:
        # Start with provided data
        x_s, ys = start_with[0].cuda(), start_with[1].cuda()

    # get summed gradients from theta_t to theta_p 
    theta_t_ws = list(utils.get_relevant_params(theta_t.named_parameters()))
    theta_p_ws = list(utils.get_relevant_params(theta_p.named_parameters()))

    weight_diffs = []
    for i in range(len(theta_t_ws)):
        weight_diff = (theta_t_ws[i] - theta_p_ws[i])
        weight_diff = weight_diff.clone().detach().requires_grad_(False)
        weight_diffs.append(weight_diff)
    
    # Compute dataset gradients
    # Can be done more efficiently by caching gradients on D_c, but we will worry about that later
    dataset_gradients = get_dataset_gradients(
        model=theta_t, ds=ds, batch_size=1024,
        weight_decay=None, verbose=False, is_train=True,
        negate=False)
    
    # Loss we care about
    ce_loss = nn.CrossEntropyLoss(reduction='mean').cuda()

    # start the optimization process
    for j in range(n_classes * trials):
        x_ = x_s[j:j+1]
        y = ys[j:j+1]
        
        x_ = autograd.Variable(x_.clone(), requires_grad=True)
        optim = ch.optim.Adam([x_], lr=step_size, weight_decay=0)

        if dynamic_lr:
            scheduler = ReduceLROnPlateau(
                optim, mode='max', factor=0.5,
                cooldown=15,
                patience=50,
                min_lr=1e-2,
                threshold=1e-2,
                threshold_mode='abs',)
        
        iterator = range(num_steps)
        if verbose:
            iterator = tqdm(iterator)
        
        for i in iterator:
            theta_t.zero_grad()
            optim.zero_grad()
            # Compute gradients of current poison point w.r.t. curr model
            loss = ce_loss(theta_t(x_), y)

            # grads, = ch.autograd.grad(loss, [x_],retain_graph=True, create_graph=True)
            grads = list(ch.autograd.grad(loss, theta_t.parameters(), retain_graph=True, create_graph=True))

            # reconstruction loss for maximizing the cosine similarity
            reconst_loss, indies = dataset_gradient_loss(
                grads, weight_diffs, dataset_gradients)

            if verbose:
                with ch.no_grad():
                    cosines = ",".join(["%.5f" % x for x in indies])
                    iterator.set_description(
                            "[%d] loss: %.4f | %s"
                            % (y.item(), reconst_loss.item(), cosines))
            
            # compute gradients w.r.t x data
            # reconst_loss = ch.sum(reconst_loss)
            reconst_loss.backward()
            optim.step()

            with ch.no_grad(): 
                # Clip data back to [0, 1] range
                # x_ = ch.clamp(x_, 0, 1) # this will actally cause vanishing gradients 
                x_.data = ch.clamp(x_.data, 0, 1)

            # check the similarity again on the clipped data and pick best one
            loss_new = ce_loss(theta_t(x_), y)
            
            # grads_new, = ch.autograd.grad(loss_new, [x_])
            grads_new = ch.autograd.grad(loss_new, theta_t.parameters())
            # reconstruction loss for maximizing the cosine similarity
            reconst_loss_new, _ = dataset_gradient_loss(grads_new, weight_diffs, dataset_gradients)

            # Keep track of best loss, class
            if best_loss > reconst_loss_new:
                best_loss = reconst_loss_new
                # best_x = x_.squeeze().data.clone()
                best_x = x_.data.clone()
                # best_y = y.squeeze()
                best_y = y

            # Stop optimization if loss seems to have converged
            this_iter_loss_on_clipped_data = reconst_loss_new.item()
            # print("(x*,y*) loss stagnation", prev_loss,
            #     this_iter_loss_on_clipped_data)

            # if np.abs(prev_loss - this_iter_loss_on_clipped_data) < 1e-6:
            #     print("(x*,y*) loss stagnation", prev_loss,
            #         this_iter_loss_on_clipped_data)
            #     break

            # Keep track of this loss for next iteration comparison
            prev_loss = this_iter_loss_on_clipped_data

            if dynamic_lr:
                # Update scheduler
                scheduler.step(prev_loss)
    
    # best_x = best_x.detach().unsqueeze(0)
    # best_y = ch.ones((1,), dtype=ch.long).cuda() * best_y.detach().item()

    return (best_x, best_y), best_loss


def estimate_stability(model_1, model_2, x, n_samples=100, epsilon=0.1):
    with ch.no_grad():
        # Get predictions on current data
        preds_1 = ch.argmax(model_1(ch.clamp(x, 0, 1)), 1)
        preds_2 = ch.argmax(model_2(ch.clamp(x, 0, 1)), 1)
        scores = ch.zeros_like(preds_1).float()
        for _ in range(n_samples):
            # Get perturbed data
            rp = ch.randn_like(x)
            rp_norm = rp.view(
                rp.shape[0],
                -1).norm(dim=1).view(-1, *([1]*(len(x.shape) - 1)))
            pert = ch.clamp(x + epsilon * rp / (rp_norm + 1e-10), 0, 1)
            # Make not of predictions by both models
            scores += (ch.argmax(model_1(pert), 1) == preds_1)
            scores += (ch.argmax(model_2(pert), 1) == preds_2)
        # Normalize scores, according to number of trials
        scores /= (n_samples * 2)
    return scores


def update_gradients(model, thetas, weight_decay, x_opt, y_opt):
    model.eval()

    # Define CE Loss
    ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
    # Get L2 regularization loss for model
    # l2_reg = dnn_utils.get_model_l2_norm(model)

    # Get total loss, compute gradients
    loss = ce_loss(model(x_opt), y_opt)

    # Compute gradients
    grads = autograd.grad(
        loss, utils.get_relevant_params(model.named_parameters()))

    # Update gradients for each layer
    for i, c in enumerate(grads):
        thetas[i] -= c.clone().detach()

    # Clear gradients
    model.zero_grad()

    return thetas


def get_argmax_loss(model, thetas, num_points, method='layer_sum'):
    # Get dot-product across all layers
    layer_dot_products = []
    relevant_params = utils.get_relevant_params(model.named_parameters())
    for theta, param in zip(thetas, relevant_params):
        # Compute <w, theta>
        w_flat = param.view(-1)
        theta_flat = theta.view(-1)
        layer_dot_products.append(ch.sum(ch.mul(w_flat, theta_flat)))

    if method == 'layer_mean':
        full_loss = ch.mean(ch.stack(layer_dot_products)) / num_points
    if method == 'layer_sum':
        full_loss = ch.sum(ch.stack(layer_dot_products)) / num_points
    else:
        raise ValueError(
            "This <w,theta> loss selection method not implemented yet")

    return full_loss


def w_optimal_gradient_ascent(model, thetas, method,
                              num_points_seen_virtually,
                              lr, weight_decay=0,
                              iters=2000, verbose=False):
    # Set to train mode
    model.train()

    # Define optimizer
    optim = ch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    iterator = range(iters)
    if verbose:
        print(utils.yellow_print("[Performing gradient ascent (OGA)]"))
        iterator = tqdm(iterator)

    for _ in iterator:
        # Zero-out accumulated gradients
        optim.zero_grad()

        # Compute loss for current model:
        # When pasing thetas, normalize by number of points
        # Model has (kind of) seen so far
        wt_loss = get_argmax_loss(model=model, thetas=thetas,
                                  num_points=num_points_seen_virtually,
                                  method=method)

        # Minimize R(w) - <w, theta_t>
        # Same as maximizing  <w, theta_t> - R(w)
        r_w = (weight_decay * dnn_utils.get_model_l2_norm(model))
        loss = r_w - wt_loss

        if verbose:
            with ch.no_grad():
                iterator.set_description("Loss: %.4f, <w, theta>: %.4f, R(w): %.4f" % (
                    loss.item(), wt_loss.item(), r_w))

        # Compute gradients
        loss.backward()
        optim.step()

    # Set back to eval mode
    model.eval()

    return model


def train_clean_model(ds, args, epochs=None):
    if epochs is None:
        epochs = args.epochs

    # Construct model
    model = dnn_utils.get_seeded_wrapped_model(args, n_classes=ds.n_classes)
    model.train()

    # Fetch appropriate dataset
    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = len(ds.train)
    train_loader, _ = ds.get_loaders(batch_size)

    # Define optimizer, get data-loaders
    optim = ch.optim.Adam(model.parameters(), lr=args.pretrain_lr,
                          weight_decay=args.pretrain_weight_decay)

    # Train model
    iterator = range(epochs)
    for e in iterator:
        # Train epoch
        dnn_utils.epoch(model, train_loader, optim, e + 1,
                        None, None, verbose=True,
                        lossfn=args.loss)

    # Set to eval mode before returning
    model.eval()

    return model, optim


def log_information(logger, best_loss, x_opt, norm_diffs,
                    trn_sub_acc, trn_nsub_acc, tst_sub_acc,
                    tst_nsub_acc, num_iters, args, label=None):
    # Calculate number of points added
    num_points = num_iters * args.n_copies

    # Log optimized image
    logger.add_image("X*", x_opt[0], num_points)

    # Log weight-norm difference, if same archs
    if norm_diffs is not None:
        logger.add_scalar("Model-norm difference", norm_diffs,
                          global_step=num_points)

    # Log population accuracies on train, test data
    logger.add_scalar("Accuracy on population (train)", trn_sub_acc,
                      global_step=num_points)
    logger.add_scalar("Accuracy on non-population (train)", trn_nsub_acc,
                      global_step=num_points)
    logger.add_scalar("Accuracy on population (test)", tst_sub_acc,
                      global_step=num_points)
    logger.add_scalar("Accuracy on non-population (test)", tst_nsub_acc,
                      global_step=num_points)

    # Log best loss
    logger.add_scalar("Loss on x*,y*", best_loss.item(),
                      global_step=num_points)

    # Note y*, if given
    if label is not None:
        logger.add_scalar("Y*", label.item(),
                          global_step=num_points)
