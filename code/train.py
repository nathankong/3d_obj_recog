import sys
import os
import pickle

import torch
import torch.nn as nn

import numpy as np

from collections import defaultdict

from utils import \
    AverageMeter, \
    save_checkpoint, \
    load_checkpoint, \
    check_best_accuracy, \
    compute_accuracy, \
    seed_torch

# TODO: Set seed as an input argument and update the save paths to include seed
# TODO: Write code to also test the training stuff on regular images
# TODO: Write code to validate on blur images

def train_step(model, train_loader, optimizer, loss_func, device, epoch, results):
    losses = AverageMeter("Loss", ':.4e')
    top1 = AverageMeter("Acc@1", ':6.2f')
    num_steps = len(train_loader)

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        predictions = model(data)
        loss = loss_func(predictions, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

        # Metrics
        acc1 = compute_accuracy(predictions, labels, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))

        print(
            "[Epoch {}; Step {}/{}] Train Loss: {:.6f}; Train Accuracy: {:.6f}"\
                .format(epoch+1, i+1, num_steps, loss, acc1)
        )
        sys.stdout.flush()

        # Save per iteration train losses/accuracies
        results["train"]["iter_losses"].append(loss.item())
        results["train"]["iter_top1_accs"].append(acc1.item())

    results["train"]["losses"].append(losses.avg)
    results["train"]["top1_accs"].append(top1.avg)

    # Save losses and accuracies every epoch so we can plot loss and accuracy
    pickle.dump(results, open(results["fname"], "wb"))

    return losses.avg, top1.avg

def test_step(model, test_loader, loss_func, device, epoch, results):
    losses = AverageMeter("Loss", ':.4e')
    top1 = AverageMeter("Acc@1", ':6.2f')
    num_steps = len(test_loader)

    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            predictions = model(data)
            loss = loss_func(predictions, labels)

            # Metrics
            acc1 = compute_accuracy(predictions, labels, topk=(1,))[0]

            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))

            print(
                "[Epoch {}; Step {}/{}] Val Loss: {:.6f}; Val Accuracy: {:.6f}"\
                    .format(epoch+1, i+1, num_steps, loss, acc1)
            )
            sys.stdout.flush()

    results["val"]["losses"].append(losses.avg)
    results["val"]["top1_accs"].append(top1.avg)

    # Save losses and accuracies every epoch so we can plot loss and accuracy
    pickle.dump(results, open(results["fname"], "wb"))

    return losses.avg, top1.avg

def train(m, train_params, num_epochs, device, save_dir, device_ids):
    # Load model
    if device_ids is not None:
        m = torch.nn.DataParallel(m, device_ids=device_ids)
    m.to(device)

    # Load optimizer
    optimizer = train_params["optimizer"]

    # Load LR scheduler
    scheduler = train_params["lr_scheduler"]

    # Load from checkpoint if needed
    if train_params["resume"] != '':
        m, optimizer, train_params, scheduler, start_epoch_idx, best_acc, results, save_dir = \
                load_checkpoint(train_params["resume"], m, optimizer) # Load ckpt
        print "Updated training parameters:", train_params
    else:
        start_epoch_idx = 0
        best_acc = 0.0
        results = dict() # Initialize the dictionary that records losses and accuracies
        results["train"] = defaultdict(list)
        results["val"] = defaultdict(list)
        results["fname"] = save_dir + "/results.pkl"

    # Load data loaders
    train_loader, val_loader = train_params["dataloaders"]["train"], train_params["dataloaders"]["val"]

    # Load loss function and optimizer
    loss_func = nn.CrossEntropyLoss(reduction="mean")

    # Loop through epochs
    for epoch_idx in range(start_epoch_idx, num_epochs):
        print "Epoch {}; LR ".format(epoch_idx+1)
        for param_group in optimizer.param_groups:
            print "  ", param_group['lr']

        # Do a train step
        _, _ = train_step(
            m,
            train_loader,
            optimizer,
            loss_func,
            device,
            epoch_idx,
            results
        )

        # Do a validation step
        test_loss, test_top1 = test_step(
            m,
            val_loader,
            loss_func,
            device,
            epoch_idx,
            results
        )

        # TODO: As above, this can be done in a cleaner way.
        # Step learning rate scheduler
        if train_params["scheduler"] == "plateau":
            scheduler.step(test_loss)
        elif train_params["scheduler"] == "step" or train_params["scheduler"] == "constant":
            scheduler.step()
        else:
            assert 0, "Condition should not be reached."

        # Set flag if current test set accuracy is the best so far
        best_acc, is_best = check_best_accuracy(test_top1, best_acc)

        # Every 10 epochs, save into new checkpoint file. Overwrite existing otherwise.
        if ((epoch_idx+1) % 10 == 0) or (epoch_idx == 0):
            save_epoch = epoch_idx
        else:
            save_epoch = None

        save_checkpoint(
            {
                'epoch': epoch_idx+1,
                'state_dict': m.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'results': results,
                'train_params': train_params,
                'best_acc': best_acc
            }, 
            save_dir,
            is_best,
            save_epoch
        )


if __name__ == "__main__":
    import imp, argparse, shutil, copy
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_bnn.py")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    #### Load config file and set random seed
    cfg_fname = args.config.split('/')[-1]
    cfg = imp.load_source("configs", args.config)
    cfg = cfg.load_config()
    seed_torch(seed=int(args.seed))

    #### Use GPU or CPU
    if torch.cuda.is_available():
        device_ids = cfg["device_ids"]
        if device_ids is not None:
            cuda_idx = np.min(device_ids)
        else:
            cuda_idx = int(cfg["cuda"])
        device = torch.device("cuda:{}".format(cuda_idx))
        print "Device IDs in data parallel:", device_ids
    else:
        device = torch.device("cpu")
        assert 0 # Don't use CPU!
    print "Device:", device

    #### Save model and stats directory. Copy config file to save directory.
    save_dir = cfg["save_dir"] + '/' + cfg["exp_id"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg["save_dir"] = save_dir
    shutil.copyfile(args.config, save_dir+"/{}".format(cfg_fname))

    #### Load model
    model = cfg["model"]

    #### Start training
    print "Training parameters:", cfg 
    train(
        model,
        cfg,
        int(cfg["num_epochs"]),
        device,
        save_dir,
        device_ids
    )


