import os
import sys
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from collections import defaultdict

from model import BinocularNetwork
from utils import \
    acquire_data_loaders, \
    compute_num_correct, \
    print_update, \
    save_checkpoint, \
    check_best_accuracy

def train_step(model, loader, optimizer, loss_func, epoch, device):
    losses = list()
    accuracies = list()
    total_samples = 0.
    num_steps = len(loader)

    model.train()
    for i, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        total_samples += data.shape[0]

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        predictions = model(data)
        loss = loss_func(predictions, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

        accuracy = compute_num_correct(
            predictions,
            labels
        )

        losses.append(loss.item() * data.shape[0])
        accuracies.append(accuracy)

        print(
            "[Epoch {}; Step {}/{}] Train Loss: {:.6f}; Train Accuracy: {:.6f}"\
                .format(epoch+1, i+1, num_steps, loss.item(), float(accuracy) / data.shape[0])
        )
        sys.stdout.flush()

    return losses, accuracies, total_samples

def val_step(model, loader, loss_func, epoch, device):
    losses = list()
    accuracies = list()
    total_samples = 0.
    num_steps = len(loader)

    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            total_samples += data.shape[0]

            predictions = model(data)
            loss = loss_func(predictions, labels)

            accuracy = compute_num_correct(
                predictions,
                labels
            )

            losses.append(loss.item() * data.shape[0])
            accuracies.append(accuracy)

            print(
                "[Epoch {}; Step {}/{}] Validation Loss: {:.6f}; Validation Accuracy: {:.6f}"\
                    .format(epoch+1, i+1, num_steps, loss.item(), float(accuracy) / data.shape[0])
            )

    return losses, accuracies, total_samples

def train(train_params, num_epochs, device, verbose=True):
    # Initialize the dictionary that records losses and accuracies
    results = dict()
    results["losses"] = defaultdict(list)
    results["accs"] = defaultdict(list)
    results_fname = train_params["save_dir"] + "/results.pkl"

    # Data loaders
    train_loader, test_loader, val_loader = acquire_data_loaders(
        train_params["image_dir"],
        train_params["batch_size"],
        do_transforms=False # automatically convert to tensor and [0,1] range
    )

    # Load model
    m = BinocularNetwork(
        n_filters=train_params["num_kernels"],
        k_size=train_params["kernel_size"],
        input_size=train_params["img_size"],
        n_latent=train_params["num_latent"],
        relu_latent=True # first few exps were not relu'd
    ).to(device)

    # Loss function
    loss_func = nn.CrossEntropyLoss(reduction="mean")

    # Optimizer
    optimizer = optim.SGD(
        m.parameters(),
        lr=train_params["learning_rate"],
        momentum=0.9,
        weight_decay=0.0005
    )

    # Do training
    best_acc = 0.0
    for epoch_idx in range(num_epochs):

        # Train step
        train_losses, correct, total_samples = train_step(
            m,
            train_loader,
            optimizer,
            loss_func,
            epoch_idx,
            device
        )
        train_loss = np.sum(train_losses) / total_samples
        train_acc = np.sum(correct) / total_samples
        results["losses"]["train"].append(train_loss)
        results["accs"]["train"].append(train_acc)

        # Validation step
        val_losses, correct, total_samples = val_step(
            m,
            val_loader,
            loss_func,
            epoch_idx,
            device
        )
        val_loss = np.sum(val_losses) / total_samples
        val_acc = np.sum(correct) / total_samples
        results["losses"]["val"].append(val_loss)
        results["accs"]["val"].append(val_acc)

        # Set flag if current test set accuracy is the best so far
        best_acc, is_best = check_best_accuracy(val_acc, best_acc)

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
                'results': results,
                'train_params': train_params,
                'best_acc': best_acc
            }, 
            train_params["save_dir"],
            is_best,
            save_epoch
        )

        # Save losses and accuracies every epoch so we can plot loss and accuracy
        pickle.dump(results, open(results_fname, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    default_dataset_dir = "/data5/nclkong/norb/"
    default_save_dir = "/mnt/fs5/nclkong/trained_models/norb/"
    parser.add_argument('--image-dir', type=str, default=default_dataset_dir)
    parser.add_argument('--save-dir', type=str, default=default_save_dir)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num-kernels', type=int, default=40)
    parser.add_argument('--num-latent', type=int, default=1000)
    parser.add_argument('--kernel-size', type=int, default=19)
    parser.add_argument('--image-size', type=int, default=108)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp-id', type=str, default="default")
    args = parser.parse_args()

    # Use GPU or CPU
    if torch.cuda.is_available():
        cuda_idx = int(args.cuda)
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    print "Device:", device

    # Save directory
    save_dir = args.save_dir + '/' + args.exp_id
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training parameters
    train_params = dict()
    train_params["exp_id"] = args.exp_id
    train_params["image_dir"] = args.image_dir
    train_params["save_dir"] = save_dir
    train_params["num_kernels"] = int(args.num_kernels)
    train_params["num_latent"] = int(args.num_latent)
    train_params["kernel_size"] = int(args.kernel_size)
    train_params["img_size"] = int(args.image_size)
    train_params["batch_size"] = int(args.batch_size)
    train_params["learning_rate"] = float(args.lr)

    # Do the training
    print "Training parameters:", train_params
    train(train_params, int(args.num_epochs), device)


