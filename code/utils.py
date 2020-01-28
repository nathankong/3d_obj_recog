import os
import sys
import shutil

import numpy as np

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, random_split

from functools import partial

def image_loader(do_transforms, image_name):
    image = np.load(image_name)
    image = np.copy(image).astype('uint8')

    if do_transforms:
        loader = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = loader(image)
    else:
        # Image dimensions input: (chan, height, width)
        # Tranpose to dimensions: (height, weight, chan)
        image = np.transpose(image, (1,2,0))
        # ToTensor() normalizes to [0,1] range and tranpose to (chan, height, width)
        loader = transforms.Compose([transforms.ToTensor()])
        image = loader(image)

    image = Variable(image, requires_grad=False)
    return image

def acquire_datasets(images_dir, do_transforms):
    # Load dataset. This function is specific for the NORB data set

    train_dataset = DatasetFolder(
        images_dir + "/train_npy/",
        extensions="npy",
        loader=partial(image_loader, do_transforms)
    )

    test_dataset = DatasetFolder(
        images_dir + "/test_npy/",
        extensions="npy",
        loader=partial(image_loader, do_transforms)
    )

    # Split train dataset to train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size]
    )
    return train_set, val_set, test_dataset

def acquire_data_loaders(
    images_dir,
    batch_size,
    do_transforms=False
):
    # Acquire train/test/validation datasets
    train_set, test_set, val_set = acquire_datasets(images_dir, do_transforms)

    # Initialize data loaders
    train_data_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )
    val_data_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )

    return train_data_loader, test_data_loader, val_data_loader

def record_weights(model, simple_unit_list, complex_unit_list):
    # Saves weights/filters into the two lists. This utility function
    # is specific to the BinocularNetwork defined in code/model.py
    simple_unit_list.append(model.simple_unit[0].weight.data.cpu().numpy())
    complex_unit_list.append(model.complex_unit[0].weight.data.cpu().numpy())

def compute_num_correct(predictions, labels):
    # Predictions and labels should be on the same torch.device and
    # are of type torch.Tensor. total_samples should be a float.

    preds = predictions.argmax(dim=1, keepdim=True)
    correct = preds.eq(labels.view_as(preds)).sum().item()
    return correct

def print_update(dataset_type, loss, accuracy, epoch_idx, total_epochs):
    assert dataset_type.lower() in ["train", "test", "validation"]

    print(
        "[Epoch {}/{}] {} Loss: {:.6f}; {} Accuracy: {:.6f}"\
            .format(epoch_idx+1, total_epochs, dataset_type, loss, dataset_type, accuracy)
    )
    sys.stdout.flush()

def save_checkpoint(state, save_dir, is_best, save_epoch):
    fname = save_dir+"/checkpoint.pt"
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, save_dir+"/model_best.pt")

    if save_epoch is not None:
        shutil.copyfile(fname, save_dir+"/checkpoint_epoch_{}.pt".format(save_epoch+1))

def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        cpt = torch.load(checkpoint_path)
        return cpt
    else:
        print("No checkpoint at '{}'".format(checkpoint_path))
        assert 0

def check_best_accuracy(current_acc, best_acc):
    if current_acc > best_acc:
        return current_acc, True
    return best_acc, False

if __name__ == "__main__":
    im_dir = "/mnt/fs5/nclkong/datasets/bnn_dataset/"
    batch_size = 20
    device = torch.device("cpu")

    tr, te, va = acquire_datasets(im_dir, device, False)
    d,t = te[100]
    print(type(d), t)
    assert 0

    tr, te, va = acquire_data_loaders(im_dir, batch_size, device)

    for i, (data, label) in enumerate(tr):
        print "Train batch {}/{}: {} {}".format(i+1, len(tr), data.size(), label.size())

    for i, (data, label) in enumerate(te):
        print "Test batch {}/{}: {} {}".format(i+1, len(te), data.size(), label.size())
        print label
        assert 0

    for i, (data, label) in enumerate(va):
        print "Val batch {}/{}: {} {}".format(i+1, len(va), data.size(), label.size())


