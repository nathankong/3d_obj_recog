"""
This is an example config file. It must contain the function 
called "load_config()". Modify or copy this config file in 
order to conduct new experiments.
"""

import sys
sys.path.insert(0, "../code/")

import torch.optim as optim

from model import BinocularNetwork
from dataloader_utils import acquire_data_loaders

def load_config():

    config = dict()
    config["image_dir"] = "/data5/nclkong/norb/"
    config["exp_id"] = "exp11_nlatent_50"
    config["save_dir"] = "/mnt/fs5/nclkong/trained_models/norb/"
    config["dataset"] = "norb"
    config["cuda"] = 1
    config["device_ids"] = None # TODO
    config["num_workers"] = 8
    #config["resume"] = ""
    config["resume"] = "/mnt/fs5/nclkong/trained_models/norb/exp11_nlatent_50/checkpoint.pt"
    config["batch_size"] = 512
    config["num_epochs"] = 30
    config["learning_rate"] = 0.01
    config["lr_stepsize"] = 5
    config["lr_gamma"] = 0.5
    config["scheduler"] = "step"
    config["weight_decay"] = 0.0001
    config["momentum"] = 0.9
    config["num_kernels"] = 16
    config["kernel_size"] = 19
    config["img_size"] = 108
    config["num_latent"] = 50

    tr, _, va = acquire_data_loaders(config["image_dir"], config["batch_size"], do_transforms=False)
    config["dataloaders"] = {"train": tr, "val": va}

    config["model"] = BinocularNetwork(
        n_filters=config["num_kernels"],
        k_size=config["kernel_size"],
        input_size=config["img_size"],
        n_latent=config["num_latent"],
        relu_latent=True # first few exps were not relu'd
    )

    optimizer = optim.SGD(
        config["model"].parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    config["optimizer"] = optimizer

    if config["scheduler"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif config["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_stepsize"],
            gamma=config["lr_gamma"]
        )
    elif config["scheduler"] == "constant": # gamma = 1 => no update to LR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    else:
        assert 0, "Scheduler {} is not implemented.".format(config["scheduler"])
    config["lr_scheduler"] = scheduler

    return config

if __name__ == "__main__":
    cfg = load_config()
    print cfg


