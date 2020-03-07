import numpy as np

import torch
import torch.nn as nn
torch.manual_seed(0)

from model import BinocularNetwork
from dataloader_utils import acquire_data_loaders

# Use GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(8))
else:
    device = torch.device("cpu")

def main(exp_id="exp01"):
    #images_dir = "/mnt/fs5/nclkong/datasets/norb/"
    images_dir = "/data5/nclkong/norb/"

    best_model_path = "/mnt/fs5/nclkong/trained_models/norb/{}/model_best.pt".format(exp_id)
    params = torch.load(best_model_path)

    print(params.keys())
    print("Training parameters:", params["train_params"])

    # Get model and load trained parameters and set in eval mode
    # Hack for now just so the legacy trained models can be loaded.
    if "num_latent" in params["train_params"].keys():
        m = BinocularNetwork(
            n_filters=params["train_params"]["num_kernels"],
            k_size=params["train_params"]["kernel_size"],
            input_size=params["train_params"]["img_size"],
            n_latent = params["train_params"]["num_latent"],
            relu_latent = True
        ).to(device)
    else:
        m = BinocularNetwork(
            n_filters=params["train_params"]["num_kernels"],
            k_size=params["train_params"]["kernel_size"],
            input_size=params["train_params"]["img_size"]
        ).to(device)
    m.load_state_dict(params["state_dict"])
    m.eval()

    # Save weights
    weights = m.simple_unit[0].weight.data.cpu().numpy()
    np.save("tmp/{}_simple_unit_weights.npy".format(exp_id), weights)

    # Acquire test loader: the images are actually different instances 
    # of the same object class.
    _, test_loader, _ = acquire_data_loaders(images_dir, 256)

    total_correct = 0.0
    total = 0.0
    total_loss = 0.0
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    for i, (data, labels) in enumerate(test_loader):

        if (i+1) % 100 == 0:
            print "Batch {}/{}".format(i+1, len(test_loader))

        data, labels = data.to(device), labels.to(device)
        outputs = m(data)
        loss = loss_func(outputs, labels).item()
        total_loss += (loss * labels.shape[0])

        predictions = outputs.argmax(dim=1, keepdim=False)
        correct = predictions.eq(labels.view_as(predictions)).sum().item()
        total_correct += correct
        total += labels.shape[0]

    print "Test Accuracy:", total_correct / total
    print "Test Loss:", total_loss / total


if __name__ == "__main__":
    #main(exp_id="exp03")
    #main(exp_id="exp04")
    #main(exp_id="exp05")
    #main(exp_id="exp06")
    #main(exp_id="exp07")
    #main(exp_id="exp08")
    #main(exp_id="exp09")
    #main(exp_id="exp10")
    #main(exp_id="test")
    #main(exp_id="exp11_nlatent_50")
    main(exp_id="exp12_nlatent_100")


