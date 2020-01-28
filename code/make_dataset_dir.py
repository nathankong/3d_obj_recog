# Script to put data from pickle file into a directory format that is amenable
# for usage with PyTorch's DatasetFolder class.

import pickle
import numpy as np

def main():
    pkl_fname = "/mnt/fs5/nclkong/datasets/bnn_dataset/lytroPatches_30x30.pkl"
    f = open(pkl_fname)
    X, y = pickle.load(f)
    f.close()

    assert X.shape[0] == y.shape[0], "Number of labels do not align with number of samples."
    assert len(np.unique(y)) == 2, "Only two classes allowed."

    n_samples = X.shape[0]
    for i in range(n_samples):
        if (i+1) % 100 == 0:
            print "Image {}".format(i+1)

        if y[i] == 0:
            label = "crossed"
        elif y[i] == 1:
            label = "uncrossed"
        else:
            assert 0, "label {} does not exist.".format(y[i])

        sample = (X[i,:,:,:] / 255.) * 2. - 1.
        sample = np.transpose(sample, (1,2,0)) # output dim: (30,30,2)

        save_dir = "/mnt/fs5/nclkong/datasets/bnn_dataset/{}/".format(label)
        np.save(save_dir+"img_{}.npy".format(i+1), sample)

if __name__ == "__main__":
    main()

