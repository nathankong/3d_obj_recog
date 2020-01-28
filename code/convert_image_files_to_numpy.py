import os
import sys
import gzip
import struct

import numpy as np
from scipy.io import loadmat

# GLOBALS THAT DEFINE STEREO DATA SET SPECIFICS SUCH AS THE
# NUMBER OF IMAGES AND THE IMAGE SIZE. THEY ALSO SPECIFY THE
# STARTING LOCATION OF THE BYTE TO BE READ DEPENDING ON THE
# TYPE OF BITSTREAM WE ARE DEALING WITH, WHICH IS EITHER IMAGES
# OR LABELS AT THE MOMENT. IN THE FUTURE WE MIGHT WANT TO USE
# THE META DATA OF THE IMAGES AND THIS FILE SHOULD BE UPDATED
# ACCORDINGLY.
IMAGE_HEIGHT = 108
IMAGE_WIDTH = 108
IMAGE_SIZE_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH # 108 pixels x 108 pixels
IMAGE_START_BYTE = 24
LABEL_START_BYTE = 20
NUM_IMAGES = 29160
CATEGORY_MAPPING = {0: "animal", 1: "human", 2: "plane", 3: "truck", 4: "car", 5: "blank"}

def convert_file_to_bytestream(fname, data_type):
    assert data_type in ["images", "labels"]
    with gzip.open(fname, "rb") as fp:
        b = fp.read()
    if data_type == "images":
        b = list(map(ord, b))
    else:
        assert data_type == "labels"
    return b

def process_images_bytestream(b):
    images = list()
    start_idx = IMAGE_START_BYTE
    for j in range(NUM_IMAGES):
        if (j+1) % 1000 == 0:
            print "Processing image {}".format(j+1)
            sys.stdout.flush()

        im = list()
        for i in range(start_idx, start_idx + (2 * IMAGE_SIZE_BYTES)):
            im.append(b[i])
        a = np.reshape(np.array(im), [2, IMAGE_WIDTH, IMAGE_HEIGHT])
        images.append(a)
        start_idx = start_idx + (2 * IMAGE_SIZE_BYTES)
    return np.array(images)

def process_labels_bytestream(b):
    labels = list()
    begin = LABEL_START_BYTE
    for i in range(NUM_IMAGES):
        start = begin + (4*i)
        end = start + 4
        label = struct.unpack_from("<i", b[start:end])[0]
        labels.append(label)
    return np.array(labels)

def process_bytestream(filename, data_type):
    # data_type is either "images" or "labels" for the time being
    # data_type is important since it determins whether we are
    # reading in data one byte at a time or four bytes at a time
    assert data_type in ["images", "labels"]

    if data_type == "images":
        data = process_images_bytestream(filename)
    elif data_type == "labels":
        data = process_labels_bytestream(filename)
    else:
        assert 0, "This condition should never be reached."

    return data

def process_data_file(filename, data_type):
    b = convert_file_to_bytestream(filename, data_type)
    data = process_bytestream(b, data_type)
    return data

def save_images_to_directory(images, labels, image_name_base, save_dir_base):
    assert images.shape[0] == labels.shape[0] == NUM_IMAGES
    for i in range(NUM_IMAGES):
        text_label = CATEGORY_MAPPING[labels[i]]
        save_dir = save_dir_base + "/{}/".format(text_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image_fname = image_name_base + "_{}_{}.npy".format(text_label, i+1)
        np.save(save_dir + image_fname, images[i])

def process_dataset_files(dataset_base_dir, dataset_type):
    if dataset_type == "training":
        num_sets = 10
        file_template = dataset_base_dir + "/norb-5x46789x9x18x6x2x108x108-training-{}-{}.mat.gz"
        image_name_base_template = "image_train_{}"
        save_dir = "/mnt/fs5/nclkong/datasets/norb/train_npy/"
    elif dataset_type == "testing":
        num_sets = 2
        file_template = dataset_base_dir + "/norb-5x01235x9x18x6x2x108x108-testing-{}-{}.mat.gz"
        image_name_base_template = "image_test_{}"
        save_dir = "/mnt/fs5/nclkong/datasets/norb/test_npy/"
    else:
        assert 0, "Condition should not be reached."

    for i in range(num_sets):
        if (i+1) > 9:
            dataset_num = str(i+1)
        else:
            dataset_num = '0' + str(i+1)

        image_name_base = image_name_base_template.format(dataset_num)
        im_file = file_template.format(dataset_num, "dat")
        label_file = file_template.format(dataset_num, "cat")

        print "Processing {} and".format(im_file)
        print "{}".format(label_file)
        print "..."

        # Process byte streams
        images = process_data_file(im_file, "images")
        labels = process_data_file(label_file, "labels")

        # Save images as npy
        save_images_to_directory(images, labels, image_name_base, save_dir)

def main(dataset_type):
    # save_dir: should be the base directory that branches off into
    # subdirectories for each image category.
    # dataset_type: should either be "training" or "testing"
    assert dataset_type in ["training", "testing"]
    process_dataset_files(dataset_type)

if __name__ == "__main__":
    testing_base_dir = "/mnt/fs5/nclkong/datasets/norb/test/"
    training_base_dir = "/mnt/fs5/nclkong/datasets/norb/train/"

    #main(testing_base_dir, "testing")
    main(training_base_dir, "training")


