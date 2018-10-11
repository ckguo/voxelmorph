"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.losses import mean_absolute_error

# project imports
import datagenerators
import networks
import losses
from restrict import restrict_GPU_tf, restrict_GPU_keras


## some data prep
# Volume size used in our experiments. Please change to suit your data.
vol_size = (160, 192, 224)  

# prepare the data
# for the CVPR paper, we have data arranged in train/validate/test folders
# inside each folder is a /vols/ and a /asegs/ folder with the volumes
# and segmentations
base_data_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/'
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
random.shuffle(train_vol_names)  # shuffle volume list

# load atlas from provided files. This atlas is 160x192x224.
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol'][np.newaxis,...,np.newaxis]


def train(num_downsample, model_dir, gpu_id, lr, n_iterations, reg_param, model_save_iter, batch_size=1):
    """
    model training function
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param model_save_iter: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    """
    
    restrict_GPU_tf(str(gpu_id))
    restrict_GPU_keras(str(gpu_id))
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018

    assert num_downsample <= 4, 'num_downsample has to be less than 4'
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32]
    if num_downsample < 4:
        nf_enc = nf_enc[:num_downsample]
        nf_dec = nf_dec[:num_downsample]
    # prepare the model
    # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
    # in the experiments, we use image_2 as atlas
    full_model, train_model = networks.autoencoder(vol_size, nf_enc, nf_dec)
    train_model.compile(optimizer=Adam(lr=lr), 
                  loss=[mean_absolute_error])

    # if you'd like to initialize the data, you can do it here:
    # model.load_weights(os.path.join(model_dir, '120000.h5'))

    # prepare data for training
    train_example_gen = datagenerators.example_gen(train_vol_names)

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    for step in range(0, n_iterations):

        # get data
        X = next(train_example_gen)[0]

        # train
        train_loss = train_model.train_on_batch([X], [X])
        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        # print the loss. 
        print_loss(step, 1, train_loss)

        # save model
        if step % model_save_iter == 0:
            full_model.save(os.path.join(model_dir, str(step) + '.h5'))


def print_loss(step, training, train_loss):
    """
    Prints training progress to std. out
    :param step: iteration number
    :param training: a 0/1 indicating training/testing
    :param train_loss: model loss at current iteration
    """
    s = str(step) + "," + str(training)

    if isinstance(train_loss, list) or isinstance(train_loss, np.ndarray):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_downsample", type=int,
                        dest="num_downsample", default=4,
                        help="number of times to downsample image")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/autoencoder/',
                        help="models folder")

    args = parser.parse_args()

    train(**vars(args))
