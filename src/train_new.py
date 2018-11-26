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
from keras.losses import mean_squared_error

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


def train(model, pretrained_path, model_name, gpu_id, lr, n_iterations, autoencoder_iters, autoencoder_model, autoencoder_num_downsample, feature_coef, norm_percentile, seg_path, reg_param, model_save_iter, batch_size=1):
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    restrict_GPU_tf(str(gpu_id))
    restrict_GPU_keras(str(gpu_id))

    model_dir = "../models/" + model_name
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
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # prepare the model
    # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
    # in the experiments, we use image_2 as atlas

    autoencoder_path = '../models/%s/%s.h5' % (autoencoder_model, autoencoder_iters)

    ac_weights = np.zeros(80)
    good = [41, 62, 44, 45, 39, 35, 73, 78, 15, 71]
    ac_weights[good] = 1

    # for i in range(16):
    #     ac_weights[i] = 1
    if seg_path == None:
        loss_function = losses.autoencoderLoss(autoencoder_path, autoencoder_num_downsample, ac_weights, feature_coef, mean_squared_error, norm_percentile)
    else:
        loss_function = losses.segNetworkLoss(seg_path, feature_coef=feature_coef, loss_function=mean_squared_error, percentile=norm_percentile)

    model = networks.unet(vol_size, nf_enc, nf_dec)
    model.compile(optimizer=Adam(lr=lr), 
                  loss=[loss_function, losses.gradientLoss('l2')],
                  loss_weights=[1.0, reg_param])


    # if you'd like to initialize the data, you can do it here:
    if pretrained_path != None:
        model.load_weights(pretrained_path)

    # prepare data for training
    train_example_gen = datagenerators.example_gen(train_vol_names)
    zero_flow = np.zeros([batch_size, *vol_size, 3])

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    for step in range(0, n_iterations):

        # get data
        X = next(train_example_gen)[0]

        # train
        train_loss = model.train_on_batch([X, atlas_vol], [atlas_vol, zero_flow])
        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        # print the loss. 
        print_loss(step, 1, train_loss)

        # save model
        if step % model_save_iter == 0:
            model.save(os.path.join(model_dir, str(step) + '.h5'))


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
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2'], default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--pretrained_path", type=str,
                        dest="pretrained_path", default='../models/cvpr_pretrained_vm2_l2/0.h5',
                        help="path of pretrained model")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--ac_iters", type=int,
                        dest="autoencoder_iters", default=135400,
                        help="autoencoder number of iterations")
    parser.add_argument("--ac_model", type=str,
                        dest="autoencoder_model", default='autoencoder_3',
                        help="autoencoder model name")
    parser.add_argument("--ac_num_downsample", type=int,
                        dest="autoencoder_num_downsample", default=3,
                        help="autoencoder number of downsample layers")
    parser.add_argument("--feature_coef", type=float,
                        dest="feature_coef", default=1,
                        help="coefficient to weight feature loss")
    parser.add_argument("--norm_percentile", type=float,
                        dest="norm_percentile", default=None,
                        help="percentile used when normalizing")
    parser.add_argument("--seg_path", type=str,
                        dest="seg_path", default=None,
                        help="seg model path")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=0.01,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")
    parser.add_argument("--model_name", type=str,
                        dest="model_name", default='autoencoder_cost',
                        help="models folder")

    args = parser.parse_args()
    train(**vars(args))
