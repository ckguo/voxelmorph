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
from keras.losses import mean_squared_error, sparse_categorical_crossentropy
import scipy.io as sio
import nibabel as nib

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
train_seg_dir = base_data_dir + 'train/asegs/'

# load atlas from provided files. This atlas is 160x192x224.
atlas_vol = nib.load('../data/t2_atlas_027_S_2219.nii').get_data()[np.newaxis,...,np.newaxis]
seg = nib.load('../data/t2_atlas_seg_027_S_2219.nii').get_data()[np.newaxis,...,np.newaxis]


def train(model, pretrained_path, model_name, gpu_id, lr, n_iterations, use_mi, gamma, num_bins, patch_size, max_clip, reg_param, model_save_iter, local_mi, sigma_ratio, batch_size=1):
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

    train_labels = sio.loadmat('../data/labels.mat')['labels'][0]
    n_labels = train_labels.shape[0]

    normalized_atlas_vol = atlas_vol/np.max(atlas_vol) * max_clip

    atlas_seg = datagenerators.split_seg_into_channels(seg, train_labels)
    atlas_seg = datagenerators.downsample(atlas_seg)

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

    bin_centers = np.linspace(0, max_clip, num_bins*2+1)[1::2]
    loss_function = losses.mutualInformation(bin_centers, max_clip=max_clip, local_mi=local_mi, patch_size=patch_size, sigma_ratio=sigma_ratio)

    model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec, use_seg=True, n_seg=len(train_labels))
    model.compile(optimizer=Adam(lr=lr), 
                  loss=[loss_function, losses.gradientLoss('l2'), sparse_categorical_crossentropy],
                  loss_weights=[1 if use_mi else 0, reg_param, gamma])

    # if you'd like to initialize the data, you can do it here:
    if pretrained_path != None and pretrained_path != '':
        model.load_weights(pretrained_path)

    # prepare data for training
    train_example_gen = datagenerators.example_gen(train_vol_names, return_segs=True, seg_dir=train_seg_dir)
    zero_flow = np.zeros([batch_size, *vol_size, 3])

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    for step in range(0, n_iterations):

        # get data
        X = next(train_example_gen)
        X_seg = X[1]

        X_seg = datagenerators.split_seg_into_channels(X_seg, train_labels)
        X_seg = datagenerators.downsample(X_seg)

        # train
        train_loss = model.train_on_batch([X[0], normalized_atlas_vol, X_seg], [normalized_atlas_vol, zero_flow, atlas_seg])
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
    parser.add_argument('--use_mi', dest='use_mi', action='store_true')
    parser.set_defaults(use_mi=False)
    parser.add_argument("--num_bins", type=int,
                        dest="num_bins", default=48,
                        help="number of bins when calculating mutual information")
    parser.add_argument("--patch_size", type=int,
                        dest="patch_size", default=1,
                        help="patch size when doing local MI")
    parser.add_argument("--max_clip", type=float,
                        dest="max_clip", default=0.7,
                        help="maximum input value to calculate bins")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")
    parser.add_argument("--model_name", type=str,
                        dest="model_name", default='autoencoder_cost',
                        help="models folder")
    parser.add_argument("--sigma_ratio", type=float,
                        dest="sigma_ratio", default=0.5,
                        help="sigma to bin width ratio in MI")
    parser.add_argument("--local_mi", dest="local_mi", action="store_true")
    parser.set_defaults(local_mi=False)
    parser.add_argument("--gamma", type=float,
                        dest="gamma", default=1,
                        help="diceloss weight")

    args = parser.parse_args()
    train(**vars(args))
