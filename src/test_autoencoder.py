# py imports
import os
import sys
import glob
import random
from argparse import ArgumentParser

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
from neuron.plot import slices

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators

base_data_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/'
val_vol_names = glob.glob(base_data_dir + 'validate/vols/*.npz')
random.shuffle(val_vol_names)  # shuffle volume list

# load atlas from provided files. This atlas is 160x192x224.
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol'][np.newaxis,...,np.newaxis]


def test(model_name, iter_num, gpu_id, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

    gpu = '/gpu:' + str(gpu_id)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        full_model, train_model = networks.autoencoder(vol_size, nf_enc, nf_dec)
        full_model.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    val_example_gen = datagenerators.example_gen(val_vol_names)

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    total_loss = 0
    for step in range(1):

        # get data
        X = next(val_example_gen)[0]

        # get output
        output, enc = full_model.predict([X])

        loss = tf.reduce_mean(tf.square(output - X))

        # print the loss. 
        print(step, 0, loss)
        total_loss += loss

        slices(output[0])

    print(total_loss)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        dest="model_name", default='autoencoder',
                        help="models folder")
    parser.add_argument("--iter_num", type=int,
                        dest="iter_num", default=100,
                        help="which iteration of model to use")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")


    args = parser.parse_args()
    test(**vars(args))

