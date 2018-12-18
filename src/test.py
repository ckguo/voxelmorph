# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
from restrict import restrict_GPU_tf, restrict_GPU_keras
import time
from argparse import ArgumentParser
import keras.backend as K

# project
import networks
import datagenerators
import losses

# Test file and anatomical labels we want to evaluate
test_brain_file = open('val_files.txt')
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
good_labels = sio.loadmat('../data/labels.mat')['labels'][0]


base_data_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/'
val_vol_names = glob.glob(base_data_dir + 'validate/vols/*.npz')
seg_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/validate/asegs/'

def test(model_name, iter_num, gpu_id, n_test, invert_images, max_clip, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  
    start_time = time.time()
    gpu = '/gpu:' + str(gpu_id)
    print(gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Anatomical labels we want to evaluate
    labels = sio.loadmat('../data/labels.mat')['labels'][0]

    atlas = np.load('../data/atlas_norm.npz')
    atlas_vol = atlas['vol']
    atlas_seg = atlas['seg']
    atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    dice_means = []
    dice_stds = []

    sz = atlas_seg.shape
    z_inp1 = tf.placeholder(tf.float32, sz)
    z_inp2 = tf.placeholder(tf.float32, sz)
    z_out = losses.kdice(z_inp1, z_inp2, labels)
    kdice_fn = K.function([z_inp1, z_inp2], [z_out])

    nn_trf_model = networks.nn_trf(vol_size)

    for step in range(0, n_test):

        # get data
        if n_test == 1:
            X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')
        else:
            vol_name, seg_name = test_brain_strings[step].split(",")
            X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        if invert_images:
            X_vol = max_clip - X_vol

        with tf.device(gpu):
            pred = net.predict([X_vol, atlas_vol])
            warp_seg = nn_trf_model.predict([X_seg, pred[1]])
            vals = kdice_fn([warp_seg[0,:,:,:,0], atlas_seg])

        # print(np.mean(vals), np.std(vals))
        mean = np.mean(vals)
        std = np.std(vals)
        dice_means.append(mean)
        dice_stds.append(std)
        print(step, mean, std)


    print('average dice:', np.mean(dice_means))
    print('std over patients:', np.std(dice_means))
    print('average std over regions:', np.mean(dice_stds))
    print('time taken:', time.time() - start_time)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        dest="model_name", default=None,
                        help="models folder")
    parser.add_argument("--iter_num", type=int,
                        dest="iter_num", default=0,
                        help="number of iterations of saved model")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--n_test", type=int, default=50,
                        dest="n_test", help="number of test subjects")
    parser.add_argument("--max_clip", type=float,
                        dest="max_clip", default=0.7,
                        help="maximum input value to calculate bins")
    parser.add_argument("--invert_images", dest="invert_images", action="store_true")
    parser.set_defaults(invert_images=False)

    args = parser.parse_args()
    test(**vars(args))
