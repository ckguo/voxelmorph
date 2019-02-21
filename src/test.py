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
import nibabel as nib

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators

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

    restrict_GPU_tf(str(gpu_id))
    restrict_GPU_keras(str(gpu_id))

    # Anatomical labels we want to evaluate
    labels = sio.loadmat('../data/labels.mat')['labels'][0]

    # atlas = np.load('../data/atlas_norm.npz')
    # atlas_vol = atlas['vol']
    # atlas_seg = atlas['seg']
    # atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

    atlas_vol = nib.load('../t2_atlas_warped.nii').get_data()[np.newaxis,...,np.newaxis]
    atlas_seg = nib.load('../t2_atlas_seg_warped.nii').get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    dice_means = []
    dice_stds = []

    for step in range(0, n_test):

        # get data
        if n_test == 1:
            X_vol = nib.load('../t1_atlas.nii').get_data()[np.newaxis,...,np.newaxis]
            X_seg = nib.load('../t1_atlas_seg.nii').get_data()[np.newaxis,...,np.newaxis]
        else:
            vol_name, seg_name = test_brain_strings[step].split(",")
            X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        if invert_images:
            X_vol = max_clip - X_vol

        with tf.device(gpu):
            pred = net.predict([X_vol, atlas_vol])

        # Warp segments with flow
        flow = pred[1][0, :, :, :, :]
        sample = flow+grid
        sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
        warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

        vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
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
