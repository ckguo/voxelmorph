# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
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

def test(model_name, epoch, gpu_id, n_test, invert_images, max_clip, indexing, use_miccai, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16]):
    start_time = time.time()
    good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

    # setup
    gpu = '/gpu:' + str(gpu_id)
    #     print(gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    restrict_GPU_tf(str(gpu_id))
    restrict_GPU_keras(str(gpu_id))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))
    
    atlas_vol = nib.load('../data/t2_atlas_027_S_2219.nii').get_data()[np.newaxis,...,np.newaxis]
    atlas_seg = nib.load('../data/t2_atlas_seg_027_S_2219.nii').get_data()
    
    atlas_vol = atlas_vol/np.max(atlas_vol) * max_clip

    sz = atlas_seg.shape
    z_inp1 = tf.placeholder(tf.float32, sz)
    z_inp2 = tf.placeholder(tf.float32, sz)
    z_out = losses.kdice(z_inp1, z_inp2, good_labels)
    kdice_fn = K.function([z_inp1, z_inp2], [z_out])

    trf_model = networks.trf_core(vol_size, nb_feats=len(good_labels)+1, indexing=indexing)

    # load weights of model
    with tf.device(gpu):
        if use_miccai:
            net = networks.miccai2018_net(vol_size, nf_enc, nf_dec)
            net.load_weights('../models/' + model_name +
                             '/' + str(epoch) + '.h5')
        else:
            net = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)
            net.load_weights('../models/' + model_name +
                             '/' + str(epoch) + '.h5')

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
            all_labels = np.unique(X_seg)
            for l in all_labels:
                if l not in good_labels:
                    X_seg[X_seg==l] = 0
            for i in range(len(good_labels)):
                X_seg[X_seg==good_labels[i]] = i+1
            seg_onehot = tf.keras.utils.to_categorical(X_seg[0,:,:,:,0], num_classes=len(good_labels)+1)
            warp_seg_onehot = trf_model.predict([seg_onehot[tf.newaxis,:,:,:,:], pred[1]])
            warp_seg = np.argmax(warp_seg_onehot[0,:,:,:], axis=3)
            
            warp_seg_correct = np.zeros(warp_seg.shape)
            for i in range(len(good_labels)):
                warp_seg_correct[warp_seg==i+1] = good_labels[i]
            
            dice = kdice_fn([warp_seg_correct, atlas_seg])

            mean = np.mean(dice)
            std = np.std(dice)
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
    parser.add_argument("--epoch", type=int,
                        dest="epoch", default=0,
                        help="epoch number of saved model")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--n_test", type=int, default=50,
                        dest="n_test", help="number of test subjects")
    parser.add_argument("--max_clip", type=float,
                        dest="max_clip", default=0.7,
                        help="maximum input value to calculate bins")
    parser.add_argument("--indexing", type=str,
                        dest="indexing", default='ij',
                        help="indexing to use (ij or xy)")
    parser.add_argument("--invert_images", dest="invert_images", action="store_true")
    parser.set_defaults(invert_images=False)
    parser.add_argument("--miccai", dest="use_miccai", action="store_true")
    parser.set_defaults(use_miccai=False)

    args = parser.parse_args()
    test(**vars(args))
