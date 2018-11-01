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
import pickle

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

def append_to_dict(d, key, value):
    lst = d.get(key, [])
    lst.append(value)
    d[key] = lst

def test(model_name, iter_num, gpu_id, n_test, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16]):
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

    autoencoder_path = '../models/autoencoder_3/135400.h5'
    num_downsample = 3
    enc = [16, 32, 32, 32][:num_downsample]
    dec = [32]*num_downsample
    autoencoder, _ = networks.autoencoder(vol_size, enc, dec)
    autoencoder.load_weights(autoencoder_path)
    autoencoder.trainable = False

    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    dice_means = []

    results = {}

    for step in range(0, n_test):

        res = {}

        vol_name, seg_name = test_brain_strings[step].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        with tf.device(gpu):
            pred = net.predict([X_vol, atlas_vol])
            warped_image = pred[0][:, :, :, :, :]
            _, pred_ac_features = autoencoder.predict([warped_image])
            _, orig_ac_features = autoencoder.predict([X_vol])

        # Warp segments with flow
        flow = pred[1][0, :, :, :, :]
        sample = flow+grid
        sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
        warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

        vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
        # print(np.mean(vals), np.std(vals))
        mean = np.mean(vals)
        std = np.std(vals)
        
        res['dice_mean'] = mean
        res['dice_std'] = std

        for i in range(16):
            pred_feature = pred_ac_features[:,:,:,:,i]
            orig_feature = orig_ac_features[:,:,:,:,i]
            append_to_dict(res, 'l1_diff', np.mean(np.abs(pred_feature-orig_feature)))
            append_to_dict(res, 'l2_diff', np.mean(np.square(pred_feature-orig_feature)))

            append_to_dict(res, 'pred_mean', np.mean(pred_feature))
            append_to_dict(res, 'pred_std', np.std(pred_feature))
            append_to_dict(res, 'pred_99pc', np.percentile(pred_feature, 99))
            append_to_dict(res, 'pred_1pc', np.percentile(pred_feature, 1))

            append_to_dict(res, 'orig_mean', np.mean(orig_feature))
            append_to_dict(res, 'orig_std', np.std(orig_feature))
            append_to_dict(res, 'orig_99pc', np.percentile(orig_feature, 99))
            append_to_dict(res, 'orig_1pc', np.percentile(orig_feature, 1))
        
        dice_means.append(mean)

        results[vol_name] = res

        print(step, mean, std)


    print('average dice:', np.mean(dice_means))
    print('time taken:', time.time() - start_time)
    for key, value in results.items():
        print(key)
        print(value)

    with open('seg_feature_stats.txt', 'wb') as file:
        file.write(pickle.dumps(results)) # use `pickle.loads` to do the reverse

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
