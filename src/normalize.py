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
test_brain_file = open('train_files.txt')
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

def normalize(model_name, iter_num, gpu_id, n_test, vol_size=(160,192,224), nf_enc=[16,32,32], nf_dec=[32,32,32]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  
    start_time = time.time()
    gpu = '/gpu:' + str(gpu_id)

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
        full_model, train_model = networks.autoencoder(vol_size, nf_enc, nf_dec)
        full_model.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    feature_stats = [{} for i in range(len(nf_enc))]
    percentiles = [0.1, 1, 5, 10, 90, 95, 99, 99.9]

    for step in range(0, n_test):
        vol_name, seg_name = test_brain_strings[step].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        with tf.device(gpu):
            enc_array = full_model.predict([X_vol])
            output = enc_array[0]

        for i in range(len(nf_enc)):
            enc = enc_array[i+1]
            # batch size is 1
            mean = np.mean(enc, axis=(0, 1, 2, 3))
            mean_keepdim = np.mean(enc, axis=(0, 1, 2, 3), keepdims=True)
            var = np.mean(np.square(enc - mean_keepdim), axis=(0,1,2,3))
            means = feature_stats[i].get('mean', [])
            variances = feature_stats[i].get('var', [])
            means.append(mean)
            variances.append(var)
            feature_stats[i]['mean'] = means
            feature_stats[i]['var'] = variances

            for q in percentiles:
                pc = np.percentile(enc, q, axis=(0,1,2,3))
                lst = feature_stats[i].get(q, [])
                lst.append(pc)
                feature_stats[i][q] = lst
        print(step)

    for i in range(len(nf_enc)):
        for key, value in feature_stats[i].items():
            if key == 'mean' or 'var':
                feature_stats[i][key] = np.mean(np.array(value), axis=0)
            else:
                feature_stats[i][key] = np.median(np.array(value), axis=0)

    print(feature_stats)
    with open('feature_stats.txt', 'wb') as file:
         file.write(pickle.dumps(feature_stats)) # use `pickle.loads` to do the reverse

if __name__ == "__main__":
    normalize('autoencoder_3', 135400, sys.argv[1], int(sys.argv[2]))
