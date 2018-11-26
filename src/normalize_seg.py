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

def normalize(model_name, iter_num, gpu_id, n_test, vol_size=(160,192,224)):
    start_time = time.time()
    gpu = '/gpu:' + str(gpu_id)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        seg_path = '../models/' + model_name + '/' + str(iter_num) + '.h5'
        feature_model, num_features = networks.segmenter_feature_model(seg_path)

    feature_stats = [{} for i in range(len(num_features))]
    percentiles = [0.1, 1, 5, 10, 90, 95, 99, 99.9]

    for step in range(0, n_test):
        vol_name, seg_name = test_brain_strings[step].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)
        print('X_vol shape', X_vol.shape)

        with tf.device(gpu):
            print('input', feature_model.inputs)
            print('actual', tf.transpose(X_vol[0,:,:,:,:], perm=[2,0,1,3]).shape)
            enc_array = feature_model.predict([np.transpose(X_vol[0,:,:,:,:], (2,0,1,3))], batch_size=16)

        print(step, time.time()-start_time)
        for i in range(len(num_features)):
            enc = enc_array[i]
            # batch size is 1
            mean = np.mean(enc, axis=(0, 1, 2))
            mean_keepdim = np.mean(enc, axis=(0, 1, 2), keepdims=True)
            var = np.mean(np.square(enc - mean_keepdim), axis=(0,1,2))
            means = feature_stats[i].get('mean', [])
            variances = feature_stats[i].get('var', [])
            means.append(mean)
            variances.append(var)
            feature_stats[i]['mean'] = means
            feature_stats[i]['var'] = variances

            for q in percentiles:
                pc = np.percentile(enc, q, axis=(0,1,2))
                lst = feature_stats[i].get(q, [])
                lst.append(pc)
                feature_stats[i][q] = lst
        print(step, time.time()-start_time)

    for i in range(len(num_features)):
        for key, value in feature_stats[i].items():
            if key == 'mean' or 'var':
                feature_stats[i][key] = np.mean(np.array(value), axis=0)
            else:
                feature_stats[i][key] = np.median(np.array(value), axis=0)

    print(feature_stats)
    with open('seg_feature_stats.txt', 'wb') as file:
         file.write(pickle.dumps(feature_stats)) # use `pickle.loads` to do the reverse

if __name__ == "__main__":
    normalize('seg_pretrained', 0, sys.argv[1], int(sys.argv[2]))
