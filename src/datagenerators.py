import os
import numpy as np
import tensorflow as tf

def load_example_by_name(vol_name, seg_name):

    X = np.load(vol_name)['vol_data']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    X_seg = np.load(seg_name)['vol_data']
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)

    return tuple(return_vals)


def example_gen(vol_names, return_segs=False, seg_dir=None):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names))
        X = np.load(vol_names[idx])['vol_data']
        X = np.reshape(X, (1,) + X.shape + (1,))

        return_vals = [X]

        if(return_segs):
            name = os.path.basename(vol_names[idx])
            X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals.append(X_seg)

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'

        yield tuple(return_vals)


def split_seg_into_channels(seg, labels, dn=1):
    seg_new = np.zeros((seg.shape[0:4] + (len(labels),)))

    for i in range(len(labels)):
        seg_new[:, :, :, :, i] = (seg == labels[i])[:, :, :, :, 0]

    return seg_new

def downsample(X):
    return X[:, 0:-1:2, 0:-1:2, 0:-1:2, :]

def normalize_percentile(features, percentile, feature_stats, layer=0, twod=False):
    pcs = feature_stats[layer][percentile]
    if not twod:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    else:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, :]
    return np.clip(features, 0, 1)
