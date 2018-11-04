
# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import networks
from keras.losses import mean_squared_error
import pickle

vol_size = (160, 192, 224)  
# batch_sizexheightxwidthxdepthxchan

# def normalize(features):
#     mean = tf.reduce_mean(features, axis=(1, 2, 3), keepdims=True)
#     var = tf.reduce_mean(tf.square(features - mean), axis=(1,2,3), keepdims=True)
#     return (features - mean)/tf.sqrt(var)

# def normalize_percentile(features, percentile):
    # assert percentile > 50
    # top = tf.contrib.distributions.percentile(features, percentile, axis=(1,2,3), keep_dims=True)
    # bottom = tf.contrib.distributions.percentile(features, 100-percentile, axis=(1,2,3), keep_dims=True)
    # return (features - bottom)/(top-bottom)

def normalize_percentile(features, percentile, feature_stats):
    pcs = feature_stats[percentile]
    features = features / pcs[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    return tf.clip_by_value(features, 0, 1)

def autoencoderLoss(autoencoder_path, num_downsample, ac_weights, ac_coef=1, loss_function=None, use_normalize=True, percentile=None):
    enc = [16, 32, 32, 32][:num_downsample]
    dec = [32]*num_downsample
    autoencoder, _ = networks.autoencoder(vol_size, enc, dec)
    autoencoder.load_weights(autoencoder_path)
    autoencoder.trainable = False

    with open('feature_stats.txt', 'rb') as file:
        feature_stats = pickle.loads(file.read()) # use `pickle.loads` to do the reverse

    print('use_normalize:', use_normalize)
    print('percentile:', percentile)

    def loss(y_true, y_pred):
        if percentile != None:
            tgt_features = normalize_percentile(autoencoder(y_true)[1], percentile, feature_stats)
            src_features = normalize_percentile(autoencoder(y_pred)[1], percentile, feature_stats)
        elif use_normalize:
            tgt_features = normalize(autoencoder(y_true)[1])
            src_features = normalize(autoencoder(y_pred)[1])
        else:
            tgt_features = autoencoder(y_true)[1]
            src_features = autoencoder(y_pred)[1]
        ac_loss = tf.reduce_mean(tf.square(tgt_features - src_features), axis=(0, 1, 2, 3))
        ac_loss_weighted = tf.reduce_sum(tf.multiply(ac_loss, ac_weights))
        if loss_function:
            return ac_coef * ac_loss_weighted + loss_function(y_true, y_pred)
        else:
            return ac_coef * ac_loss
    return loss


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0

    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        return -1.0*tf.reduce_mean(cc)

    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss




## Losses for the MICCAI2018 Paper
def kl_loss(alpha):
    def loss(_, y_pred):
        """
        KL loss
        y_pred is assumed to be 6 channels: first 3 for mean, next 3 for logsigma
        """
        mean = y_pred[..., 0:3]
        log_sigma = y_pred[..., 3:]

        # compute the degree matrix.
        # TODO: should only compute this once!
        # z = K.ones((1, ) + vol_size + (3, ))
        sz = log_sigma.get_shape().as_list()[1:]
        z = K.ones([1] + sz)

        filt = np.zeros((3, 3, 3, 3, 3))
        for i in range(3):
            filt[1, 1, [0, 2], i, i] = 1
            filt[[0, 2], 1, 1, i, i] = 1
            filt[1, [0, 2], 1, i, i] = 1
        filt_tf = tf.convert_to_tensor(filt, dtype=tf.float32)
        D = tf.nn.conv3d(z, filt_tf, [1, 1, 1, 1, 1], "SAME")
        D = K.expand_dims(D, 0)

        sigma_terms = (alpha * D * tf.exp(log_sigma) - log_sigma)

        # note needs 0.5 twice, one here, one below
        prec_terms = 0.5 * alpha * kl_prec_term_manual(_, mean)
        kl = 0.5 * tf.reduce_mean(sigma_terms, [1, 2, 3]) + 0.5 * prec_terms
        return kl

    return loss

def kl_prec_term_manual(y_true, y_pred):
    """
    a more manual implementation of the precision matrix term
            P = D - A
            mu * P * mu
    where D is the degree matrix and A is the adjacency matrix
            mu * P * mu = sum_i mu_i sum_j (mu_i - mu_j)
    where j are neighbors of i
    """
    dy = y_pred[:,1:,:,:,:] * (y_pred[:,1:,:,:,:] - y_pred[:,:-1,:,:,:])
    dx = y_pred[:,:,1:,:,:] * (y_pred[:,:,1:,:,:] - y_pred[:,:,:-1,:,:])
    dz = y_pred[:,:,:,1:,:] * (y_pred[:,:,:,1:,:] - y_pred[:,:,:,:-1,:])
    dy2 = y_pred[:,:-1,:,:,:] * (y_pred[:,:-1,:,:,:] - y_pred[:,1:,:,:,:])
    dx2 = y_pred[:,:,:-1,:,:] * (y_pred[:,:,:-1,:,:] - y_pred[:,:,1:,:,:])
    dz2 = y_pred[:,:,:,:-1,:] * (y_pred[:,:,:,:-1,:] - y_pred[:,:,:,1:,:])

    d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz) + \
        tf.reduce_mean(dy2) + tf.reduce_mean(dx2) + tf.reduce_mean(dz2)
    return d


def kl_l2loss(image_sigma):
    def loss(y_true, y_pred):
        return 1. / (image_sigma**2) * K.mean(K.square(y_true - y_pred))
    return loss