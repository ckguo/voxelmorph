
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

def normalize_percentile(features, percentile, feature_stats, layer=0, twod=False):
    pcs = feature_stats[layer][percentile]
    if not twod:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    else:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, :]
    return tf.clip_by_value(features, 0, 1)

def autoencoderLoss(autoencoder_path, num_downsample, ac_weights, ac_coef=1, loss_function=None, percentile=None):
    enc = [16, 32, 32, 32][:num_downsample]
    dec = [32]*num_downsample
    autoencoder, _ = networks.autoencoder(vol_size, enc, dec)
    autoencoder.load_weights(autoencoder_path)
    autoencoder.trainable = False

    with open('feature_stats.txt', 'rb') as file:
        feature_stats = pickle.loads(file.read()) # use `pickle.loads` to do the reverse

    print('percentile:', percentile)

    def loss(y_true, y_pred):
        tgt_ac_features = autoencoder(y_true)[1:]
        src_ac_features = autoencoder(y_pred)[1:]
        idx = 0

        ac_loss_weighted = 0

        for i in range(num_downsample):
            if percentile != None:
                tgt_features = normalize_percentile(tgt_ac_features[i], percentile, feature_stats, i)
                src_features = normalize_percentile(src_ac_features[i], percentile, feature_stats, i)
            else:
                print('no normalization!')
                tgt_features = tgt_ac_features[i]
                src_features = src_ac_features[i]
            ac_loss = tf.reduce_mean(tf.square(tgt_features - src_features), axis=(0, 1, 2, 3))
            ac_loss_weighted += tf.reduce_sum(tf.multiply(ac_loss, ac_weights[idx:idx+enc[i]]))

            idx += enc[i]

        loss_function = None
        if loss_function:
            return ac_coef * ac_loss_weighted + loss_function(y_true, y_pred)
        else:
            return ac_coef * ac_loss_weighted
    return loss

def segNetworkLoss(seg_path, feature_coef=1, loss_function=None, feature_weights=None, percentile=None):
    # unet_full = networks.unet_full(vol_size, nf_enc, nf_dec)
    # unet_full.load_weights(seg_path)
    # unet_full.trainable = False

    feature_model, num_features = networks.segmenter_feature_model(seg_path)

    # for i in range(len(model.layers)):
    #     l = model.layers[i]
    #     print('layer', i, l.output)

    if percentile != None:
        with open('seg_feature_stats.txt', 'rb') as file:
            feature_stats = pickle.loads(file.read()) # use `pickle.loads` to do the reverse
    # feature_weights = [1]*sum(num_features)
    print('percentile:', percentile)

    def loss(y_true, y_pred):
        tgt_seg_features = feature_model(tf.transpose(y_true[0,:,:,:,:], perm=[2,0,1,3]))
        src_seg_features = feature_model(tf.transpose(y_pred[0,:,:,:,:], perm=[2,0,1,3]))
        idx = 0

        feature_loss_weighted = 0

        for i in range(1):
        # for i in range(len(tgt_seg_features)):
            if percentile != None:
                tgt_features = normalize_percentile(tgt_seg_features[i], percentile, feature_stats, i, twod=True)
                src_features = normalize_percentile(src_seg_features[i], percentile, feature_stats, i, twod=True)
            else:
                print('no normalization!')
                tgt_features = tgt_seg_features[i]
                src_features = src_seg_features[i]
            feature_loss = tf.reduce_mean(tf.square(tgt_features - src_features), axis=(0, 1, 2))

            if feature_weights != None:
                feature_loss_weighted += tf.reduce_sum(tf.multiply(feature_loss, feature_weights[idx:idx+num_features[i]]))
            else:
                feature_loss_weighted += tf.reduce_sum(feature_loss)

            idx += num_features[i]
            # print('idx', idx)

        loss_function = None
        if loss_function:
            return feature_coef * feature_loss_weighted + loss_function(y_true, y_pred)
        else:
            return feature_coef * feature_loss_weighted
    return loss


def mutualInformation(bin_centers,
                      sigma=None,    # sigma for soft MI. If not provided, it will be half of a bin length
                      weights=None,  # optional weights, size [1, nb_labels]
                      vox_weights=None,
                      max_clip=1,
                      crop_background=False):
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
        
    """

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    weights = None if weights is None else K.variable(weights)
    vox_weights = None if vox_weights is None else K.variable(vox_weights)
    sigma = sigma
    
    if sigma is None:
        sigma = np.mean(np.diff(bin_centers)/2)
    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 10
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)
        
        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

        print('mi', mi)
        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss
'''
    def mean_mi(self, y_true, y_pred):
        """ weighted mean mi across all patches and labels """

        # compute dice, which will now be [batch_size, nb_labels]
        mi_metric = self.mi(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            mi_metric *= self.weights
        if self.vox_weights is not None:
            mi_metric *= self.vox_weights

        # return one minus mean dice as loss
        mean_mi_metric = K.mean(mi_metric)
        tf.verify_tensor_all_finite(mean_mi_metric, 'metric not finite')
        return mean_mi_metric


    def loss(self, y_true, y_pred):
        """ loss is negative MI """

        # compute dice, which will now be [batch_size, nb_labels]
        mi_metric = self.mi(y_true, y_pred)

        # loss
        mi_loss = - mi_metric

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            mi_loss *= self.weights

        # return one minus mean dice as loss
        mean_mi_loss = K.mean(mi_loss)
        tf.verify_tensor_all_finite(mean_mi_loss, 'Loss not finite')
        return mean_mi_loss
'''


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


def kdice(vol1, vol2, labels):
    dicem = [None] * len(labels)
    for idx, lab in enumerate(labels):
        vol1l = tf.cast(tf.equal(vol1, lab), 'float32')
        vol2l = tf.cast(tf.equal(vol2, lab), 'float32')
        top = 2 * K.sum(vol1l * vol2l)
        bottom = K.sum(vol1l) + K.sum(vol2l)
        bottom = K.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom
    return tf.stack(dicem)

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