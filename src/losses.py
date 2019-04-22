"""
losses for VoxelMorph
"""


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
                      sigma_ratio=0.5,    # sigma for soft MI. If not provided, it will be half of a bin length
                      max_clip=1,
                      crop_background=False, # crop_background should never be true if local_mi is True
                      local_mi=False,
                      patch_size=1):
    if local_mi:
        return localMutualInformation(bin_centers, sigma_ratio, max_clip, patch_size)
    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)


def globalMutualInformation(bin_centers,
                      sigma_ratio=0.5,
                      max_clip=1,
                      crop_background=False):
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
        
    """

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
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

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss

def localMutualInformation(bin_centers,
                      sigma_ratio=0.5,
                      max_clip=1,
                      patch_size=1):
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
        
    """

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, num_bins]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute padding sizes
        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- preterm * K.square(tf.pad(y_true, padding, 'CONSTANT')  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT')  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, num_bins])

        I_b_patch = tf.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = tf.reduce_mean(I_a_patch, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.mean(K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss

def mind(d, patch_size, use_ssc=False, use_gaussian_kernel=False, use_fixed_var=True):
    # see http://www.mpheinrich.de/pub/MEDIA_mycopy.pdf
    epsilon = 0.000001
    if use_gaussian_kernel:
        dist = tf.distributions.Normal(0., 1.)

        vals = dist.prob(tf.range(start = -(patch_size-1)/2, limit = (patch_size-1)/2 + 1, dtype = tf.float32))
        kernel = tf.einsum('i,j,k->ijk', vals, vals, vals)
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = kernel[:,:,:,tf.newaxis, tf.newaxis]
    else:
        kernel = tf.ones([patch_size, patch_size, patch_size, 1, 1])/(patch_size**3)
        
    def ssd_shift(image, direction):
        # expects a 3d image
        x,y,z = vol_size
        new_shift = np.clip(direction, 0, None)
        old_shift = -np.clip(direction, None, 0)

        # translate images
        new_image = image[new_shift[0]:x-old_shift[0], new_shift[1]:y-old_shift[1], new_shift[2]:z-old_shift[2]]
        old_image = image[old_shift[0]:x-new_shift[0], old_shift[1]:y-new_shift[1], old_shift[2]:z-new_shift[2]]
        # get squared difference
        diff = tf.square(new_image - old_image)

        # pad the diff
        padding = np.transpose([old_shift, new_shift])
        diff = tf.pad(diff, padding)

        # apply convolution
        conv = tf.nn.conv3d(diff[tf.newaxis,:,:,:,tf.newaxis], kernel, [1]*5, 'SAME')
        return conv

    def mind_loss(y_true, y_pred):
        ndims = 3
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        loss_tensor = 0

        if use_fixed_var:
            y_true_var = 0.004
            y_pred_var = 0.004
        else:
            y_true_var = 0
            y_pred_var = 0
            for i in range(ndims):
                direction = [0]*ndims
                direction[i] = d

                y_true_var += ssd_shift(y_true, direction)
                y_pred_var += ssd_shift(y_pred, direction)

                direction = [0]*ndims
                direction[i] = -d
                y_true_var += ssd_shift(y_true, direction)
                y_pred_var += ssd_shift(y_pred, direction)

            y_true_var = y_true_var/(ndims*2) + epsilon
            y_pred_var = y_pred_var/(ndims*2) + epsilon

        print(y_true_var)
        for i in range(ndims):
            direction = [0]*ndims
            direction[i] = d

            loss_tensor += tf.reduce_mean(tf.abs(tf.exp(-ssd_shift(y_true, direction)/y_true_var) - tf.exp(-ssd_shift(y_pred, direction)/y_pred_var)))

            direction = [0]*ndims
            direction[i] = -d
            loss_tensor += tf.reduce_mean(tf.abs(tf.exp(-ssd_shift(y_true, direction)/y_true_var) - tf.exp(-ssd_shift(y_pred, direction)/y_pred_var)))

        return loss_tensor/(ndims*2)

    def ssc_loss(y_true, y_pred):
        ndims = 3
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        loss_tensor = 0
        directions = []
        for i in range(ndims):
            direction = [0]*3
            direction[i] = d
            directions.append(direction)

            direction = [0]*3
            direction[i] = -d
            directions.append(direction)

        for i in range(len(directions)):
            for j in range(i, len(directions)):
                d1 = directions[i]
                d2 = directions[j]

                loss_tensor += tf.reduce_mean(tf.abs(ssd_shift(y_true, d1) - ssd_shift(y_pred, d2)))

        return loss_tensor/(len(directions)*(len(directions)-1)/2)

    if use_ssc:
        return ssc_loss
    else:
        return mind_loss


def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.range(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
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


class NCC():
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)


class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.get_shape()) - 2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well


    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))
