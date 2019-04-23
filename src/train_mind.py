"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 
import nibabel as nib

# project imports
import datagenerators
import networks
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen


def train(data_dir,
          model,
          model_name,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          batch_size,
          load_model_file,
          atlas_file,
          max_clip,
          distance,
          patch_size,
          use_ssc,
          use_gaussian_kernel,
          use_fixed_var,
          use_miccai,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    atlas_vol = nib.load(atlas_file).get_data()[np.newaxis,...,np.newaxis]
    atlas_vol = atlas_vol/np.max(atlas_vol) * max_clip
    # atlas_vol = nib.load('../data/t1_atlas.nii').get_data()[np.newaxis,...,np.newaxis]
    vol_size = atlas_vol.shape[1:-1] 
    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else: # 'vm2double': 
        nf_enc = [f*2 for f in nf_enc]
        nf_dec = [f*2 for f in [32, 32, 32, 32, 32, 16, 16]]

    model_dir = "../models/" + model_name
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    gpu = '/gpu:%d' % gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # prepare the model
        # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
        # in the experiments, we use image_2 as atlas
        if use_miccai:
            print('miccai: therefore diffeomorphic')
            model = networks.miccai2018_net(vol_size, nf_enc, nf_dec)
        else:
            model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)

        # load initial weights
        if load_model_file is not None and load_model_file != '':
            print('loading', load_model_file)
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # data generator
    # nb_gpus = len(gpu_id.split(','))
    # assert np.mod(batch_size, nb_gpus) == 0, \
    #     'batch_size should be a multiple of the nr. of gpus. ' + \
    #     'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)
    nb_gpus = 1

    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size)
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    cvpr2018_gen = datagenerators.cvpr2018_gen(train_example_gen, atlas_vol_bs, batch_size=batch_size)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    loss_function = losses.mind(distance, patch_size, use_ssc=use_ssc, use_gaussian_kernel=use_gaussian_kernel, use_fixed_var=use_fixed_var)

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single-gpu
        else:
            save_callback = ModelCheckpoint(save_file_name, verbose=1)
            mg_model = model

        # compile
        mg_model.compile(optimizer=Adam(lr=lr), 
                         loss=[loss_function, losses.Grad('l2').loss],
                         loss_weights=[1.0, reg_param])
            
        # fit
        mg_model.fit_generator(cvpr2018_gen, 
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/train/vols/',
                        help="data folder")
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2', 'vm2double'], default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--model_name", type=str,
                        dest="model_name", default='test',
                        help="models folder")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='../models/cvpr2018_vm2_l2.h5',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default='../data/t2_atlas_027_S_2219.nii',
                        help="filename of the atlas to use")
    parser.add_argument("--max_clip", type=float,
                        dest="max_clip", default=0.7,
                        help="maximum input value to calculate bins")
    parser.add_argument("--distance", type=int,
                        dest="distance", default=1,
                        help="distance for MIND")
    parser.add_argument("--patch_size", type=int,
                        dest="patch_size", default=1,
                        help="patch size for MIND")
    parser.add_argument("--ssc", dest="use_ssc", action="store_true")
    parser.set_defaults(use_ssc=False)
    parser.add_argument("--gaussian", dest="use_gaussian_kernel", action="store_true")
    parser.set_defaults(use_gaussian_kernel=False)
    parser.add_argument("--fixed_var", dest="use_fixed_var", action="store_true")
    parser.set_defaults(use_fixed_var=False)
    parser.add_argument("--miccai", dest="use_miccai", action="store_true")
    parser.set_defaults(use_miccai=False)
    args = parser.parse_args()
    train(**vars(args))
