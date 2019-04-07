import numpy as np
from pynd.segutils import seg_overlay, seg_overlap
import matplotlib.pyplot as plt
from neuron.plot import slices

def visualize_seg_contour(vol, seg, region_numbers=[4], slice_idx=[80, 90, 110], title=''):
    if title != '':
        title += ', '
    normalized_vol = vol/np.max(vol)
    seg_one_region = np.zeros(seg.shape)
    for num in region_numbers:
        seg_one_region[seg==num] = num
    img = seg_overlap(normalized_vol, seg_one_region.astype(np.int))
    plt.figure(figsize=(20,10))
    slices([img[slice_idx[0], :, :, :], img[:, slice_idx[1], :, :], img[:, :, slice_idx[2], :]], titles=[title+'x=%d'%slice_idx[0], title+'y=%d'%slice_idx[1], title+'z=%d'%slice_idx[2]])

def visualize_seg(vol, seg, region_numbers=[4], slice_idx=[80, 90, 110], title=''):
    if title != '':
        title += ', '
    normalized_vol = vol/np.max(vol)
    seg_one_region = np.zeros(seg.shape)
    for num in region_numbers:
        seg_one_region[seg==num] = num
    img = seg_overlay(normalized_vol, seg_one_region.astype(np.int))
    plt.figure(figsize=(20,10))
    slices([img[slice_idx[0], :, :, :], img[:, slice_idx[1], :, :], img[:, :, slice_idx[2], :]], titles=[title+'x=%d'%slice_idx[0], title+'y=%d'%slice_idx[1], title+'z=%d'%slice_idx[2]])
