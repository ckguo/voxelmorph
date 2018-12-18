import glob
import numpy as np

base_data_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adnisel/'
val_vol_names = sorted(glob.glob(base_data_dir + 'validate/vols/*.npz'))
val_seg_names = sorted(glob.glob(base_data_dir + 'validate/asegs/*.npz'))

random_idxes = np.random.choice(3157, 3157, replace=False)

f = open('val_files.txt', 'w')
for i in random_idxes:
	f.write(val_vol_names[i] + ',' + val_seg_names[i] + '\n')
f.close()
