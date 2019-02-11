import sys
import numpy as np; 
import nibabel as nib; 

infile = sys.argv[1]
outdir = sys.argv[2]


if len(sys.argv) >= 4:
    var_name = sys.argv[3]
else:
    var_name = 'vol_data'

v = np.load(infile)[var_name]; 

for i in range(255, int(np.max(v))+1):
	seg = np.copy(v)
	seg[seg!=i] = 0
	seg[seg==i] = 1

	if np.max(seg) > 0:
		nii = nib.Nifti1Image(seg, np.eye(4)); 
		nib.save(nii, outdir+str(i)+'.nii');

print("done converting %s to %s" % (infile, outdir))