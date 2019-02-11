import sys
import numpy as np; 
import nibabel as nib; 

infile = sys.argv[1]
outfile = sys.argv[2]

if len(sys.argv) >= 4:
    var_name = sys.argv[3]
else:
    var_name = 'vol_data'

v = np.load(infile)[var_name]; 

nii = nib.Nifti1Image(v, np.eye(4)); 

nib.save(nii, outfile);

print("done converting %s to %s" % (infile, outfile))