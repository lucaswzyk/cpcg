from os.path import join as pjoin
import numpy as np
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from os import getcwd
from os.path import join, dirname
from dipy.segment.mask import median_otsu
import vtk
import fury
import mrtrix

data_fnames = get_fnames('scil_b0')
# cwd = getcwd()
# dname = join(dirname(cwd), 'sherbrooke_3shell')
# fdwi = join(dname, 'HARDI193.nii.gz')

data, affine = load_nifti(data_fnames[0])
print(data[0])
data = np.squeeze(data)
print(data[0])

b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
bla = 1
