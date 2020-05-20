import dipy
from dipy.data import fetch_sherbrooke_3shell
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table
from os import getcwd
from os.path import join, dirname
import matplotlib.pyplot as plt

# Uncomment the following to download sample data
# fetch_sherbrooke_3shell()
cwd = getcwd()
dname = join(dirname(cwd), 'sherbrooke_3shell')
fdwi = join(dname, 'HARDI193.nii.gz')
fbval = join(dname, 'HARDI193.bval')
fbvec = join(dname, 'HARDI193.bvec')

data, affina, img = load_nifti(fdwi, return_img=True)
print(data.shape)
axial_middle = 50
plt.figure('Showing the datasets')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(data[:, :, axial_middle, 10].T, cmap='gray', origin='lower')

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)
print(gtab.bvecs)

plt.show()
