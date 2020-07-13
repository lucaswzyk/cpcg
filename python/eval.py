from dipy.io.image import load_nifti, save_nifti
from dipy.io.utils import get_reference_info
from dipy.io.streamline import load_trk
from dipy.align.reslice import reslice

from os import walk, remove
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_path = ''
img_path = ''
training_ids = []
training_paths = []
non_training_ids = []
non_training_paths = []
tracts = []


def set_paths():
    global data_path, img_path
    data_path = '/data/academics/02_info/CG/cpcg/data/'
    for root, dirs, files in walk(data_path + 'training'):
        for d in dirs:
            training_ids.append(d)
            training_paths.append(data_path + 'training/' + d + '/')
        break
    for root, dirs, files in walk(data_path + 'non_training'):
        for d in dirs:
            non_training_ids.append(d)
            non_training_paths.append(data_path + 'non_training/' + d + '/')
        break
    img_path = '/T1w/Diffusion/data.nii.gz'
    for root, dirs, files in walk(training_paths[0] + 'T1w/Diffusion/tractseg_output/bundle_segmentations'):
        for f in files:
            tracts.append(f[:-7])


fig = plt.figure()
angle1 = 15
angle2 = 250
size = 3
alpha = .015


def draw_mask(data, pos):
    ax = fig.add_subplot(pos, projection='3d')
    y, x, z = data.nonzero()
    ax.scatter(x / 10, y, z, zdir='z', c='red', s=size, alpha=alpha)
    ax.view_init(angle1, angle2)


def example():
    cg_left_101107 = "/data/academics/02_info/CG/cpcg/data/101107/T1w/Diffusion/tractseg_output/bundle_segmentations" \
                   "/CG_left.nii.gz"
    data_101107, affine_101107, img_101107 = load_nifti(cg_left_101107, return_img=True)
    data_101107 = data_101107 != 0

    cg_left_599469 = "/data/academics/02_info/CG/cpcg/data/599469/diffusion/tractseg_output/bundle_segmentations" \
                   "/CG_left.nii.gz"
    data_599469, affine_599469, img_599469 = load_nifti(cg_left_599469, return_img=True)
    data_599469 = data_599469 != 0

    cg_left_tractseg = "/home/cc/Downloads/CG_left_bin.nii.gz"
    data_tractseg, affine_tractseg, img_tractseg = load_nifti(cg_left_tractseg, return_img=True)
    data_tractseg = data_tractseg != 0

    data1 = data_599469
    data2 = data_tractseg

    data_union = data1 | data2
    data_intersect = data1 & data2
    data_diff = data_union & np.logical_not(data_intersect)

    dice = 2 * np.sum(data_intersect) / (np.sum(data1) + np.sum(data2))
    print("dice = ", dice)

    draw_mask(data1, '131')
    draw_mask(data2, '132')
    draw_mask(data_diff, '133')
    plt.show()


def downsample(subject_folder):
    print('Down-sampling in ', subject_folder)
    # load 4D volume
    data_folder = subject_folder + 'T1w/Diffusion/'
    low_res_folder = subject_folder + 'T1w/Diffusion_low_res/'

    # make a folder to save new data into
    try:
        Path(low_res_folder).mkdir(parents=True, exist_ok=True)
    except OSError:
        print('Could not create output dir. Aborting...')
        return

    # load bvals and make binary mask (True for b = 1000)
    with open(data_folder + 'bvals') as f:
        bvals = [int(x) for x in next(f).split()]
    mask_b1000 = [i == 1000 for i in bvals]

    bvals = np.asarray(bvals)[mask_b1000]
    bvals_low_res_file = low_res_folder + 'bvals'
    if Path(bvals_low_res_file).exists():
        remove(bvals_low_res_file)
    with open(bvals_low_res_file, 'x') as f:
        f.write(' '.join(map(str, bvals)))

    # load bvecs
    bvecs_low_res_file = low_res_folder + 'bvecs'
    if Path(bvecs_low_res_file).exists():
        remove(bvecs_low_res_file)
    new_file = open(bvecs_low_res_file, 'x')
    with open(data_folder + 'bvecs') as f:
        for line in f:
            # read line and mask it
            new_coords = np.asarray([float(x) for x in line.split()])[mask_b1000]
            new_file.write(' '.join(map(str, new_coords)) + '\n')
    new_file.close()

    img = nib.load(data_folder + 'data.nii.gz')
    affine = img.affine
    zooms = img.header.get_zooms()[:3]
    data = np.asarray(img.dataobj)
    data = np.einsum('xyzb->bxyz', data)
    data = data[mask_b1000]
    data = np.einsum('bxyz->xyzb', data)

    new_zooms = (2.5, 2.5, 2.5)
    new_data, new_affine = reslice(data, affine, zooms, new_zooms)
    print('Down-sampled to shape ', new_data.shape)
    save_nifti(low_res_folder + 'data.nii.gz', new_data, new_affine)

    mask_img = nib.load(data_folder + 'nodif_brain_mask.nii.gz')
    mask = np.asarray(mask_img.dataobj)
    mask_zooms = mask_img.header.get_zooms()[:3]
    mask_affine = mask_img.affine
    new_mask, new_maks_affine = reslice(mask, mask_affine, mask_zooms, new_zooms)
    save_nifti(low_res_folder + 'nodif_brain_mask.nii.gz', new_mask, new_maks_affine)


def downsample_segmentations(subject):
    print('Downsampling for ' + subject)

    folder = data_path + 'tractseg/' + subject + '/tracts/'
    for t in tracts:
        img = nib.load(folder + t + '.nii.gz')
        data = np.asarray(img.dataobj)
        zooms = img.header.get_zooms()[:3]
        affine = img.affine
        new_data, new_affine = reslice(data, affine, zooms, new_zooms=[2.5, 2.5, 2.5])
        save_nifti(folder + t + '_low_res.nii.gz', new_data, new_affine)
        print('\rSaved ' + t + '         ', end='')
    print('')


def alltrks2bin(tractseg_folder):
    reference_anatomy = tractseg_folder + 'reference_anatomy.nii.gz'
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info(reference_anatomy)
    low_res = (2.5, 2.5, 2.5)

    subjects = []
    for root, dirs, files in walk(tractseg_folder):
        if 'average' in dirs: dirs.remove('average')
        subjects.extend(dirs)
        break

    # assuming all subject folders contain all tracts (given for TractSeg samples)
    tracts = []
    for root, dirs, files in walk(tractseg_folder + subjects[0] + '/tracts'):
        for f in files:
            if f[-4:] == ".trk":
                tracts.append(f[:-4])

    # make a folder to save average masks into
    try:
        Path(tractseg_folder + 'average').mkdir(parents=True, exist_ok=True)
    except OSError:
        print('Could not create output dir. Aborting...')
        return

    i = 0
    errs = []
    for t in tracts:
        print('Current tract: ', t, ' (', (i * 100) / len(tracts), '%)')
        high_res_path = tractseg_folder + 'average/' + t + '.nii.gz'
        low_res_path = tractseg_folder + 'average/' + t + '_low_res.nii.gz'

        if not (Path(high_res_path).exists() & Path(low_res_path).exists()):
            j = 0
            t_ave = np.zeros(dimensions, dtype=np.float)
            for s in subjects:
                try:
                    print('Current subject: ', s, ' (', (j * 100) / len(subjects), '%)')
                    trk_data = load_trk(tractseg_folder + s + '/tracts/' + t + '.trk', reference_anatomy)
                    trk_data.to_vox()
                    streams = np.vstack(trk_data.streamlines).astype(np.int32)
                    mask = np.zeros(dimensions, dtype=np.float)
                    mask[streams[:, 0], streams[:, 1], streams[:, 2]] = 1
                    save_nifti(tractseg_folder + s + '/tracts/' + t + '.nii.gz', mask, affine)

                    # make low res version
                    # mask_low_res, affine_low_res = reslice(mask, affine, voxel_sizes, low_res)
                    # save_nifti(tractseg_folder + s + '/tracts/' + t + '_low_res.nii.gz', mask_low_res, affine_low_res)

                    t_ave += mask
                    j += 1
                except:
                    errs.append('Error with tract ' + t + ' in subject ' + s)

            try:
                t_ave /= len(subjects)
                save_nifti(high_res_path, t_ave, affine)
                t_ave_low_res, affine_low_res = reslice(t_ave, affine, voxel_sizes, low_res)
                save_nifti(low_res_path, t_ave_low_res, affine_low_res)
            except:
                errs.append('Error with average of tract ' + t)
        i += 1

    for e in errs:
        print(e)


def d_st(mask1_file, mask2_file):
    mask1, _ = load_nifti(mask1_file)
    mask2, _ = load_nifti(mask2_file)
    intersection = np.multiply(mask1, mask2)

    if mask1.shape == mask2.shape:
        return 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))
    else:
        print('Could not dice ' + mask1_file + ' and ' + mask2_file)
        return 0


def d_s(subject, base_dir, high_res=True):
    print('')
    low_res_str = ''
    if not high_res:
        low_res_str = '_low_res'

    res = 0
    for t in tracts:
        mask1_file = data_path + base_dir + subject + '/T1w/Diffusion' + low_res_str + \
                     '/tractseg_output/bundle_segmentations/' + t + '.nii.gz'
        mask2_file = data_path + 'tractseg/' + subject + '/tracts/' + t + low_res_str + '.nii.gz'
        res_t = d_st(mask1_file, mask2_file)
        print('\rResult for ' + t + ' was ' + str(res_t) + '                   ', end='')
        res += res_t

    return res / len(tracts)


def calc_dices():
    txt = data_path + 'training/dices.txt'
    if Path(txt).exists():
        remove(txt)
    with open(txt, 'w+') as f:
        for subject in training_ids:
            line = subject + ' '
            line += str(d_s(subject, 'training/')) + ' '
            line += str(d_s(subject, 'training/', high_res=False))
            f.write(line + '\n')


set_paths()
# for p in training_paths:
#     downsample(p)
# for p in non_training_paths:
#     downsample(p)
# for t_id in training_ids:
#     downsample_segmentations(t_id)
#alltrks2bin(data_path + 'tractseg/')
calc_dices()
