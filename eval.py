from os.path import basename

from dipy.data.fetcher import get_two_hcp842_bundles
from fury import window, actor

from dipy.align.streamlinear import whole_brain_slr, BundleMinDistanceAsymmetricMetric
from dipy.direction import peaks_from_model, PeaksAndMetrics
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.io.peaks import save_peaks, load_peaks
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info, create_tractogram_header
from dipy.io.streamline import load_trk, save_trk
from dipy.align.reslice import reslice
from dipy.data import default_sphere, fetch_target_tractogram_hcp, fetch_bundle_atlas_hcp842, get_bundle_atlas_hcp842, \
    get_target_tractogram_hcp
from dipy.segment.mask import median_otsu
from dipy.tracking.local_tracking import LocalTracking

from os import walk, remove
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.reconst.csdeconv import auto_response, ConstrainedSphericalDeconvModel
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.tracking.utils import random_seeds_from_mask
from mpl_toolkits.mplot3d import Axes3D
from nibabel.streamlines import Tractogram, ArraySequence, LazyTractogram
from dipy.segment.bundles import RecoBundles

data_path = ''
tractseg_dir = ''
img_path = ''
reference_anatomy = ''
training_ids = []
training_paths = []
non_training_ids = []
non_training_paths = []
tracts = []


def set_paths():
    global data_path, img_path, tractseg_dir, reference_anatomy
    # data_path = '/data/academics/02_info/CG/cpcg/data/'
    data_path = 'D:/academics/02_info/CG/cpcg/data/'
    tractseg_dir = data_path + 'tractseg/'
    reference_anatomy = tractseg_dir + 'reference_anatomy.nii.gz'
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


def alltrks2bin():
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info(reference_anatomy)
    low_res = (2.5, 2.5, 2.5)

    subjects = ['983773']
    # for root, dirs, files in walk(tractseg_dir):
    #     if 'average' in dirs:
    #         dirs.remove('average')
    #     subjects.extend(dirs)
    #     break
    #
    # # assuming all subject folders contain all tracts (given for TractSeg samples)
    # tracts = []
    # for root, dirs, files in walk(tractseg_dir + subjects[0] + '/tracts'):
    #     for f in files:
    #         if f[-4:] == ".trk":
    #             tracts.append(f[:-4])
    #
    # # make a folder to save average masks into
    # try:
    #     Path(tractseg_dir + 'average').mkdir(parents=True, exist_ok=True)
    # except OSError:
    #     print('Could not create output dir. Aborting...')
    #     return

    i = 0
    errs = []
    for t in tracts:
        print('Current tract: ', t, ' (', (i * 100) / len(tracts), '%)')
        # high_res_path = tractseg_dir + 'average/' + t + '.nii.gz'
        # low_res_path = tractseg_dir + 'average/' + t + '_low_res.nii.gz'

        if True: # not (Path(high_res_path).exists() & Path(low_res_path).exists()):
            j = 0
            t_ave = np.zeros(dimensions, dtype=np.float)
            for s in subjects:
                print('Current subject: ', s, ' (', (j * 100) / len(subjects), '%)')
                # trk_data = load_trk(tractseg_dir + s + '/tracts/' + t + '.trk', reference_anatomy)
                p = data_path + 'training/' + s + '/T1w/Diffusion/qsdr/recobundles_native/' + t
                trk_data = load_trk(p + '.trk', 'same', bbox_valid_check=False)
                trk_data.to_vox()
                streams = np.vstack(trk_data.streamlines).astype(np.int32)
                mask = np.zeros(dimensions, dtype=np.float)
                affine = trk_data.affine
                mask[streams[:, 0], streams[:, 1], streams[:, 2]] = 1
                save_nifti(p + '.nii.gz', mask, affine)

                # make low res version
                # mask_low_res, affine_low_res = reslice(mask, affine, voxel_sizes, low_res)
                # save_nifti(data_path + 'training/' + s + '/T1w/Diffusion/dti/recobundles_output/' + t + '_low_res.nii.gz', mask_low_res, affine_low_res)

                t_ave += mask
                j += 1
                # try:
                # except:
                #     errs.append('Error with tract ' + t + ' in subject ' + s)

            # try:
            #     t_ave /= len(subjects)
            #     save_nifti(high_res_path, t_ave, affine)
            #     t_ave_low_res, affine_low_res = reslice(t_ave, affine, voxel_sizes, low_res)
            #     save_nifti(low_res_path, t_ave_low_res, affine_low_res)
            # except:
            #     errs.append('Error with average of tract ' + t)
        i += 1

    for e in errs:
        print(e)


def to_bin_mask(tractogram):
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info(reference_anatomy)
    low_res = (2.5, 2.5, 2.5)

    mask = np.zeros(dimensions, dtype=np.float)
    tractogram.to_vox()
    try:
        streams = np.vstack(tractogram.streamlines).astype(np.int32)
        mask[streams[:, 0], streams[:, 1], streams[:, 2]] = 1
    except:
        print('Trivial mask!')

    return mask, affine


def d_st(mask1_file, mask2_file):
    mask1, _ = load_nifti(mask1_file)
    mask2, _ = load_nifti(mask2_file)
    intersection = np.multiply(mask1, mask2)

    if mask1.shape == mask2.shape:
        return 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))
    else:
        print('Could not dice ' + mask1_file + ' and ' + mask2_file)
        return 0


def d_s(subject, input_dir, out_file):
    f = open(out_file, 'w+')
    res = 0
    for t in tracts:
        mask1_file = input_dir + t + '.nii.gz'
        mask2_file = tractseg_dir + subject + '/tracts/' + t + '.nii.gz'
        res_t = d_st(mask1_file, mask2_file)
        f.write(t + ' ' + str(res_t) + '\n')
        print('\rResult for ' + t + ' was ' + str(res_t) + '                   ', end='')
        res += res_t

    f.close()

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


def mu(mask_file, tract_file, ks, shuffle=False):
    # load mask and ground truth
    mbar_t, _ = load_nifti(tract_file)
    mask, _ = load_nifti(mask_file)
    if shuffle:
        np.random.shuffle(mask)

    mu_val = np.subtract(mask, mbar_t)
    num_elem = np.sum(np.bitwise_or(mask != 0, mbar_t != 0))
    result = []
    for k in ks:
        part_res = np.power(np.abs(mu_val), k)
        part_res = np.sum(part_res) / num_elem
        result.append(np.power(part_res, 1 / k))

    return np.asarray(result)


def calc_all_mus():
    emin = 0
    emax = 10
    ks = np.power(1.7, range(emin, emax + 1))

    mus_file = data_path + 'all_mus.txt'
    if Path(mus_file).exists():
        f = open(mus_file)
        training_mus_hr = [float(val) for val in next(f).split()[1:]]
        training_mus_lr = [float(val) for val in next(f).split()[1:]]
        non_training_mus_hr = [float(val) for val in next(f).split()[1:]]
        non_training_mus_lr = [float(val) for val in next(f).split()[1:]]
        rb_dti_native = [float(val) for val in next(f).split()[1:]]
        rb_dti_foreign = [float(val) for val in next(f).split()[1:]]
        rb_qsdr_native = [float(val) for val in next(f).split()[1:]]
        rb_qsdr_foreign = [float(val) for val in next(f).split()[1:]]
        shuffle_mus = [float(val) for val in next(f).split()[1:]]
    else:
        ground_truth_path = data_path + 'tractseg/average/'
        training_mus_hr = np.zeros(len(ks))
        training_mus_lr = np.zeros(len(ks))
        non_training_mus_hr = np.zeros(len(ks))
        non_training_mus_lr = np.zeros(len(ks))
        rb_dti_native = np.zeros(len(ks))
        rb_dti_foreign = np.zeros(len(ks))
        rb_qsdr_native = np.zeros(len(ks))
        rb_qsdr_foreign = np.zeros(len(ks))
        shuffle_mus = np.zeros(len(ks))

        sel_tracts = tracts

        f = open(mus_file, 'w+')

        for t in sel_tracts:
            t_ave_hr = ground_truth_path + t + '.nii.gz'
            t_ave_lr = ground_truth_path + t + '_low_res.nii.gz'
            for p in training_paths:
                print('\rTract ' + t + ', subject ' + p[-7:-1] + '         ', end='')
                full_path_hr = p + 'T1w/Diffusion/tractseg_output/bundle_segmentations/' + t + '.nii.gz'
                training_mus_hr += mu(full_path_hr, t_ave_hr, ks)
                full_path_lr = p + 'T1w/Diffusion_low_res/tractseg_output/bundle_segmentations/' + t + '.nii.gz'
                training_mus_lr += mu(full_path_lr, t_ave_lr, ks)
            for p in non_training_paths:
                print('\rTract ' + t + ', subject ' + p[-7:-1] + '         ', end='')
                full_path_hr = p + 'T1w/Diffusion/tractseg_output/bundle_segmentations/' + t + '.nii.gz'
                non_training_mus_hr += mu(full_path_hr, t_ave_hr, ks)
                full_path_lr = p + 'T1w/Diffusion_low_res/tractseg_output/bundle_segmentations/' + t + '.nii.gz'
                non_training_mus_lr += mu(full_path_lr, t_ave_lr, ks)
                shuffle_mus += mu(p + 'T1w/Diffusion/tractseg_output/bundle_segmentations/' + t + '.nii.gz',
                                  t_ave_hr, ks, shuffle=True)
            print('\rRecobundles...           ', end='')
            rb_dti_native += mu(training_paths[0] + 'T1w/Diffusion/dti/recobundles_native/' + t + '.nii.gz', t_ave_hr, ks)
            rb_dti_foreign += mu(training_paths[0] + 'T1w/Diffusion/dti/recobundles_foreign/' + t + '.nii.gz', t_ave_hr, ks)
            rb_qsdr_native += mu(training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_native/' + t + '.nii.gz', t_ave_hr, ks)
            rb_qsdr_foreign += mu(training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_foreign/' + t + '.nii.gz', t_ave_hr, ks)
            print('\rTract ' + t + ', shuffling...         ', end='')

        training_mus_hr /= len(sel_tracts) * len(training_paths)
        f.write('training_hr ' + ' '.join([str(val) for val in training_mus_hr]) + '\n')
        training_mus_lr /= len(sel_tracts) * len(training_paths)
        f.write('training_lr ' + ' '.join([str(val) for val in training_mus_lr]) + '\n')
        non_training_mus_hr /= len(sel_tracts) * len(non_training_paths)
        f.write('non_training_hr ' + ' '.join([str(val) for val in non_training_mus_hr]) + '\n')
        non_training_mus_lr /= len(sel_tracts) * len(non_training_paths)
        f.write('non_training_lr ' + ' '.join([str(val) for val in non_training_mus_lr]) + '\n')
        rb_dti_native /= len(sel_tracts)
        f.write('rb_dti_native ' + ' '.join([str(val) for val in rb_dti_native]) + '\n')
        rb_dti_foreign /= len(sel_tracts)
        f.write('rb_dti_foreign ' + ' '.join([str(val) for val in rb_dti_foreign]) + '\n')
        rb_qsdr_native /= len(sel_tracts)
        f.write('rb_qsdr_native ' + ' '.join([str(val) for val in rb_qsdr_native]) + '\n')
        rb_qsdr_foreign /= len(sel_tracts)
        f.write('rb_qsdr_foreign ' + ' '.join([str(val) for val in rb_qsdr_foreign]) + '\n')
        shuffle_mus /= len(sel_tracts)
        f.write('shuffle ' + ' '.join([str(val) for val in shuffle_mus]) + '\n')

    plt.figure(dpi=300)
    plt.semilogx(ks, training_mus_hr, label='TractSeg (HR Training)')
    # plt.semilogx(ks, training_mus_lr, label='TractSeg (LR Training)')
    # plt.semilogx(ks, non_training_mus_hr, label='TractSeg (HR Non-Training)')
    plt.semilogx(ks, non_training_mus_lr, label='TractSeg (LR Non-Training)')
    # plt.semilogx(ks, rb_dti_native, label='Recobundles (DTI, native)')
    # plt.semilogx(ks, rb_dti_foreign, label='Recobundles (DTI, foreign)')
    plt.semilogx(ks, rb_qsdr_native, label='Recobundles (QSDR, native)')
    # plt.semilogx(ks, rb_qsdr_foreign, label='Recobundles (QSDR, foreign)')
    plt.semilogx(ks, shuffle_mus, label='Randomized data')
    plt.xlabel('k')
    plt.ylabel('\u03bc\u2096')
    plt.legend()
    plt.show()


def run_recobundles(input_folder, atlas_file, ground_truth_folder, output_folder):
    print('Running Recobundles in ' + input_folder)

    # make a folder to save output
    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    except OSError:
        print('Could not create output dir. Aborting...')
        return

    # Uncomment for first exemplary use
    # target_file, target_folder = fetch_target_tractogram_hcp()
    # atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

    # target_file = get_target_tractogram_hcp()

    target_file = input_folder + 'whole_brain.trk'

    # use this line to select tracts if necessary
    sel_tracts = tracts
    # sel_bundle_paths = [data_path + 'tractseg/599469/tracts/AF_left.trk']
    # print(sel_bundle_paths)
    sft_atlas = load_trk(atlas_file, 'same', bbox_valid_check=False)
    atlas = sft_atlas.streamlines

    sft_target = load_trk(target_file, 'same', bbox_valid_check=False)
    target = sft_target.streamlines
    target_header = create_tractogram_header(target_file,
                                             *sft_atlas.space_attributes)

    target, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
        target, atlas, x0='affine', progressive=True)
    print(transform)
    sft_rec = StatefulTractogram(nib.streamlines.array_sequence.concatenate([target, atlas], 0), target_header, Space.RASMM)
    save_trk(sft_rec, output_folder + 'test.trk', bbox_valid_check=False)

    # errs = []
    # for t in sel_tracts:
    #     tract_file = t + '.trk'
    #     print(t)
    #     try:
    #         # tg = Tractogram(load_trk(ground_truth_folder + tract_file, 'same', bbox_valid_check=False).streamlines)
    #         # # tg.apply_affine(transform)
    #         # model = tg.streamlines
    #         # rb = RecoBundles(target, verbose=True)
    #         # recognized, labels = rb.recognize(model_bundle=model,
    #         #                                   model_clust_thr=5,
    #         #                                   # reduction_thr=10,
    #         #                                   slr=True,
    #         #                                   # corrected example here
    #         #                                   # slr_metric=BundleMinDistanceAsymmetricMetric(),
    #         #                                   # pruning_distance='mam'
    #         #                                   )
    #
    #         # sft_tg = StatefulTractogram(model, target_header, Space.RASMM)
    #         # save_trk(sft_tg, output_folder + 'affine_applied.trk', bbox_valid_check=False)
    #         #
    #         # sft_rec = StatefulTractogram(target[labels], target_header, Space.RASMM)
    #         sft_rec = StatefulTractogram(nib.streamlines.array_sequence.concatenate([target], 0), target_header, Space.RASMM)
    #         save_trk(sft_rec, output_folder + tract_file, bbox_valid_check=False)
    #
    #         mask, affine = to_bin_mask(sft_rec)
    #         affine[1][1] *= -1
    #         save_nifti(output_folder + tract_file[:-4] + '.nii.gz', mask, affine)
    #     except:
    #         errs.append('Error with ' + tract_file)
    #         print('\rError with ' + tract_file)


def make_atlas():
    print('Making atlas...')
    trk_list = []
    sft = None
    for t in tracts:
        print('\r' + t + '      ', end='')
        sft = load_trk(tractseg_dir + training_ids[1] + '/tracts/' + t + '.trk', 'same', bbox_valid_check=False)
        trk_list.append(sft.streamlines)
    out_file = training_paths[1] + 'T1w/Diffusion/atlas.trk'
    target_header = create_tractogram_header(out_file,
                                             *sft.space_attributes)
    sft_rec = StatefulTractogram(nib.streamlines.array_sequence.concatenate(trk_list, 0), target_header, Space.RASMM)
    save_trk(sft_rec, out_file, bbox_valid_check=False)


def plot_dices():
    tractseg_hr = np.zeros(len(tracts), dtype=float)
    tractseg_lr = np.zeros(len(tracts), dtype=float)
    for i in range(len(training_paths)):
        s_arr = []
        with open(training_paths[i] + 'T1w/dices_tractseg_hr.txt') as f:
            for line in f:
                s_arr.append(line.split()[1])
        tractseg_hr += np.array(s_arr, dtype=float)
        s_arr = []
        with open(training_paths[i] + 'T1w/dices_tractseg_lr.txt') as f:
            for line in f:
                s_arr.append(line.split()[1])
        tractseg_lr += np.array(s_arr, dtype=float)
    tractseg_hr /= len(training_paths)
    tractseg_lr /= len(training_paths)

    rb_hr_dti_native = []
    with open(training_paths[0] + 'T1w/dices_rb_hr_dti_native.txt') as f:
        for line in f:
            rb_hr_dti_native.append(float(line.split()[1]))
    rb_hr_dti_foreign = []
    with open(training_paths[0] + 'T1w/dices_rb_hr_dti_foreign.txt') as f:
        for line in f:
            rb_hr_dti_foreign.append(float(line.split()[1]))
    rb_hr_qsdr_native = []
    with open(training_paths[0] + 'T1w/dices_rb_hr_qsdr_native.txt') as f:
        for line in f:
            rb_hr_qsdr_native.append(float(line.split()[1]))
    rb_hr_qsdr_foreign = []
    with open(training_paths[0] + 'T1w/dices_rb_hr_qsdr_foreign.txt') as f:
        for line in f:
            rb_hr_qsdr_foreign.append(float(line.split()[1]))

    tractseg_pairs = [(tracts[i],
                       tractseg_hr[i],
                       tractseg_lr[i],
                       rb_hr_dti_native[i],
                       rb_hr_dti_foreign[i],
                       rb_hr_qsdr_native[i],
                       rb_hr_qsdr_foreign[i]) for i in range(len(tracts))]
    tractseg_pairs.sort(key=lambda pair: pair[1])
    y_labels = [pair[0] for pair in tractseg_pairs]
    tractseg_hr = [pair[1] for pair in tractseg_pairs]
    tractseg_lr = [pair[2] for pair in tractseg_pairs]
    rb_hr_dti_native = [pair[3] for pair in tractseg_pairs]
    rb_hr_dti_foreign = [pair[4] for pair in tractseg_pairs]
    rb_hr_qsdr_native = [pair[5] for pair in tractseg_pairs]
    rb_hr_qsdr_foreign = [pair[6] for pair in tractseg_pairs]

    rs = range(len(tracts))
    plt.figure(dpi=300)
    plt.plot(tractseg_hr, rs, 'o', color='lightcoral', label='TractSeg HR')
    plt.plot(tractseg_lr, rs, 'o', color='deepskyblue', label='TractSeg LR')
    plt.plot(rb_hr_dti_native, rs, 'P', color='olive', label='Recobundles (DTI, native)')
    plt.plot(rb_hr_dti_foreign, rs, 'X', color='navy', label='Recobundles (DTI, foreign)')
    plt.plot(rb_hr_qsdr_native, rs, 'd', color='darkorange', label='Recobundles (QSDR, native)')
    plt.plot(rb_hr_qsdr_foreign, rs, '*', color='orchid', label='Recobundles (QSDR, foreign)')

    plt.gcf().set_size_inches(8, 12)
    plt.yticks(rs, y_labels)
    axes = plt.gca()
    axes.set_xlim([.0, .9])
    axes.set_ylim([-1, len(tracts) + 1])
    plt.xlabel('Dice Score')
    plt.ylabel('Bundle')
    plt.grid(linestyle='--', lw=.5)
    plt.legend()
    plt.show()


# def tracking(folder):
#     print('Tracking in ' + folder)
#     output_folder = folder + 'dipy_out/'
#
#     # make a folder to save new data into
#     try:
#         Path(output_folder).mkdir(parents=True, exist_ok=True)
#     except OSError:
#         print('Could not create output dir. Aborting...')
#         return
#
#     # load data
#     print('Loading data...')
#     img = nib.load(folder + 'data.nii.gz')
#     dmri = np.asarray(img.dataobj)
#     affine = img.affine
#     mask, _ = load_nifti(folder + 'nodif_brain_mask.nii.gz')
#     bvals, bvecs = read_bvals_bvecs(folder + 'bvals', folder + 'bvecs')
#     gtab = gradient_table(bvals, bvecs)
#
#     # extract peaksoutput_folder + 'peak_vals.nii.gz'
#     if Path(output_folder + 'peaks.pam5').exists():
#         peaks = load_peaks(output_folder + 'peaks.pam5')
#     else:
#         print('Extracting peaks...')
#         response, ration = auto_response(gtab, dmri, roi_radius=10, fa_thr=.7)
#         csd_model = ConstrainedSphericalDeconvModel(gtab, response)
#
#         peaks = peaks_from_model(model=csd_model,
#                                  data=dmri,
#                                  sphere=default_sphere,
#                                  relative_peak_threshold=.5,
#                                  min_separation_angle=25,
#                                  parallel=True)
#
#         save_peaks(output_folder + 'peaks.pam5', peaks, affine)
#         scaled = peaks.peak_dirs * np.repeat(np.expand_dims(peaks.peak_values, -1), 3, -1)
#
#         cropped = scaled[:, :, :, :3, :].reshape(dmri.shape[:3] + (9, ))
#         save_nifti(output_folder + 'peaks.nii.gz', cropped, affine)
#         #save_nifti(output_folder + 'peak_dirs.nii.gz', peaks.peak_dirs, affine)
#         #save_nifti(output_folder + 'peak_vals.nii.gz', peaks.peak_values, affine)
#
#     # tracking
#     print('Tracking...')
#     maskdata, mask = median_otsu(dmri, vol_idx=range(0, dmri.shape[3]), median_radius=3,
#                                  numpass=1, autocrop=True, dilate=2)
#     tensor_model = TensorModel(gtab, fit_method='WLS')
#     tensor_fit = tensor_model.fit(maskdata)
#     fa = fractional_anisotropy(tensor_fit.evals)
#     fa[np.isnan(fa)] = 0
#     bla = np.average(fa)
#     tissue_classifier = ThresholdStoppingCriterion(fa, .1)
#     seeds = random_seeds_from_mask(fa > 1e-5, affine, seeds_count=1)
#
#     streamline_generator = LocalTracking(direction_getter=peaks,
#                                          stopping_criterion=tissue_classifier,
#                                          seeds=seeds,
#                                          affine=affine,
#                                          step_size=.5)
#     streamlines = Streamlines(streamline_generator)
#     save_trk(StatefulTractogram(streamlines, img, Space.RASMM), output_folder + 'whole_brain.trk')


set_paths()
# for p in training_paths:
#     downsample(p)
# for p in non_training_paths:
#     downsample(p)
# for t_id in training_ids:
#     downsample_segmentations(t_id)
# alltrks2bin()
# calc_dices()
# for p in training_paths:
#     tracking(p + 'T1w/Diffusion_low_res/')
# for p in training_paths:
#     tracking(p + 'T1w/Diffusion_low_res/')
calc_all_mus()
# make_atlas()
# run_recobundles(training_paths[0] + 'T1w/Diffusion/dti/',
#                 training_paths[0] + 'T1w/Diffusion/atlas.trk',
#                 tractseg_dir + training_ids[0] + '/tracts/',
#                 training_paths[0] + 'T1w/Diffusion/dti/recobundles_native/')
# run_recobundles(training_paths[0] + 'T1w/Diffusion/dti/',
#                 training_paths[1] + 'T1w/Diffusion/atlas.trk',
#                 tractseg_dir + training_ids[1] + '/tracts/',
#                 training_paths[0] + 'T1w/Diffusion/dti/recobundles_foreign/')
# run_recobundles(training_paths[0] + 'T1w/Diffusion/qsdr/',
#                 training_paths[0] + 'T1w/Diffusion/atlas.trk',
#                 tractseg_dir + training_ids[0] + '/tracts/',
#                 data_path + '/test/')
                # training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_native/')
# run_recobundles(training_paths[0] + 'T1w/Diffusion/qsdr/',
#                 training_paths[1] + 'T1w/Diffusion/atlas.trk',
#                 tractseg_dir + training_ids[1] + '/tracts/',
#                 training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_foreign/')
# run_recobundles(training_paths[0] + 'T1w/Diffusion/qsdr/')
# run_recobundles(training_paths[0] + 'T1w/Diffusion_low_res/')
# run_recobundles(training_paths[1] + 'T1w/Diffusion/dti/')
# run_recobundles(training_paths[1] + 'T1w/Diffusion/qsdr/')
# run_recobundles(training_paths[1] + 'T1w/Diffusion_low_res/')
# run_recobundles(non_training_paths[0] + 'T1w/Diffusion/dti/')
# run_recobundles(non_training_paths[0] + 'T1w/Diffusion/qsdr/')
# run_recobundles(non_training_paths[0] + 'T1w/Diffusion_low_res/')
# run_recobundles(non_training_paths[1] + 'T1w/Diffusion/dti/')
# run_recobundles(non_training_paths[1] + 'T1w/Diffusion/qsdr/')
# run_recobundles(non_training_paths[1] + 'T1w/Diffusion_low_res/')
# d_s(training_ids[0], training_paths[0] + 'T1w/Diffusion/dti/recobundles_output/', training_paths[0] + 'T1w/Diffusion/tractseg_output/bundle_segmentations/dices.txt')
# d_s(training_ids[0], training_paths[0] + 'T1w/Diffusion/dti/recobundles_foreign/', training_paths[0] + 'T1w/dices_rb_hr_dti_foreign.txt')
# d_s(training_ids[0], training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_native/', training_paths[0] + 'T1w/dices_rb_hr_qsdr_native.txt')
# d_s(training_ids[0], training_paths[0] + 'T1w/Diffusion/qsdr/recobundles_native/', training_paths[0] + 'T1w/dices_rb_hr_qsdr_foreign.txt')
# d_s(training_ids[0], training_paths[0] + 'T1w/Diffusion/dti/recobundles_native/', training_paths[0] + 'T1w/dices_rb_hr_dti_native.txt')
# for i in range(5):
#     d_s(non_training_ids[i], non_training_paths[i] + 'T1w/Diffusion/tractseg_output/bundle_segmentations/', non_training_paths[i] + 'T1w/dices_tractseg_hr.txt')
# for i in range(5):
#     d_s(training_ids[i], training_paths[i] + 'T1w/Diffusion_low_res/tractseg_output/bundle_segmentations/', training_paths[i] + 'T1w/dices_tractseg_lr.txt')
# plot_dices()
