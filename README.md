# cpcg

## Team's diary

### Lucas
report parts: 1.2, 1.4, 5, 6
* KW20 literature on FBA, installation and first tutorials for dipy, general literature on MRI / DWI / DTI
* KW21 literature review, first version of paragraphs about AD, HCP, finished paragraph about AD
* KW22/23 reading into hcp data sets, communication with data admin, result: they do not offer tractographies, appr. 10h work
* KW24 (unsuccessfully) tried to install FLS / MRTrix on my laptop, worked on re-creating tract-seg ground truth pipeline
* KW25 "ground truth" script aborted on all computers avail. to me due to lack of memory, now running on Nils' computer (32GB), preparation of slides for intermediate presentation
* KW26 clarifaction of task as stated in WhatsApp on 24.06., paragraph about ADNI, simple descriptions of how we will evaluate
* KW28 implementation in dipy, generation of data sets to compare (tractseg and binary masks from ground truth), development of measure for non-training data, description of evaluation techniques in report
* KW29 implementated in dipy, generated data for Recobundles, ran TractSeg (without tracking stage) and Recobundles
* KW30 implemented evaluation tools (dice, plots), wrote evaluation
* KW31 reviewed / restructured report, finalized intro + evaluation, wrote conclusion

### Liza
* KW21 general review of tools and literature regarded MRI, installation and first tutorials for dipy, denoising of demo dataset
* KW22 literature review of preprocessing and denoising, denoising of given dataset
* KW23 literature review of skull stripping, skull stripping of demo dataset, skull stripping of given dataset 
* KW24 literature review of motion correction and data registration, data registration of demo dataset
* KW25 preparation of slides and speech for intermediate presentation
* KW26 clarification of further task
* KW27 data registration of given dataset
* KW28 preparation of report (introduction and preprocessing chapters)
* KW 29 preparation of report (preprocessing chapter)


### Nils
* KW21 Broad Literature review on Diffusion MRI and fibre tracking
* KW22 Review on DIPY for MRI processing
* KW23 Literature research on unconstraint and constraint spherical deconvoltion
* KW24 Use of DIPY to process preliminarily denoised data using CSD and peak extraction
* KW25 preliminary presentation & preparation
* KW26 Clarification of further task
* KW27 Definition of processing pipeline for proveded dataset, adjustments regarding export needs
* KW28 TractSeg running to create comparison tracs for further analysis
* KW29: Writeup of reconstruction
* KW30: Registration of Peak data using RegF3d
* KW31: Finishing Writeup, Histogramm based comparison

### Raveen
* **KW20** Initial Literature review on topics: MRI and dMRI
* **KW21** Literature review on topics: DTI, CSD, and Tractography
* **KW22** Explore and install tools: FMRIB Software Library, FreeSurfer Software Suite, MRtrix3, DIPY, and more GUIs for visualization.
* **KW23** Familiarize with TractSeg pipeline. Read about UNet architecture by Ronneberger et al.
* **KW24** Obtain tracks by utilizing TractSeg and FSL on publicly available data from HCP. Going through FSL courses. Preparation of introduction (a part of the introduction) for the report.
* **KW25** Preparation for the intermedial presentation and additional reading on Ball and Sticks model for fiber orientation modeling.
* **KW26&27** Discussion on how to proceed and further. Read about VoxelMorph and implementing it. Requires DIPY usage for altering the volume and registering to Talairach space 
       unlike TractSeg, which uses MNI space. Unfortunately, results where not as expected but stumbled upon NiftyReg, which has the highest accuracy for the registration task.
* **KW28&29** Resuming the preparation of the introduction (a part of the introduction) for the report. Used NiftyReg for the registration task. Unfortunately lost all the data due           to board failure.
* **KW30** Update and compile sections 4 to 4.3 and a part of 4.4 of the report.
* **KW31** Editing and revision of the report.
