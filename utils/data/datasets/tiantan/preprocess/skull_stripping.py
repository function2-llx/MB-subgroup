from copy import deepcopy

from HD_BET.run import run_hd_bet

from convert import output_dir as nifti_dir

if __name__ == '__main__':
    mri_fnames = list(nifti_dir.glob('*/*.nii.gz'))
    out_fnames = []
    for mri_fname in mri_fnames:
        out_fname = deepcopy(mri_fname)
        out_fname.name = f'{mri_fname.name[:-7]}_ss.nii.gz'
        out_fnames.append(out_fname)

    run_hd_bet(list(map(str, mri_fnames)), out_fnames, postprocess=True)
