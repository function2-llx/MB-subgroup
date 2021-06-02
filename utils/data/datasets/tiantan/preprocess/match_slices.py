import json

import matplotlib.pyplot as plt
import nibabel as nib
from utils.dicom_utils import ScanProtocol
from matplotlib import pyplot
from monai.transforms import Resize

from histogram_matching import output_dir as matched_dir
from registration import output_dir as registered_dir

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    protocol = ScanProtocol.T2
    # resize = Resize(spatial_size=(240, 240, -1))
    for info in cohort[2:]:
        patient = info['patient']
        print(patient)
        matched = nib.load(matched_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        matched = Resize(spatial_size=(240, 240, -1))(matched[None, :, :, :])[0]
        registered = nib.load(registered_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        # registered = Resize(spatial_size=(-1, -1, matched.shape[2]))(registered[None, :, :, :])[0]

        for i in range(matched.shape[2]):
            print(i)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(matched[:, :, i], cmap='gray')
            ax2.imshow(registered[:, :, i * registered.shape[2] // matched.shape[2]], cmap='gray')
            plt.show()
        break
        # registered_slice_id = 0
        # for matched_slice_id in range(matched.shape[2]):
        #     matched_slice = matched[:, :, matched_slice_id]
