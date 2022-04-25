import json

import matplotlib.pyplot as plt
import nibabel as nib
from mbs.utils.dicom_utils import ScanProtocol
from matplotlib import pyplot

from histogram_matching import output_dir as matched_dir
from registration import output_dir as registered_dir

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    protocol = ScanProtocol.T1
    for info in cohort:
        patient = info['patient']
        print(patient)
        matched = nib.load(matched_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        registered = nib.load(registered_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        print(matched.shape)
        print(registered.shape)
        # print(registered.size)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        v_max = max(matched.max(), registered.max())
        ax1.hist(matched.ravel(), bins=256, range=(1, v_max))
        ax2.hist(registered.ravel(), bins=256, range=(1, v_max))
        plt.show()
        break
