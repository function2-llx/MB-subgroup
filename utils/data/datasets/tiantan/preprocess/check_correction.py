import json

import numpy as np
from matplotlib import pylab as plt
import nibabel as nib

from convert import output_dir as origin_dir
from correction import output_dir as corrected_dir
from utils.dicom_utils import ScanProtocol

# patient = 'old_MBMR044GYH'
protocol: ScanProtocol

def check(info):
    patient = info['patient']
    origin = nib.load(origin_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
    corrected = nib.load(corrected_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
    print('origin:', np.histogram(origin))
    print('corrected:', np.histogram(corrected))

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    for protocol in ScanProtocol:
        check(cohort[0])
        # ax_o.hist(origin.ravel(), bins=256, range=(1, 1000))
        # ax_o.set_title('origin')
        # ax_c.hist(corrected.ravel(), bins=256, range=(1, 1000))
        # ax_c.set_title('corrected')
        # plt.show()
