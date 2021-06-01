from pathlib import Path

import matplotlib
from matplotlib import pylab as plt
from monai.transforms import *
import numpy as np

from preprocess.convert import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

loader = Compose([
    LoadImageD('img'),
    AddChannelD('img'),
    ResizeD('img', spatial_size=(240, 240, -1))
])

def plot_example():
    subgroup_examples = {
        'WNT': ('MBMR112ARN', 5, 1024),
        'SHH': ('MBMR105WSB', 7, 1024),
        'G3': ('MBMR163LWH', 7, 1024),
        'G4': ('MBMR148WTY', 5, 400),
    }
    # norm = plt.Normalize(vmin=0, vmax=1024)

    fig, axes = plt.subplots(4, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(7, 9), facecolor='black')
    for ax, protocol in zip(axes[0, :], ScanProtocol):
        ax.set_title(protocol.name, fontsize=25, color='white')

    for i, (group, (patient, slice_id, vmax)) in enumerate(subgroup_examples.items()):
        axes[i, 0].annotate(group, xy=(0, 0), xytext=(-0.25, 0.5), xycoords='axes fraction', color='white', fontsize=25, rotation=90, va='center')
        for ax, protocol in zip(axes[i, :], ScanProtocol):
            src = input_dir / patient / f'{protocol.name}.nii.gz'
            img = loader({'img': str(src)})
            data = img['img'][0][:, :, slice_id]
            ax.imshow(np.rot90(data), cmap='gray', aspect='auto', vmax=vmax)
            ax.axis('off')

    fig.savefig('mri-example.pdf', bbox_inches='tight')
    plt.show()

def plot_preprocess():
    # matplotlib.rcParams['font.family'] = ['Microsoft YaHei']
    steps = [
        ('原始图像', 'nifti', 9),
        ('去除头骨', 'skull-stripped', 9),
        ('配准与重采样', 'sri24', 47),
    ]
    norm = plt.Normalize(vmin=0, vmax=1024)
    patient = 'MBMR105WSB'
    fig, axes = plt.subplots(len(steps), 3, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(7, 8), facecolor='black')
    for ax, protocol in zip(axes[0, :], ScanProtocol):
        ax.set_title(protocol.name, fontsize=25, color='white')
    for i, (op, data_dir, slice_id) in enumerate(steps):
        data_dir = Path(data_dir)
        axes[i, 0].annotate(op, xy=(0, 0), xytext=(-0.15, 0.5), xycoords='axes fraction', color='white', fontsize=18, rotation=90, va='center')
        axes[i, 0].annotate(chr(i + ord('a')), xy=(0, 0), xytext=(-0.4, 0.5), xycoords='axes fraction', color='white', fontsize=18, va='center')
        for ax, protocol in zip(axes[i, :], ScanProtocol):
            ax.axis('off')
            src = data_dir / patient / f'{protocol.name}.nii.gz'
            img = loader({'img': str(src)})
            data = img['img'][0][:, :, slice_id]
            print(data.shape)
            ax.imshow(np.rot90(data), cmap='gray', aspect='auto', norm=norm)

    fig.savefig('preprocess.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_example()
    plot_preprocess()
