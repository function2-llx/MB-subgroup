from pathlib import Path

import matplotlib
from matplotlib import pylab as plt
from monai.transforms import *
import numpy as np

from preprocess.convert import output_dir as input_dir
from mbs.utils.dicom_utils import ScanProtocol

plot_dir = Path('plots')

def plot_example():
    loader = Compose([
        LoadImageD('img'),
        AddChannelD('img'),
        ResizeD('img', spatial_size=(240, 240, -1))
    ])
    subgroup_examples = {
        'WNT': ('MBMR112ARN', 5, 700),
        'SHH': ('MBMR105WSB', 7, 1024),
        'G3': ('MBMR163LWH', 7, 1200),
        'G4': ('MBMR148WTY', 5, 500),
    }

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

    fig.savefig(plot_dir / 'mri-example.pdf', bbox_inches='tight')
    plt.show()

def plot_preprocess():
    loader = Compose([
        LoadImageD('img'),
        AddChannelD('img'),
        ResizeD('img', spatial_size=(240, 240, -1))
    ])
    steps = [
        ('原始图像', 'nifti', 9),
        ('脑部提取', 'stripped', 9),
        ('强度正则化', 'matched', 9),
        ('配准与重采样', 'sri24', 47),
    ]
    norm = plt.Normalize(vmin=0, vmax=1024)
    # norm = None
    patient = 'MBMR105WSB'
    fig, axes = plt.subplots(len(steps), 3, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(7, 9), facecolor='black')
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
            ax.imshow(np.rot90(data), cmap='gray', aspect='auto', norm=norm)

    fig.savefig(plot_dir / 'preprocess.pdf', bbox_inches='tight')
    plt.show()

def plot_histograms():
    loader = LoadImage(image_only=True)
    patients = ['MBMR130LCY', 'MBMR148WTY']
    steps = [
        ('原始图像（T1）', 'nifti', 32),
        ('脑部提取', 'stripped', 256),
        ('N4偏置校正', 'n4itk', 256),
        ('强度直方图匹配', 'matched', 256),
    ]
    fig, axes = plt.subplots(len(steps), len(patients), gridspec_kw={'wspace': 0.3, 'hspace': 0.3}, squeeze=True, figsize=(7, 9), sharex='row', sharey='row')
    protocol = ScanProtocol.T1
    for j, ax in enumerate(axes[0, :]):
        ax.set_title(f"患者{j + 1}", fontsize=18, pad=20)

    for i, (op, data_dir, bins) in enumerate(steps):
        data_dir = Path(data_dir)
        axes[i, 0].annotate(op, xy=(0, 0), xytext=(-0.35, 0.5), xycoords='axes fraction', fontsize=15, rotation=90, va='center')
        data = list(map(lambda x: loader(str(data_dir / x / f'{protocol.name}.nii.gz')).ravel(), patients))
        v_max = max(map(np.ndarray.max, data))
        for ax, values in zip(axes[i, :], data):
            ax.hist(values, bins=bins, range=(1, v_max))

    fig.savefig(plot_dir / 'histogram.pdf', bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    # plot_example()
    # plot_preprocess()
    plot_histograms()
