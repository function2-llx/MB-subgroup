from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import ListedColormap
import monai.transforms as monai_transforms
from matplotlib.figure import Figure

from utils import PathLike
from utils.dicom_utils import ScanProtocol

data_dir = Path(__file__).parent / 'origin'

class IndexTracker:
    def __init__(self, ax: Axes, img: np.ndarray, seg: np.ndarray):
        self.ax = ax

        self.img = img
        self.seg = seg.astype(int)
        self.seg_visible = True

        rows, cols, self.n_slices = img.shape
        self.ind = np.argmax(seg.sum(axis=(0, 1)))

        self.ax_img = ax.imshow(self.img[:, :, self.ind], cmap='gray')
        self.ax_seg = ax.imshow(self.seg[:, :, self.ind], cmap=ListedColormap(['none', 'green']), alpha=0.5, vmin=0, vmax=1)
        self.update()

    def on_scroll(self, event: MouseEvent):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = min(self.n_slices - 1, self.ind + 1)
        else:
            self.ind = max(0, self.ind - 1)
        # print(self.ind)
        self.update()

    def on_click(self, event: MouseEvent):
        self.seg_visible ^= 1
        # print('click!', self.seg_visible)
        self.update()

    def update(self):
        self.ax_img.set_data(self.img[:, :, self.ind])
        self.ax_seg.set_data(self.seg[:, :, self.ind])
        self.ax_seg.set_visible(self.seg_visible)
        self.ax.set_ylabel('slice %s' % self.ind)

        self.ax_img.axes.figure.canvas.draw()

loader = monai_transforms.Compose([
    monai_transforms.LoadImaged(('img', 'seg')),
    monai_transforms.AddChanneld(('img', 'seg')),
    monai_transforms.Orientationd(('img', 'seg'), 'LAS'),
    monai_transforms.Transposed(('img', 'seg'), (0, 2, 1, 3)),
    monai_transforms.Lambdad(('img', 'seg'), lambda x: np.rot90(x[0], k=2)),
])

def plot_seg(patient: str, seg_type: str, patient_dir: Path, img_path: PathLike, seg_path: PathLike):
    data = loader({
        'img': patient_dir / img_path,
        'seg': patient_dir / seg_path,
    })
    img = data['img']
    seg = data['seg']
    # img = loader(patient_dir / img_path)[0].transpose((1, 0, 2))
    # # seg = np.rot90(np.rot90(loader(patient_dir / seg_path)[0].transpose((1, 0, 2))))
    # seg = loader(patient_dir / seg_path)[0].transpose((1, 0, 2))
    ax: Axes
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'{patient} {seg_type}\n{img.shape} {seg.shape}')
    tracker = IndexTracker(ax, img, seg)
    fig.canvas.mpl_connect('button_press_event', tracker.on_click)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

results = pd.read_excel('results.xlsx', index_col=0)

def main():
    for patient, row in results.iterrows():
        patient_dir = data_dir / patient
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

        fig: Figure
        fig.suptitle(patient)
        ax_map: Dict[Axes, IndexTracker] = {}

        for protocol, seg_type, ax in [
            (ScanProtocol.T2, 'AT', ax1),
            (ScanProtocol.T1c, 'CT', ax2),
            (ScanProtocol.T1, 'CT', ax3),
        ]:
            ax: Axes
            if not pd.isna(row[protocol.name]) and not pd.isna(row[seg_type]):
                data = loader({
                    'img': patient_dir / row[protocol.name],
                    'seg': patient_dir / row[seg_type],
                })
                img = data['img']
                seg = data['seg']
                ax.set_title(f'{protocol.name} {seg_type}\n{img.shape} {seg.shape}')
                tracker = IndexTracker(ax, img, seg)
                ax_map[ax] = tracker
        fig.canvas.mpl_connect('button_press_event', lambda event: ax_map[event.inaxes].on_click(event) if event.inaxes in ax_map else None)
        fig.canvas.mpl_connect('scroll_event', lambda event: ax_map[event.inaxes].on_scroll(event) if event.inaxes in ax_map else None)
        plt.show()

if __name__ == '__main__':
    main()
