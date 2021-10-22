import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.axes import Axes
from monai.transforms import LoadImage

from utils.dicom_utils import ScanProtocol

loader = LoadImage()

class IndexTracker:
    def __init__(self, ax: Axes, img: np.ndarray, seg: np.ndarray):
        print(img.shape)
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.img = img
        self.seg = seg.astype(int)
        self.seg_visible = True

        rows, cols, self.n_slices = img.shape
        self.ind = self.n_slices // 2

        self.ax_img = ax.imshow(self.img[:, :, self.ind], cmap='gray')
        self.ax_seg = ax.imshow(self.seg[:, :, self.ind], cmap=colors.ListedColormap(['none', 'green']), alpha=0.5, vmin=0, vmax=1)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = min(self.n_slices - 1, self.ind + 1)
        else:
            self.ind = max(0, self.ind - 1)
        self.update()

    def on_click(self, event):
        self.seg_visible ^= 1
        print('click!', self.seg_visible)
        self.update()

    def update(self):
        self.ax_img.set_data(self.img[:, :, self.ind])
        self.ax_seg.set_visible(self.seg_visible)
        self.ax_seg.set_data(self.seg[:, :, self.ind])

        self.ax_img.axes.figure.canvas.draw()
        self.ax.set_ylabel('slice %s' % self.ind)

def plot_slices(img: np.ndarray, mask: np.ndarray):
    img = img.transpose((1, 0, 2))
    mask = np.rot90(np.rot90(mask.transpose((1, 0, 2))))
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, img, mask)
    fig.canvas.mpl_connect('button_press_event', tracker.on_click)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

results_path = Path('seg_check.json')
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
else:
    results = {}


def check_patient(patient_dir: Path):
    patient = patient_dir.name
    if patient in results:
        return

    files = list(filter(lambda s: not s.name.startswith('.'), patient_dir.glob('*.nii.gz')))

    def find_file(pattern: str) -> Optional[str]:
        pattern = pattern.lower()
        for file in files:
            if file.name.lower().endswith(f'{pattern}.nii.gz'):
                return file.name
        return None

    result = {}
    for protocol in ScanProtocol:
        result[protocol.name] = find_file(protocol.name)
    for seg in ['AT', 'CT', 'WT']:
        result[seg] = find_file(seg)

    results[patient] = result
    if sum(v is not None for v in result.values()) < 5:
        print(patient, list(map(lambda file: file.name, files)))

def main():
    data_dir = Path('MB-MRI(2021-10-07)')
    try:
        for patient in data_dir.iterdir():
            check_patient(patient)
    finally:
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        pd.DataFrame(results).transpose().to_excel('results.xlsx')

if __name__ == '__main__':
    main()
