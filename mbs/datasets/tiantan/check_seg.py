from pathlib import Path
from typing import Optional

import monai.transforms as monai_transforms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.colors import ListedColormap



from mbs.utils.dicom_utils import ScanProtocol

data_dir = Path(__file__).parent / 'preprocessed'

class IndexTracker:
    def __init__(self, ax: Axes):
        self.ax = ax

    def set_data(self, img: Optional[np.ndarray], seg: Optional[np.ndarray], title: str):
        assert img.shape[2] == seg.shape[2]

        self.ax.clear()
        self.ax.set_title(title)
        self.img = img
        self.seg = seg.astype(int)
        self.seg_visible = True
        rows, cols, self.n_slices = img.shape
        # self.ind = np.argmax(seg.sum(axis=(0, 1)))
        self.ind = 5
        self.ax_img = self.ax.imshow(self.img[:, :, self.ind], cmap='gray', vmax=1000)
        self.ax_seg = self.ax.imshow(self.seg[:, :, self.ind], cmap=ListedColormap(['none', 'green']), alpha=0.5, vmin=0, vmax=1, interpolation='nearest')
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
        self.ax.set_ylabel(f'slice {self.ind}')

class Explorer:
    def __init__(self, data_dir: Path, info: pd.DataFrame):
        self.data_dir = data_dir
        self.info = info
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

        self.trackers = {
            ax: IndexTracker(ax)
            for ax in (ax1, ax2, ax3)
        }
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.set_patient(0)

    def on_key_press(self, event: KeyEvent):
        if event.key == 'left':
            if self.patient_id == 0:
                return
            self.set_patient(self.patient_id - 1)
        elif event.key == 'right':
            if self.patient_id + 1 == len(self.info):
                return
            self.set_patient(self.patient_id + 1)
        else:
            return
        self.update()

    def on_click(self, event: MouseEvent):
        tracker = self.trackers.get(event.inaxes, None)
        if tracker is None:
            return
        tracker.on_click(event)
        self.update()

    def on_scroll(self, event: MouseEvent):
        tracker = self.trackers.get(event.inaxes, None)
        if tracker is None:
            return
        tracker.on_scroll(event)
        self.update()

    def update(self):
        self.fig.canvas.draw()

    def set_patient(self, patient_id):
        self.patient_id = patient_id
        patient = self.info.index[patient_id]
        patient_dir = self.data_dir / patient
        row = self.info.iloc[patient_id]
        self.fig.suptitle(f'{patient_id} / {len(self.info)} {patient}')
        for (protocol, seg_type), tracker in zip([
            (ScanProtocol.T2, 'AT'),
            (ScanProtocol.T1c, 'CT'),
            (ScanProtocol.T1, 'CT'),
        ], self.trackers.values()):
            tracker: IndexTracker
            if pd.isna(row[protocol.name]) or pd.isna(row[seg_type]):
                tracker.ax.set_visible(False)
                continue

            tracker.ax.set_visible(True)
            data = loader({
                'img': patient_dir / row[protocol.name],
                'seg': patient_dir / row[seg_type],
            })
            img = data['img']
            seg = data['seg']
            tracker.set_data(img, seg, f'{protocol.name} {seg_type}')

loader = monai_transforms.Compose([
    monai_transforms.LoadImageD(('img', 'seg')),
    monai_transforms.AddChannelD(('img', 'seg')),
    # monai_transforms.NormalizeIntensityD('img'),
    # monai_transforms.ScaleIntensityD('img'),
    monai_transforms.OrientationD(('img', 'seg'), 'LAS'),
    monai_transforms.TransposeD(('img', 'seg'), (0, 2, 1, 3)),
    monai_transforms.Lambda(lambda data: {
        **data,
        'seg': monai_transforms.Resize(data['img'].shape[1:])(data['seg'])
    }),
    monai_transforms.LambdaD(('img', 'seg'), lambda x: np.rot90(x[0], k=2)),
])

# def plot_seg(patient: str, seg_type: str, patient_dir: Path, img_path: PathLike, seg_path: PathLike):
#     data = loader({
#         'img': patient_dir / img_path,
#         'seg': patient_dir / seg_path,
#     })
#     img = data['img']
#     seg = data['seg']
#     # img = loader(patient_dir / img_path)[0].transpose((1, 0, 2))
#     # # seg = np.rot90(np.rot90(loader(patient_dir / seg_path)[0].transpose((1, 0, 2))))
#     # seg = loader(patient_dir / seg_path)[0].transpose((1, 0, 2))
#     ax: Axes
#     fig, ax = plt.subplots(1, 1)
#     ax.set_title(f'{patient} {seg_type}\n{img.shape} {seg.shape}')
#     tracker = IndexTracker(ax, img, seg)
#     fig.canvas.mpl_connect('button_press_event', tracker.on_click)
#     fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
#     plt.show()


def main():
    results = pd.read_excel('cohort.xlsx').set_index('name(raw)')
    explorer = Explorer(data_dir, results)
    plt.show()
    # for patient, row in results.iterrows():
    #     patient_dir = data_dir / patient
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))
    #
    #     fig: Figure
    #     fig.suptitle(patient)
    #     ax_map: Dict[Axes, IndexTracker] = {}
    #
    #     for protocol, seg_type, ax in [
    #         (ScanProtocol.T2, 'AT', ax1),
    #         (ScanProtocol.T1c, 'CT', ax2),
    #         (ScanProtocol.T1, 'CT', ax3),
    #     ]:
    #         ax: Axes
    #         if not pd.isna(row[protocol.name]) and not pd.isna(row[seg_type]):
    #             data = loader({
    #                 'img': patient_dir / row[protocol.name],
    #                 'seg': patient_dir / row[seg_type],
    #             })
    #             img = data['img']
    #             seg = data['seg']
    #             ax.set_title(f'{protocol.name} {seg_type}\n{img.shape} {seg.shape}')
    #             tracker = IndexTracker(ax, img, seg)
    #             ax_map[ax] = tracker
    #     fig.canvas.mpl_connect('button_press_event', lambda event: ax_map[event.inaxes].on_click(event) if event.inaxes in ax_map else None)
    #     fig.canvas.mpl_connect('scroll_event', lambda event: ax_map[event.inaxes].on_scroll(event) if event.inaxes in ax_map else None)
    #     plt.show()

if __name__ == '__main__':
    main()
