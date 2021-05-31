from matplotlib import pylab as plt
from monai.transforms import *

from preprocess.convert import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

loader = Compose([
    LoadImageD('img'),
    AddChannelD('img'),
    ResizeD('img', (224, 244, -1)),
])

def plot_example():
    subgroup_examples = {
        'WNT': ('MBMR112ARN', 7),
        'SHH': ('MBMR105WSB', 7),
        'G3': ('MBMR163LWH', 7),
        'G4': ('MBMR148WTY', 5)
    }

    fig, axes = plt.subplots(4, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(6, 9), facecolor='black')
    for i, (group, (patient, slice)) in enumerate(subgroup_examples.items()):
        plt.setp(axes[i, :], ylabel=group)
        for j, protocol in enumerate(ScanProtocol):
            src = input_dir / patient / f'{protocol.name}.nii.gz'
            img = loader({'img': str(src)})
            data = img['img'][0][:, :, slice]
            axes[i, j].imshow(data.transpose(), cmap='gray', aspect='auto')
            axes[i, j].axis('off')
    fig.savefig('mri-example.pdf', bbox_inches='tight')
    plt.show()

# def plot_preprocess():
#     fig, axes = plt.subplots(4, 3, gridspec_kw={'wspace': 0, 'hspace': 0.01}, squeeze=True, figsize=(6, 9))
#
#     for i, (group, (patient, slice)) in enumerate(subgroup_examples.items()):
#         plt.setp(axes[i, :], ylabel=group)
#         for j, protocol in enumerate(ScanProtocol):
#             src = input_dir / patient / f'{protocol.name}.nii.gz'
#             img = loader({'img': str(src)})
#             data = img['img'][0][:, :, slice]
#             axes[i, j].imshow(data.transpose(), cmap='gray', aspect='auto')
#             axes[i, j].axis('off')
#     fig.savefig('mri-example.pdf', bbox_inches='tight')
#     plt.show()

if __name__ == '__main__':
    plot_example()
