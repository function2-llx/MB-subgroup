# resample skull-stripped images

from monai.transforms import *
import nibabel as nib

loader = Compose([
    LoadImageD('img'),
    AddChannelD('img'),
])

if __name__ == '__main__':
    