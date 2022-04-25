import json
from copy import deepcopy
from collections import Counter

from monai import transforms as monai_transforms

from preprocess.convert import output_dir

loader = monai_transforms.Compose([
    monai_transforms.LoadImaged('img'),
    monai_transforms.AddChanneld('img'),
    # monai_transforms.Orientationd('img', axcodes='LAS'),
    # monai_transforms.SpacingD('img', spacing)
    # monai_transforms.ThresholdIntensityd('img', threshold=0),
])



if __name__ == '__main__':
    counter = Counter()
    cohort = json.load(open('cohort.json'))
    for info in cohort:
        for scan in info['scans']:
            scan_info = json.load(open(output_dir / info['patient'] / f'{scan}.json'))
            scan_path = output_dir / info['patient'] / f'{scan}_ss.nii.gz'
            print(scan_path)
            monai_transforms.SaveImage
            origin = loader({'img': str(scan_path)})
            spacing = monai_transforms.SpacingD('img', pixdim=(1, 1, origin['img_meta_dict']['pixdim'][3]))
            spaced = spacing(deepcopy(origin))
            counter[spaced['img'].shape] += 1
            # img = spacing(deepcopy(img))
            # print('after:', img['img'].shape[1:], img['img_meta_dict']['pixdim'])
            # resample_path = output_dir / info['patient'] / f'{scan}_re.nii.gz'
            # print(img[['original_affine'])
            # del img['img_meta_dict']['original_affine']
            # monai_transforms.SaveImageD('img')(img)
            # exit(0)
