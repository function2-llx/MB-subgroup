import os
import csv

import numpy as np
import png
import pydicom
from tqdm import tqdm


def get_ornt(ds):
    ornt = np.array(ds.ImageOrientationPatient)
    p = sorted(np.argpartition(-abs(ornt), 2)[:2])
    if p == [0, 4]:
        return 'up'
    elif p == [1, 5]:
        return 'left'
    elif p == [0, 5]:
        return 'back'
    else:
        raise ValueError('cannot determine orientation')


if __name__ == '__main__':
    import random
    random.seed(23333)
    labels = list(csv.reader(open('../data-dicom/mst-labels.csv')))

    tot = 0
    for name, _, _ in labels:
        for dirpath, _, filenames in os.walk(f'../data-dicom/{name}'):
            for filename in filenames:
                tot += filename.endswith('.dcm')
    bar = tqdm(ncols=80, total=tot)
    for (name, _, mst), split in zip(labels, random.choices(['train', 'val'], weights=[3, 1], k=len(labels))):
        cnt = {k: 0 for k in ['back', 'left', 'up']}
        for dirpath, _, filenames in os.walk(f'../data-dicom/{name}'):
            for filename in filenames:
                if not filename.endswith('.dcm'):
                    continue
                ds = pydicom.dcmread(os.path.join(dirpath, filename))
                try:
                    shape = ds.pixel_array.shape
                except:
                    bar.update()
                    continue
                assert len(shape) == 2
                ornt = get_ornt(ds)
                output_dir = f'data/{split}/{mst}/{ornt}'
                os.makedirs(output_dir, exist_ok=True)
                # Convert to float to avoid overflow or underflow losses.
                image_2d = ds.pixel_array.astype(float)
                # Rescaling grey scale between 0-255
                image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
                # Convert to uint
                image_2d_scaled = np.uint8(image_2d_scaled)
                # Write the PNG file
                with open(os.path.join(output_dir, f'{name}-{cnt[ornt]}.png'), 'wb') as png_file:
                    w = png.Writer(shape[1], shape[0], greyscale=True)
                    w.write(png_file, image_2d_scaled)
                cnt[ornt] += 1
                bar.update()
    bar.close()
