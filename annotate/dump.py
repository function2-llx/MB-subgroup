import os

import numpy as np
import png
import pydicom
from tqdm import tqdm


def files_cnt(dirpath):
    ret = 0
    for _, _, filenames in os.walk(dirpath):
        for filename in filenames:
            ret += filename.endswith('.dcm')
    return ret


if __name__ == '__main__':
    for patient in os.listdir('.'):
        if not os.path.isdir(patient):
            continue
        if files_cnt(patient) == 0:
            print(patient)

    tot = 0
    for dirpath, _, filenames in os.walk('.'):
        for filename in filenames:
            tot += filename.endswith('.dcm')

    bar = tqdm(ncols=80, total=tot)
    for dirpath, _, filenames in os.walk('.'):
        for filename in filenames:
            if not filename.endswith('.dcm'):
                continue
            ds = pydicom.dcmread(os.path.join(dirpath, filename))
            try:
                pixel = ds.pixel_array
                shape = pixel.shape
                assert len(shape) == 2
            except:
                bar.update()
                continue
            # Convert to float to avoid overflow or underflow losses.
            image_2d = pixel.astype(float)
            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)
            # plt.imshow(image_2d_scaled, 'gray')
            # plt.show()
            # Write the PNG file
            with open(os.path.join(dirpath, filename[:-4] + '.png'), 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
            bar.update()
    bar.close()
