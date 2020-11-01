import os
import random
import shutil

random.seed(233333)

data_dir = 'data-20201030'
output_dir = 'data'

if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if dirnames:
            continue
        split = random.choices(['train', 'val'], [3, 1])[0]
        target_dir = os.path.join(output_dir, split, *dirpath.split(os.path.sep)[1:])
        os.makedirs(target_dir, exist_ok=True)
        for filename in filenames:
            if not filename.endswith('.dcm'):
                continue
            filename = filename[:-3] + 'png'
            shutil.copy(os.path.join(dirpath, filename), os.path.join(target_dir, filename))
