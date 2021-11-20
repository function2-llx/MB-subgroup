from pathlib import Path

import json
import numpy as np
from monai.networks.nets import SegResNet
from tqdm.contrib.concurrent import process_map

data_dir = Path(__file__).parent / 'processed'

def count_subject(subject):
    seg = np.load(str(data_dir / f'{subject}/data.npz'))['seg'][0]
    return (np.sum(seg, axis=(0, 1)) > 0).sum()

def main():
    cnt = process_map(count_subject, json.load(open(data_dir / 'subjects.json')))
    print(cnt)
    print(sum(cnt))

if __name__ == '__main__':
    main()
