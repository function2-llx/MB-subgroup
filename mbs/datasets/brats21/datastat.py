from pathlib import Path

import numpy as np

if __name__ == '__main__':
    for subject in Path('processed').iterdir():
        if not subject.is_dir():
            continue
        seg = np.load(subject / 'data.npz')['seg']
        print((seg == 3).sum(), (seg == 2).sum())
