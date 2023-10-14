import json
from pathlib import Path

from nnunetv2.paths import nnUNet_preprocessed

from mbs.datamodule import load_split

split = load_split()
split = split[split != 'test']

def main():
    (Path(nnUNet_preprocessed) / 'Dataset500_TTMB' / 'splits_final.json').write_text(
        json.dumps(
            [
                {
                    'train': split.index[split != i].tolist(),
                    'val': split.index[split == i].tolist(),
                }
                for i in range(5)
            ],
            indent=4, ensure_ascii=False
        )
    )


if __name__ == '__main__':
    main()
