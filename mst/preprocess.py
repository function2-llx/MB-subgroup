import os
import csv
import shutil


splits = ['train', 'val']


if __name__ == '__main__':
    import random
    random.seed(23333)
    for name, _, mst in csv.reader(open('../data/label.csv')):
        print(name)
        weights = [3, 1]
        samples = sum([[split] * weight for split, weight in zip(splits, weights)], [])
        for split in splits:
            os.makedirs(f'data/{split}/{mst}', exist_ok=True)
        cnt = {split: 0 for split in splits}
        for dirpath, _, filenames in os.walk(f'../data/{name}'):
            for filename in filenames:
                if not filename.endswith('.png'):
                    continue
                split = random.choice(samples)
                shutil.copy(os.path.join(dirpath, filename), f'data/{split}/{mst}/{name}-{cnt[split]}.png')
                cnt[split] += 1
