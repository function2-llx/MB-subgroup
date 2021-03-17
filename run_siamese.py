import itertools
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import monai
import torch
import torch_optimizer as optim
from monai.data import DataLoader
from monai.transforms import *
from torch import nn
from torch.nn import DataParallel, MSELoss
from tqdm import tqdm

from siamese import generate_model, load_pretrained_model, PairDataset, Siamese
from utils.data_3d import load_folds, MultimodalDataset, ToTensorDeviced
from utils.dicom_utils import ScanProtocol
from utils.report import Reporter


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='equal')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_root', type=Path, default='output_siamese')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--targets', choices=['all', 'G3G4'], default='all')
    parser.add_argument(
        '--pretrained_name',
        default='r3d18_KM_200ep',
        choices=[
            'r3d18_K_200ep',
            'r3d18_KM_200ep',
            'r3d34_K_200ep',
            'r3d34_KM_200ep',
            'r3d50_KMS_200ep',
            'r2p1d18_K_200ep',
            'r2p1d34_K_200ep',
        ],
        type=str,
        help='Pretrained model name'
    )
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_slices',
                        default=16,
                        type=int,
                        help='slices of inputs, temporal size in terms of videos')
    parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='weak')
    args = parser.parse_args()
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    return args

class Runner:
    def __init__(self, args):
        self.args = args
        self.model_output_root: Path = args.output_root \
            / f'{args.targets}' \
            / f'{args.aug}_aug' \
            / f'{args.pretrained_name}' \
            / 'bs{batch_size},lr{lr},{weight_strategy},{sample_size},{sample_slices}'.format(**args.__dict__)
        self.model_output_root.mkdir(parents=True, exist_ok=True)
        self.reporters: Dict[str, Reporter] = {
            test_name: Reporter(self.model_output_root / test_name, self.args.target_names)
            for test_name in ['cross-val']
        }

        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt=logging.Formatter.default_time_format,
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.model_output_root / 'train.log', 'w'),
            ],
        )
        if self.args.train:
            self.set_determinism()
        self.folds = load_folds(self.args)

    def run(self):
        for val_id in range(len(self.folds)):
            self.run_fold(val_id)

        for reporter in self.reporters.values():
            reporter.report()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        logging.info(f'set random seed of {seed}\n')

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        modalities = list(ScanProtocol)
        val_fold = self.folds[val_id]
        val_transforms = Compose([
            Resized(
                modalities,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(modalities, 'img'),
            SelectItemsd(['img', 'label']),
            ToTensorDeviced('img', self.args.device),
        ])
        val_set = MultimodalDataset(val_fold, val_transforms, len(self.args.target_names))
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        aug: List[Transform] = []
        modalities = list(ScanProtocol)
        if self.args.aug == 'weak':
            aug = [
                RandFlipd(modalities, prob=0.5, spatial_axis=0),
                RandRotate90d(modalities, prob=0.5),
            ]
        elif self.args.aug == 'strong':
            raise NotImplementedError
        train_transforms = Compose(aug + [
            Resized(
                modalities,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(modalities, 'img'),
            SelectItemsd(['img', 'label']),
            ToTensorDeviced('img', self.args.device),
        ])
        train_set = MultimodalDataset(train_folds, train_transforms, len(self.args.target_names))
        return train_set, self.prepare_val_fold(val_id)

    def get_grouped_parameters(self, model: nn.Module):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['fc']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,
             'lr': self.args.lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.lr},
        ]
        return grouped_parameters

    def run_fold(self, val_id: int):
        output_path = self.model_output_root / f'checkpoint-{val_id}.pth.tar'
        train_set, val_set = self.prepare_fold(val_id)
        if self.args.train:
            logging.info(f'run cross validation on fold {val_id}')
            model = load_pretrained_model(self.args.pretrained_name)
            model.setup_fc()
            if self.args.device == 'cuda' and torch.cuda.device_count() > 1:
                logging.info(f'''using {torch.cuda.device_count()} GPUs\n''')
                model = nn.DataParallel(model)
            model = model.to(self.args.device)
            train_loader = DataLoader(PairDataset(train_set), batch_size=self.args.batch_size, shuffle=True)
            loss_fn = MSELoss()
            optimizer = optim.AdaBelief(self.get_grouped_parameters(model))
            best_acc = 0
            patience = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for x, y, r in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    x = x.to(self.args.device)
                    y = y.to(self.args.device)
                    r = r.float().to(self.args.device)
                    r_ = model.forward(x, y)
                    loss = loss_fn(r_, r)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_acc = self.run_eval(model.module if isinstance(model, DataParallel) else model, train_set, val_set)
                logging.info(f'cur acc:  {val_acc}')
                logging.info(f'best acc: {best_acc}')

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(
                        model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                        output_path,
                    )
                    logging.info(f'model updated, saved to {output_path}\n')
                    patience = 0
                else:
                    patience += 1
                    logging.info(f'patience {patience}/{self.args.patience}\n')
                    if patience == self.args.patience:
                        logging.info('run out of patience\n')
                        break

        model = generate_model(self.args.pretrained_name, len(self.args.target_names)).to(self.args.device)
        model.setup_fc()
        # model.load_state_dict(torch.load(output_path))
        model = model.to(self.args.device)

        self.run_eval(model, train_set, val_set, 'cross-val')

    def run_eval(self, model: Siamese, ref_set: MultimodalDataset, eval_set: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()

        with torch.no_grad():
            ref = []
            labels = []
            for data in DataLoader(ref_set, batch_size=1, shuffle=False):
                ref.append(model.feature(data['img']))
                labels.append(data['label'])
            ref = torch.cat(ref).to(self.args.device)

            acc = 0
            for data in tqdm(DataLoader(eval_set, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].item()
                x = model.feature(img)
                x = x.repeat(len(ref_set), 1)
                r = model.relation(ref, x).view(-1)
                pred = labels[r.argmax().item()]
                acc += pred == label
                if test_name:
                    self.reporters[test_name].append_pred(pred, label)

        return acc / len(eval_set)

if __name__ == '__main__':
    runner = Runner(get_args())
    runner.run()
