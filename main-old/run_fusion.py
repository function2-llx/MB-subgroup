import itertools
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import monai
import torch
import torchvision
from monai.data import DataLoader
from monai.transforms import *
from torch import nn
from torch.nn import CrossEntropyLoss, DataParallel, functional as F
from torch_optimizer import AdaBelief
from tqdm import tqdm

from models.fusion import FusionNetwork
from utils.data2d import load_folds
from utils.data import MultimodalDataset
from utils.dicom_utils import ScanProtocol
from utils.report import Reporter

rgb_normalizer = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=29)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pretrained_name', type=str, choices=['resnet'])
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='inv')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_root', type=Path, default='output_2d')
    parser.add_argument('--targets', choices=['all', 'G3G4'], default='G3G4')
    parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='no')
    parser.add_argument('--sample_size',
                        default=224,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_slices',
                        default=24,
                        type=int,
                        help='slices of MRI inputs (probably axial scan), temporal size in terms of videos')
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
            / args.targets \
            / f'{args.aug}_aug' \
            / 'bs{batch_size},lr{lr},{weight_strategy}'.format(**args.__dict__)
        self.model_output_root.mkdir(parents=True, exist_ok=True)
        self.reporters = {
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
        self.folds = [[]]

    def run(self):
        self.folds = load_folds(self.args)
        if self.args.train:
            self.set_determinism()

        for val_id in range(len(self.folds)):
            self.run_fold(val_id)

        for reporter in self.reporters.values():
            reporter.report()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        logging.info(f'set random seed of {seed}\n')

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_transforms = Compose([
            Resized(
                list(ScanProtocol),
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices)
            ),
            ConcatItemsd(list(ScanProtocol), 'img'),
            SelectItemsd(['img', 'label']),
            # normalize for each slice, trick torchvison with T * D * H * W as if T was B,
            Lambdad('img', lambda img: rgb_normalizer(torch.tensor(img.transpose(3, 0, 1, 2)))),
        ])
        val_set = MultimodalDataset(val_fold, val_transforms, len(self.args.target_names))
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        train_transforms: List[Transform] = []
        if self.args.aug == 'weak':
            train_transforms = [
                RandFlip(prob=0.2, spatial_axis=0),
                RandRotate90(prob=0.2),
            ]
        elif self.args.aug == 'strong':
            raise NotImplementedError

        train_transforms: Transform = Compose(train_transforms + [
            Resized(
                list(ScanProtocol),
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices)
            ),
            ConcatItemsd(list(ScanProtocol), 'img'),
            SelectItemsd(['img', 'label']),
            Lambdad('img', lambda img: img.transpose((3, 0, 1, 2))),
            Lambdad('img', lambda img: rgb_normalizer(torch.tensor(img))),
        ])
        train_set = MultimodalDataset(train_folds, train_transforms, len(self.args.target_names))
        return train_set, self.prepare_val_fold(val_id)

    def run_fold(self, val_id: int):
        model_output_path = self.model_output_root / f'checkpoint-{val_id}.pth.tar'
        if self.args.train:
            logging.info(f'run cross validation on fold {val_id}')
            model = FusionNetwork(self.args, n_output=len(self.args.target_names))
            train_set, val_set = self.prepare_fold(val_id)
            if self.args.device == 'cuda' and torch.cuda.device_count() > 1:
                logging.info(f'''using {torch.cuda.device_count()} GPUs\n''')
                model = nn.DataParallel(model)
            model = model.to(self.args.device)
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.args.weight_strategy).to(self.args.device))
            optimizer = AdaBelief(model.parameters(), lr=self.args.lr)
            best_loss = float('inf')
            patience = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    # inputs = inputs.to(self.args.device)
                    imgs = data['img'].to(self.args.device)
                    labels = data['label'].to(self.args.device)
                    logits = model.forward(imgs)
                    loss = loss_fn(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                val_loss = self.run_eval(model, val_set)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(
                        model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                        model_output_path,
                    )
                    logging.info(f'model updated, saved to {model_output_path}\n')
                    patience = 0
                else:
                    patience += 1
                    logging.info(f'patience {patience}/{self.args.patience}\n')
                    if patience == self.args.patience:
                        logging.info('run out of patience\n')
                        break
        else:
            val_set = self.prepare_val_fold(val_id)

        model = FusionNetwork(self.args, len(self.args.target_names)).to(self.args.device)
        model.load_state_dict(torch.load(model_output_path))
        self.run_eval(model, val_set, 'cross-val')

    def run_eval(self, model: nn.Module, eval_dataset: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].to(self.args.device)
                logit = model.forward(img)
                if test_name:
                    self.reporters[test_name].append(logit[0], label.item())

                loss = F.cross_entropy(logit, label)
                eval_loss += loss.item()
        return eval_loss

if __name__ == '__main__':
    # print(resnet18().feature(torch.randn(1, 3, 224, 224)).shape)
    runner = Runner(get_args())
    runner.run()
