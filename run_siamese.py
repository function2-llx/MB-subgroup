import itertools
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import monai
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_optimizer as optim
from monai.data import DataLoader
from monai.transforms import *
from torch import nn
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from siamese import generate_model, load_pretrained_model, Siamese
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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--val_steps', type=int, default=0)
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='equal')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_root', type=Path, default='output_siamese')
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--start_rank', type=int, default=0, help='start rank of current node')
    parser.add_argument('--n_gpu', type=int, default=2)
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
    def __init__(self, args, folds):
        self.args = args
        self.model_output_root: Path = args.model_output_root
        self.model_output_root.mkdir(parents=True, exist_ok=True)

        if dist.get_rank() == 0:
            self.reporters: Dict[str, Reporter] = {
                test_name: Reporter(self.model_output_root / test_name, self.args.target_names)
                for test_name in ['cross-val']
            }
        if self.args.train:
            self.set_determinism()
        self.folds = folds
        self.loss_fn = MSELoss()

    def run(self):
        for val_id in range(len(self.folds)):
            self.run_fold(val_id)

        if dist.get_rank() == 0:
            for reporter in self.reporters.values():
                reporter.report()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        if dist.get_rank() == 0:
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
             'lr': self.args.lr * 10},
        ]
        return grouped_parameters

    def run_fold(self, val_id: int):
        output_path = self.model_output_root / f'checkpoint-{val_id}.pth.tar'
        train_set, val_set = self.prepare_fold(val_id)
        torch.autograd.set_detect_anomaly(True)
        if self.args.train:
            if dist.get_rank() == 0:
                logging.info(f'run cross validation on fold {val_id}')
            model = load_pretrained_model(self.args.pretrained_name)
            model.setup_fc()
            model = model.to(self.args.device)
            model = DistributedDataParallel(model, [self.args.device], broadcast_buffers=False)
            sampler = DistributedSampler(train_set, num_replicas=self.args.world_size, rank=dist.get_rank(), shuffle=True)
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, sampler=sampler, shuffle=False)
            val_steps = self.args.val_steps
            if val_steps == 0:
                val_steps = len(train_loader)
            optimizer = optim.AdaBelief(self.get_grouped_parameters(model))
            best_loss = float('inf')
            # use a list for dist.broadcast_object_list
            patience = [0]
            step = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80, disable=dist.get_rank() != 0):
                    optimizer.zero_grad()
                    cur_features = model.forward(model.module.feature, data['img'])
                    all_features = self.all_gather(cur_features)
                    r_pred = model.forward(
                        model.module.relation,
                        cur_features[:, None, :].repeat(1, all_features.shape[0], 1),
                        all_features[None, :, :].repeat(cur_features.shape[0], 1, 1)
                    )
                    cur_labels = data['label'].to(self.args.device)
                    all_labels = self.all_gather(cur_labels)
                    r_true = torch.eq(
                        cur_labels[:, None].repeat(1, all_labels.shape[0]),
                        all_labels[None, :].repeat(cur_labels.shape[0], 1)
                    ).float()
                    loss = self.loss_fn(r_pred, r_true)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if step % val_steps == 0:
                        if dist.get_rank() == 0:
                            val_loss = self.run_eval(model.module, train_set, val_set)
                            logging.info(f'cur loss:  {val_loss}')
                            logging.info(f'best loss: {best_loss}')

                            if val_loss < best_loss:
                                best_loss = val_loss
                                torch.save(model.module.state_dict(), output_path)
                                logging.info(f'model updated, saved to {output_path}\n')
                                patience[0] = 0
                            else:
                                patience[0] += 1
                                logging.info(f'patience {patience[0]}/{self.args.patience}\n')
                        dist.broadcast_object_list(patience, src=0)
                        if patience[0] == self.args.patience:
                            if dist.get_rank() == 0:
                                logging.info('run out of patience\n')
                            break
                else:
                    continue
                break

        model = generate_model(self.args.pretrained_name, len(self.args.target_names)).to(self.args.device)
        model.setup_fc()
        model.load_state_dict(torch.load(output_path))
        model = model.to(self.args.device)

        if dist.get_rank() == 0:
            self.run_eval(model, train_set, val_set, 'cross-val')

    def all_gather(self, tensor):
        tensors_list = [torch.empty(tensor.shape, dtype=tensor.dtype).to(self.args.device) for _ in range(self.args.world_size)]
        dist.all_gather(tensors_list, tensor)
        return torch.cat(tensors_list)

    def run_eval(self, model: Siamese, ref_set: MultimodalDataset, eval_set: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            ref = []
            labels = []
            for data in tqdm(DataLoader(ref_set, batch_size=1, shuffle=False), ncols=80, desc='calculating ref'):
                ref.append(model.feature(data['img']))
                labels.append(data['label'].item())
            ref = torch.cat(ref).to(self.args.device)
            labels = torch.tensor(labels).to(self.args.device)

            acc = 0
            for data in tqdm(DataLoader(eval_set, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].item()
                x = model.feature(img)
                x = x.repeat(len(ref_set), 1)
                r_pred = model.relation(ref, x).view(-1)
                r_true = torch.eq(labels, label).float()
                eval_loss += self.loss_fn(r_pred, r_true)
                pred = labels[r_pred.argmax().item()].item()
                acc += pred == label
                if test_name:
                    self.reporters[test_name].append_pred(pred, label)
        return eval_loss
        # return acc / len(eval_set)

def main(gpu_id, args, folds):
    args.device = gpu_id
    rank = args.start_rank + gpu_id
    args.model_output_root = args.output_root \
        / f'{args.targets}' \
        / f'{args.aug}_aug' \
        / f'{args.pretrained_name}' \
        / 'bs{batch_size},lr{lr},{weight_strategy},{sample_size},{sample_slices}'.format(**args.__dict__)
    if rank == 0:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt=logging.Formatter.default_time_format,
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(args.model_output_root / 'train.log', 'w'),
            ],
        )
    else:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt=logging.Formatter.default_time_format,
            level=logging.INFO,
        )
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    runner = Runner(args, folds)
    runner.run()

if __name__ == '__main__':
    args = get_args()
    folds = load_folds(args)
    mp.spawn(main, (args, folds), nprocs=args.n_gpu, join=True)
