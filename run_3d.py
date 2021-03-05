import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from torch import nn
from torch.nn import DataParallel, CrossEntropyLoss, functional as F
from torch.optim import Adam
from tqdm import tqdm

from utils.data_3d import prepare_data_3d, MultimodalDataset


def load_pretrained_resnet(pretrained_name, log=True):
    from resnet_3d.model import load_pretrained_model, pretrained_root
    config = json.load(open(pretrained_root / 'config.json'))[pretrained_name]
    from resnet_3d.models import resnet
    model = resnet.generate_model(
        config['model_depth'],
        n_classes=config['n_pretrain_classes'],
        n_input_channels=3,
    )
    load_pretrained_model(model, pretrained_root / f'{pretrained_name}.pth', 'resnet', 4, log)
    return model

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal'], default='equal')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_root', type=Path, default='output_3d')
    parser.add_argument(
        '--pretrained_name',
        default=None,
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
    args = parser.parse_args()
    return args

def fix_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def run_train_val(datasets: Dict[str, MultimodalDataset], output_path: Path, args) -> Tuple[float, List[float]]:
    if args.train:
        model = load_pretrained_resnet(args.pretrained_name)
        if args.device == 'cuda' and torch.cuda.device_count() > 1:
            logging.info(f'''Let's use {torch.cuda.device_count()} GPUs!\n''')
            model = nn.DataParallel(model)

        model = model.to(args.device)
        train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
        loss_fn = CrossEntropyLoss(weight=datasets['train'].get_weight(args.weight_strategy).to(args.device))
        optimizer = Adam(model.parameters(), lr=args.lr)
        best_loss = float('inf')
        patience = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for inputs, labels in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                logits = model.forward(inputs)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss, aucs = run_eval(model, datasets['val'])
            logging.info(f'epoch {epoch}')
            logging.info(f'current val loss: {val_loss}')
            logging.info(f'best val loss:    {best_loss}', )
            logging.info(f'AUCs: {" ".join(map(str, aucs))}')
            logging.info(f'average AUC: {np.mean(aucs)}', )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                    output_path,
                )
                logging.info(f'model updated, saved to {output_path}\n')
                patience = 0
            else:
                patience += 1
                logging.info(f'patience {patience}/{args.patience}\n')
                if patience == args.patience:
                    logging.info('run out of patience\n')
                    break

    model = load_pretrained_resnet(args.pretrained_name, log=False)
    model.load_state_dict(torch.load(output_path))
    model = model.to(args.device)
    return run_eval(model, datasets['val'])

def run_eval(model, eval_dataset) -> Tuple[float, List[float]]:
    from sklearn.metrics import roc_curve, auc
    model.eval()
    y_true = []
    y_score = []
    eval_loss = 0
    with torch.no_grad():
        for input, label in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
            input = input.to(args.device)
            label = label.to(args.device)
            y_true.append(label.item())
            logit = model.forward(input)
            y_score.append(F.softmax(logit[0], dim=0).cpu().numpy())
            loss = F.cross_entropy(logit, label)
            eval_loss += loss.item()
    y_true = np.eye(4)[y_true]
    y_score = np.array(y_score)
    aucs = []
    for i in range(4):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
        aucs.append(auc(fpr, tpr))
    return eval_loss, aucs

def get_folds():
    from monai.transforms import LoadImage
    folds = json.load(open('folds.json'))
    bar = tqdm(total=sum(len(fold) for fold in folds), ncols=80, desc='loading data')
    loaded_folds = []
    for fold in folds:
        loaded_fold = []
        for info in fold:
            loaded_fold.append((
                Compose(LoadImage(image_only=True))(info['scans'][:3]),
                {
                    'WNT': 0,
                    'SHH': 1,
                    'G3': 2,
                    'G4': 3,
                }[info['subgroup']]
            ))
            bar.update()
        loaded_folds.append(loaded_fold)
    return loaded_folds

if __name__ == '__main__':
    args = get_args()
    fix_state(args.seed)

    folds = get_folds()
    folds_aucs = []

    model_output_root = args.output_root / '{pretrained_name}_lr{lr}_{weight_strategy}_{sample_size}_{sample_slices}'.format(**args.__dict__)
    model_output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(model_output_root / '{}.log'.format('train' if args.train else 'val'), 'w'),
            logging.StreamHandler(),
        ],
    )
    print(model_output_root)
    for val_id in range(len(folds)):
        logging.info(f'validation on fold {val_id}')
        datasets = prepare_data_3d(folds, val_id, args)
        val_loss, aucs = run_train_val(datasets, model_output_root / f'checkpoint-{val_id}.pth.tar', args)
        folds_aucs.append(aucs)
        logging.info(f'fold loss: {val_loss}')
        logging.info(f"fold AUCs: {' '.join(map(str, aucs))}")
        logging.info(f'fold average AUC: {np.mean(aucs)}\n')
    avg_aucs = np.array(folds_aucs).mean(axis=1)
    logging.info(f"average AUCs of all classes:\n {' '.join(map(str, avg_aucs.tolist()))}")
