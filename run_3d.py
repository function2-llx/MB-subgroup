import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import monai
import numpy as np
import torch
from monai.data import DataLoader
from torch import nn
from torch.nn import DataParallel, CrossEntropyLoss, functional as F
from torch.optim import Adam
from tqdm import tqdm

from utils.data_3d import prepare_data_3d, MultimodalDataset, get_folds


def load_pretrained_resnet(pretrained_name, logger: Optional[logging.Logger] = None):
    from resnet_3d.model import load_pretrained_model, pretrained_root
    config = json.load(open(pretrained_root / 'config.json'))[pretrained_name]
    from resnet_3d.models import resnet
    model = resnet.generate_model(
        config['model_depth'],
        n_classes=config['n_pretrain_classes'],
        n_input_channels=3,
    )
    load_pretrained_model(model, pretrained_root / f'{pretrained_name}.pth', 'resnet', 4, logger)
    return model

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--train', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal'], default='equal')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_root', type=Path, default='output_3d')
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
    parser.add_argument('--aug', choices=['no', 'weak', 'strong'])
    args = parser.parse_args()
    return args

def run_train_val(datasets: Dict[str, MultimodalDataset], output_path: Path, args) -> List[float]:
    train_logger = logging.getLogger('train')
    val_logger = logging.getLogger('val')
    model = load_pretrained_resnet(args.pretrained_name, train_logger)
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        train_logger.info(f'''Let's use {torch.cuda.device_count()} GPUs!\n''')
        model = nn.DataParallel(model)

    model = model.to(args.device)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    loss_fn = CrossEntropyLoss(weight=datasets['train'].get_weight(args.weight_strategy).to(args.device))
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')
    # best_auc = 0
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

        val_loss, aucs = run_eval(model, datasets['val'], args, train_logger)
        # mean_auc = np.mean(aucs)
        # train_logger.info(f'best auc: {best_auc}')
        train_logger.info(f'best loss: {best_loss}')

        if val_loss < best_loss:
        # if mean_auc > best_auc:
            best_loss = val_loss
            # best_auc = mean_auc
            torch.save(
                model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                output_path,
            )
            train_logger.info(f'model updated, saved to {output_path}\n')
            patience = 0
        else:
            patience += 1
            train_logger.info(f'patience {patience}/{args.patience}\n')
            if patience == args.patience:
                train_logger.info('run out of patience\n')
                break

    model = load_pretrained_resnet(args.pretrained_name)
    model.load_state_dict(torch.load(output_path))
    model = model.to(args.device)
    val_loss, aucs = run_eval(model, datasets['val'], args, val_logger)
    return aucs

def run_eval(model, eval_dataset, args, logger: Optional[logging.Logger] = None) -> Tuple[float, List[float]]:
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
    if logger is not None:
        logger.info(f"AUCs: {' '.join(map(str, aucs))}")
        logger.info(f'average AUC: {np.mean(aucs)}')
        logger.info(f'loss: {eval_loss}')
    return eval_loss, aucs

def can_skip(model_output_root: Path):
    log_path = model_output_root / 'train.log'
    if not log_path.exists():
        return
    lines = log_path.open().readlines()
    return len(lines) >= 2 and 'average AUCs of all classes' in lines[-2]

def set_logging(model_output_root: Path):
    import logging
    from logging import Formatter
    formatter = Formatter('%(asctime)s [%(levelname)s] %(message)s', Formatter.default_time_format)
    for split in ['train', 'val']:
        logger = logging.getLogger(split)
        logger.parent = None
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(model_output_root / f'{split}.log', 'w'))
        if split == 'train':
            logger.addHandler(logging.StreamHandler())
        for handler in logger.handlers:
            handler.setFormatter(formatter)

def set_determinism(seed: int, logger: logging.Logger):
    monai.utils.set_determinism(seed)
    logger.info(f'set random seed of {seed}\n')

def main(args):
    model_output_root: Path = args.output_root \
        / f'{args.aug}_aug' \
        / f'{args.pretrained_name}' \
        / 'bs{batch_size},lr{lr},{weight_strategy},{sample_size},{sample_slices}'.format(**args.__dict__)
    if not args.force_retrain:
        if can_skip(model_output_root):
            print('skip')
            return

    model_output_root.mkdir(parents=True, exist_ok=True)
    set_logging(model_output_root)
    train_logger = logging.getLogger('train')
    val_logger = logging.getLogger('val')
    val_logger.parent = train_logger

    set_determinism(args.seed, train_logger)

    folds = get_folds()
    folds_aucs = []
    for val_id in range(len(folds)):
        val_logger.info(f'\nrun cross validation on fold {val_id}')
        datasets = prepare_data_3d(folds, val_id, args)
        aucs = run_train_val(datasets, model_output_root / f'checkpoint-{val_id}.pth.tar', args)
        folds_aucs.append(aucs)

    avg_aucs = np.array(folds_aucs).mean(axis=1)
    val_logger.info(f"average AUCs of all classes:\n {' '.join(map(str, avg_aucs.tolist()))}")

if __name__ == '__main__':
    args = get_args()
    main(args)
