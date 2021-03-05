import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, DataParallel
from monai.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

from utils.data_3d import prepare_data_3d

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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
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
    print('device:', args.device)
    return args

def fix_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_train_val(datasets: Dict[str, Dataset], output_dir: Path, args) -> Tuple[float, List[float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.train:
        model = load_pretrained_resnet(args.pretrained_name)
        if args.device == 'cuda' and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model = model.to(args.device)
        train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
        optimizer = Adam(model.parameters(), lr=args.lr)
        best_loss = float('inf')
        patience = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for inputs, labels in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80, disable=True):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                logits = model.forward(inputs)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(
                    model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                    'b.pth.tar',
                )
                exit(1)


            val_loss, aucs = run_eval(model, datasets['val'])
            print('current val loss:', val_loss)
            print('best val loss:   ', best_loss)
            print('AUCs:', *aucs)
            print('average AUC:', np.mean(aucs))

            if val_loss < best_loss:
                best_loss = val_loss
                save_path = output_dir / 'checkpoint.pth.tar'
                torch.save(
                    model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                    save_path,
                )
                print(f'model updated, saved to {save_path}')
                patience = 0
            else:
                patience += 1
                print(f'patience {patience}/{args.patience}')
                if patience == args.patience:
                    print('run out of patience')
                    break

    model = load_pretrained_resnet(args.pretrained_name, log=False)
    model.load_state_dict(torch.load(output_dir / 'checkpoint.pth.tar'))
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
    folds = json.load(open('folds.json'))
    folds = list(map(
        lambda fold: list(map(
            lambda info: (info['scans'][:3], {
                'WNT': 0,
                'SHH': 1,
                'G3': 2,
                'G4': 3,
            }[info['subgroup']]), fold,
        )), folds,
    ))
    return folds


if __name__ == '__main__':
    args = get_args()
    fix_state(args.seed)

    folds = get_folds()
    folds_aucs = []
    for val_id in range(len(folds)):
        datasets = prepare_data_3d(folds, val_id, args)
        val_loss, aucs = run_train_val(
            datasets,
            args.output_root / '{pretrained_name}_lr{lr}_{sample_size}_{sample_slices}'.format(**args.__dict__) / str(val_id),
            args,
        )
        folds_aucs.append(aucs)
        print('fold loss:', val_loss)
        print('fold AUCs:', *aucs)
        print('fold average AUC:', np.mean(aucs))
    avg_aucs = np.array(folds_aucs).mean(axis=1)
    print(*avg_aucs.tolist())
