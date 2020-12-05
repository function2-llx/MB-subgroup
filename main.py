import json
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from captum.attr import IntegratedGradients, visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18 as resnet
from torchvision import transforms
from tqdm import tqdm

from utils import load_data

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test', action='store_true')
parser.add_argument('--seed', type=int, default=23333)
parser.add_argument('--ortns', type=str, nargs='*')

args = parser.parse_args()
if not args.ortns:
    args.ortns = ['back', 'left', 'up']
model_name = 'lr={lr},bs={batch_size},'.format(**args.__dict__)
model_name += ','.join(args.ortns)
print(model_name)

device = torch.device(args.device)


def run_epoch(epoch, model, data_loaders, loss_fn, optimizer):
    results = {}
    for split, data_loader in data_loaders.items():
        training = split == 'train'
        model.train(training)
        tot_loss = 0
        acc = {k: 0 for k in ['all', 'back', 'left', 'up']}
        tot = {k: 0 for k in ['all', 'back', 'left', 'up']}
        with torch.set_grad_enabled(training):
            for inputs, ortns, labels in tqdm(data_loader, ncols=80, desc=f'{split} epoch {epoch}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if training:
                    optimizer.zero_grad()
                output = model.forward(inputs)
                preds = output.argmax(dim=1)
                loss = loss_fn(output, labels)
                tot_loss += loss.item()
                for ortn, label, pred in zip(ortns, labels, preds):
                    flag = (pred == label).item()
                    for k in ['all', ortn]:
                        tot[k] += 1
                        acc[k] += flag
                if training:
                    loss.backward()
                    optimizer.step()
        results[split] = {
            'loss': tot_loss,
            'acc': {k: v * 100 / tot[k] for k, v in acc.items() if k == 'all' or k in args.ortns}
        }
    return results


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = resnet(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 4).to(device)
    output_dir = f'output/{model_name}'
    if not args.test:
        datasets = load_data(args.ortns)
        data_loaders = {split: DataLoader(datasets[split], args.batch_size, split == 'train') for split in ['train', 'val']}
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.lr)
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{model_name}')
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            results = run_epoch(epoch, model, data_loaders, loss_fn, optimizer)
            for split, result in results.items():
                writer.add_scalar(f'{split}/loss', result['loss'], epoch)
                for ortn, acc in result['acc'].items():
                    writer.add_scalar(f'{split}/acc/{ortn}', acc, epoch)
            val_acc = results['val']['acc']['all']
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint-train.pth.tar'))
            print(json.dumps(results, indent=4, ensure_ascii=False))
            print('best val acc', best_acc)
            writer.flush()
        shutil.copy(
            os.path.join(output_dir, 'checkpoint-train.pth.tar'),
            os.path.join(output_dir, 'checkpoint.pth.tar'),
        )
    else:
        datasets = load_data(args.ortns, norm=False)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pth.tar')))
        model.eval()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        for img, ortn, label in tqdm(datasets['val'], ncols=80):
            input = normalize(img).to(device).unsqueeze(0)
            output = model.forward(input)
            pred = output.argmax(dim=1)
            integrated_gradients = IntegratedGradients(model)
            attributions_ig = integrated_gradients.attribute(input, target=pred, n_steps=200)
            default_cmap = LinearSegmentedColormap.from_list(
                'custom blue',
                [
                    (0, '#ffffff'),
                    (0.25, '#000000'),
                    (1, '#000000')
                ],
                N=256
            )

            _ = viz.visualize_image_attr_multiple(
                np.transpose(
                    attributions_ig.squeeze().cpu().detach().numpy(),
                    (1, 2, 0)
                ),
                np.transpose(
                    img.detach().numpy(),
                    (1, 2, 0),
                ),
                methods=['original_image', 'heat_map'],
                signs=['all', 'positive'],
                show_colorbar=True,
                cmap=default_cmap,
                # outlier_perc=1,
            )
            a = 1
