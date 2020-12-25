import json
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from captum.attr import IntegratedGradients, visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from model import Model
from utils.data import load_data
from utils.report import ClassificationReporter

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test', action='store_true')
parser.add_argument('--seed', type=int, default=23333)
parser.add_argument('--ortns', type=str, nargs='*', choices=['back', 'up', 'left'], default=['up'])
parser.add_argument('--patience', type=int, default=3)

args = parser.parse_args()
if not args.ortns:
    args.ortns = ['back', 'up', 'left']

model_name = 'lr={lr},bs={batch_size}'.format(**args.__dict__)
print(model_name)

device = torch.device(args.device)


def map_subtype_idx_list(subtype_idxs):
    subtypes = ['WNT', 'SHH', 'G3', 'G4']
    return [subtypes[idx] for idx in subtype_idxs]


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.test:
        datasets = load_data('data')
        for ortn in args.ortns:
            output_dir = os.path.join('output', model_name, ortn)
            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
            os.makedirs(output_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', model_name, ortn))
            best_loss = float("inf")
            patience = 0
            for epoch in range(1, args.epochs + 1):
                results = model.train_epoch(epoch)
                print(json.dumps(results, indent=4, ensure_ascii=False))
                for split, result in results.items():
                    for k, v in result.items():
                        writer.add_scalar(f'{split}/{k}', v, epoch)
                val_loss = results['val']['loss']
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint-train.pth.tar'))
                    patience = 0
                else:
                    patience += 1
                    print(f'patience {patience}/{args.patience}')
                    if patience >= args.patience:
                        print('run out of patience')
                        break
                writer.flush()
            shutil.copy(
                os.path.join(output_dir, 'checkpoint-train.pth.tar'),
                os.path.join(output_dir, 'checkpoint.pth.tar'),
            )
    else:
        datasets = load_data('data', norm=False)
        subtype_reporter = ClassificationReporter(['WNT', 'SHH', 'G3', 'G4'])
        exists_reporter = ClassificationReporter([False, True])
        for ortn in args.ortns:
            output_dir = os.path.join('output', model_name, ortn)
            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
            model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pth.tar')))
            model.eval()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            for img, target in tqdm(datasets['val'][ortn], ncols=80, desc='testing'):
                exists = target['exists']
                input = normalize(img).to(device).unsqueeze(0)
                logit = model.forward(input)
                for k, v in logit.items():
                    logit[k] = v[0]
                exists_reporter.append(logit['exists'], exists)
                if exists:
                    subtype_reporter.append(logit['subtype'], target['subtype'])

                    integrated_gradients = IntegratedGradients(model)
                    attributions_ig = integrated_gradients.attribute(input, target=4, n_steps=200)
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
                        outlier_perc=1,
                    )
