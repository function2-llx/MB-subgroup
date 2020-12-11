import json
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from captum.attr import IntegratedGradients, visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from model import Model
from utils import load_data
from sklearn.metrics import roc_curve, auc

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=2e-6)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--test', action='store_true')
parser.add_argument('--seed', type=int, default=23333)
parser.add_argument('--ortns', nargs='*', choices=['back', 'up', 'left'])

args = parser.parse_args()
if not args.ortns:
    args.ortns = ['back', 'up', 'left']

model_name = 'lr={lr},bs={batch_size}'.format(**args.__dict__)
print(model_name)

device = torch.device(args.device)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.test:
        datasets = load_data()
        for ortn in ['left', 'up', 'back']:
            output_dir = os.path.join('output', model_name, ortn)
            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
            os.makedirs(output_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', model_name, ortn))
            best_acc = 0
            for epoch in range(1, args.epochs + 1):
                results = model.train_epoch(epoch)
                for split, result in results.items():
                    for k, v in result.items():
                        writer.add_scalar(f'{split}/loss', v, epoch)
                val_acc = results['val']['acc']
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
        datasets = load_data(norm=False)
        for ortn in args.ortns:
            output_dir = os.path.join('output', model_name, ortn)
            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
            model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pth.tar')))
            model.eval()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            y_true = []
            y_score = []
            with torch.no_grad():
                for img, label in tqdm(datasets['val'][ortn], ncols=80, desc='testing'):
                    input = normalize(img).to(device).unsqueeze(0)
                    output = model.resnet.forward(input)[0]
                    y_true.append(label.long().cpu().numpy())
                    y_score.append(output.cpu().numpy())
                y_true = np.array(y_true)
                y_score = np.array(y_score)
                for i in range(4):
                    fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{ortn}, ')
                    plt.legend(loc="lower right")
                    plt.show()

                    # pred = output.argmax(dim=1)
                    # integrated_gradients = IntegratedGradients(model)
                    # attributions_ig = integrated_gradients.attribute(input, target=pred, n_steps=200)
                    # default_cmap = LinearSegmentedColormap.from_list(
                    #     'custom blue',
                    #     [
                    #         (0, '#ffffff'),
                    #         (0.25, '#000000'),
                    #         (1, '#000000')
                    #     ],
                    #     N=256
                    # )
                    #
                    # _ = viz.visualize_image_attr_multiple(
                    #     np.transpose(
                    #         attributions_ig.squeeze().cpu().detach().numpy(),
                    #         (1, 2, 0)
                    #     ),
                    #     np.transpose(
                    #         img.detach().numpy(),
                    #         (1, 2, 0),
                    #     ),
                    #     methods=['original_image', 'heat_map'],
                    #     signs=['all', 'positive'],
                    #     show_colorbar=True,
                    #     cmap=default_cmap,
                    #     # outlier_perc=1,
                    # )
                    # a = 1
