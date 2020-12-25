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
from utils.data import load_data
from utils.report import ClassificationReporter
from sklearn.metrics import roc_curve, auc, classification_report

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test', action='store_true')
parser.add_argument('--seed', type=int, default=23333)
parser.add_argument('--ortns', type=str, nargs='*', choices=['back', 'up', 'left'], default=['back'])
parser.add_argument('--patience', type=int, default=5)

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
            subtype_true = []
            subtype_pred = []
            subtype_score = []

            for img, target in tqdm(datasets['val'][ortn], ncols=80, desc='testing'):
                subtype = target['subtype']
                exists = target['exists']
                if exists:
                    input = normalize(img).to(device).unsqueeze(0)
                    logit = model.resnet.forward(input)[0]

                    subtype_logit, exists_logit = logit[:4], logit[4:]
                    output = torch.softmax(subtype_logit, dim=0)
                    subtype_true.append(subtype)
                    subtype_pred.append(subtype_logit.argmax().item())
                    subtype_score.append(output.cpu().numpy())
            subtype_true = np.array(subtype_true)
            subtype_pred = np.array(subtype_pred)
            print(classification_report(subtype_true, subtype_pred))
            subtype_true = np.eye(4)[np.array(subtype_true)]
            subtype_score = np.array(subtype_score)
            for i in range(4):
                fpr, tpr, thresholds = roc_curve(subtype_true[:, i], subtype_score[:, i])
                plt.figure()
                lw = 2
                plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{ortn}, {i + 1}')
                plt.legend(loc="lower right")
                plt.show()

                # pred = output.argmax(dim=1)
                integrated_gradients = IntegratedGradients(model)
                attributions_ig = integrated_gradients.attribute(input, target=pred, n_steps=200)
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
