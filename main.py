import json
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from captum.attr import GuidedGradCam, visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report

from model import Model
from parser import parser
from utils.data import load_data
from utils.report import ClassificationReporter

if __name__ == '__main__':
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=23333)
    parser.add_argument('--ortns', type=str, nargs='*', choices=['back', 'up', 'left'], default=['up'])
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    if not args.ortns:
        args.ortns = ['back', 'up', 'left']

    model_name = 'lr={lr},bs={batch_size}'.format(**args.__dict__)
    print(model_name)

    device = torch.device(args.device)
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
            os.makedirs(output_dir, exist_ok=True)

            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
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
        subgroup_reporter = ClassificationReporter(['WNT', 'SHH', 'G3', 'G4'])
        exists_reporter = ClassificationReporter(["no", "yes"])
        for ortn in args.ortns:
            output_dir = os.path.join('output', model_name, ortn)
            report_dir = os.path.join('report', model_name, ortn)
            os.makedirs(report_dir, exist_ok=True)

            model = Model(ortn, datasets['train'][ortn], datasets['val'][ortn], args)
            model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pth.tar')))
            model.eval()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            for img, target in tqdm(datasets['val'][ortn], ncols=80, desc='testing'):
                exists = target['exists']
                input = normalize(img).to(device).unsqueeze(0)
                input.requires_grad_()
                with torch.no_grad():
                    logit = model.forward(input)
                logit = Model.convert_output_to_dict(logit)
                for k, v in logit.items():
                    logit[k] = v[0]
                exists_reporter.append(logit['exists'], exists)
                if exists:
                    subgroup_reporter.append(logit['subgroup'], target['subgroup'])

                    if args.visualize:
                        # Can work with any model, but it assumes that the model has a
                        # feature method, and a classifier method,
                        # as in the VGG models in torchvision.
                        from utils.gradcam import GradCam, show_cam_on_image
                        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4,
                                           target_layer_names=["1"], use_cuda=True)

                        # If None, returns the map for the highest scoring category.
                        # Otherwise, targets the requested index.
                        target_index = None
                        mask = grad_cam(input, 5)

                        show_cam_on_image(img.permute(1, 2, 0), mask)
                        # integrated_gradients = IntegratedGradients(model)
                        # attribution = GuidedGradCam(model, model.resnet.layer4[-1].conv2)
                        # attrs = attribution.attribute(input, target=5)
                        #
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
                        #         attrs.squeeze(0).cpu().detach().numpy(),
                        #         (1, 2, 0)
                        #     ),
                        #     np.transpose(
                        #         img.detach().numpy(),
                        #         (1, 2, 0),
                        #     ),
                        #     titles=[exists, logit['exists'].argmax().item()],
                        #     methods=['original_image', 'heat_map'],
                        #     signs=['all', 'positive'],
                        #     show_colorbar=True,
                        #     cmap=default_cmap,
                        #     outlier_perc=1,
                        # )

            exists_reporter.report().to_csv(os.path.join(report_dir, 'exists.csv'), sep='\t')
            subgroup_reporter.report().to_csv(os.path.join(report_dir, 'subgroup.csv'), sep='\t')
