import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ImageRecognitionModel
from parser import parser
from utils.data import load_data, ImageRecognitionDataset, targets


if __name__ == '__main__':
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=23333)
    parser.add_argument('--ortns', type=str, nargs='*', choices=['back', 'up', 'left'], default=['up'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--targets', nargs='*', type=str, default=['subgroup2'], choices=list(targets.keys()))
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

    for ortn in args.ortns:
        if args.visualize:
            datasets = load_data('data', norm=False)
            test_set = datasets['val'][ortn]['exists']
            model = ImageRecognitionModel(ortn, args, 'exists', load_weights=True)
            model.eval()
            for img, exists in tqdm(test_set):
                if exists:
                    input = ImageRecognitionDataset.normalize(img).to(device).unsqueeze(0)
                    input.requires_grad_()
                    logit = model.forward(input)[0]
                    if logit.argmax().item():
                        from utils.gradcam import GradCam, show_cam_on_image
                        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4, target_layer_names=['1'], use_cuda=True)
                        target_index = None
                        mask = grad_cam(input, 1)
                        show_cam_on_image(img.permute(1, 2, 0), mask)
                        continue
        else:
            datasets = load_data('data')
            if args.train:
                for target in args.targets:
                    model = ImageRecognitionModel(ortn, args, target)
                    model.run_train({
                        split: datasets[split][ortn][target]
                        for split in ['train', 'val']
                    })
            if args.test:
                for target in args.targets:
                    model = ImageRecognitionModel(ortn, args, target, load_weights=True)
                    test_set = datasets['val'][ortn][target]
                    model.run_test(test_set)

            # y_true = []
            # y_pred = []
            # y_base_pred = []
            # subgroup_model = ImageRecognitionModel(ortn, args, 'subgroup', load_weights=True)
            # subgroup2_model = ImageRecognitionModel(ortn, args, 'subgroup2', load_weights=True)
            # test_set = datasets['val'][ortn]['subgroup']
            # for patients, inputs, labels in tqdm(DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_set.collate_fn), ncols=80):
            #     inputs = inputs.to(device)
            #     subgroup_logits = subgroup_model.forward(inputs)
            #     subgroup2_logits = subgroup2_model.forward(inputs)
            #     for patient, label, subgroup_logit, subgroup2_logit in zip(patients, labels, subgroup_logits, subgroup2_logits):
            #         label = label.item()
            #         y_true.append(label)
            #
            #         true_base = int(label >= 2) * 2
            #         base_pred = true_base + subgroup_logit[true_base:true_base + 2].argmax().item()
            #         y_base_pred.append(base_pred)
            #
            #         base = subgroup2_logit.argmax().item() * 2
            #         pred = base + subgroup_logit[base:base + 2].argmax().item()
            #         y_pred.append(pred)
            #
            # report_dir = os.path.join('output', model_name, ortn)
            # from sklearn.metrics import classification_report
            # import pandas as pd
            #
            # base_report = classification_report(y_true, y_base_pred, target_names=targets['subgroup'], digits=3, output_dict=True)
            # pd.DataFrame(base_report).transpose().to_csv(os.path.join(report_dir, 'report-base.tsv'), sep='\t')
            # report = classification_report(y_true, y_pred, target_names=targets['subgroup'], digits=3, output_dict=True)
            # pd.DataFrame(report).transpose().to_csv(os.path.join(report_dir, 'report.tsv'), sep='\t')
