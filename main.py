import random

import numpy as np
import torch
from tqdm import tqdm

from model import ImageRecognitionModel
from parser import parser
from utils.data import load_data, ImageRecognitionDataset

if __name__ == '__main__':
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=2e-5)
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

    for ortn in args.ortns:
        if args.visualize:
            datasets = load_data('data', norm=False)
            test_set = datasets['val'][ortn]['exists']
            model = ImageRecognitionModel(ortn, args, 'exists', ['no', 'yes'], load_weights=True)
            model.eval()
            for img, exists in tqdm(test_set):
                if exists:
                    input = ImageRecognitionDataset.normalize(img).to(device).unsqueeze(0)
                    input.requires_grad_()
                    logit = model.forward(input)[0]
                    if logit.argmax().item():
                        from utils.gradcam import GradCam, show_cam_on_image
                        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4, target_layer_names=["1"], use_cuda=True)
                        target_index = None
                        mask = grad_cam(input, 1)
                        show_cam_on_image(img.permute(1, 2, 0), mask)
                        continue
        else:
            datasets = load_data('data')
            models = {
                target: ImageRecognitionModel(ortn, args, target, target_names, load_weights=args.test)
                for target, target_names in [
                    ('exists', ['no', 'yes']),
                    ('subgroup', ['WNT', 'SHH', 'G3', 'G4']),
                ]
            }
            if args.test:
                for model in models.values():
                    model.run_test(datasets)
            else:
                for model in models.values():
                    model.run_train(datasets)

