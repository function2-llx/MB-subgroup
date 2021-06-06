import numpy as np
import torch
from monai import transforms as monai_transforms
from monai.data import DataLoader
from torch import nn
from tqdm import tqdm

from runner_base import RunnerBase
from utils.data import MultimodalDataset

class Encoder(RunnerBase):
    def __init__(self, args):
        super().__init__(args)

if __name__ == '__main__':
    from models import generate_model, permute_img
    from finetune import parse_args

    args = parse_args()
    from utils.data.datasets.tiantan import load_cohort
    cohort = load_cohort(args)
    dataset = MultimodalDataset(
        cohort,
        monai_transforms.Compose(RunnerBase.get_inference_transforms(args)),
        len(args.target_names),
    )
    model = generate_model(args, pretrain=True).eval()
    pool = nn.AdaptiveAvgPool3d(1)
    features = {}
    with torch.no_grad():
        for data in tqdm(DataLoader(dataset, batch_size=1), ncols=80, desc='encoding data'):
            feature: torch.Tensor = model.forward(permute_img(data['img'].to(args.device)))['c5']
            feature = pool(feature).view(-1)
            features[data['patient'][0]] = feature.detach().cpu().numpy()
    np.savez_compressed('features.npz', **features)
