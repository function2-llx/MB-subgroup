from __future__ import annotations

import itertools
from collections.abc import Sequence, Mapping

import pytorch_lightning as pl
import torch
from torch import nn

from luolib.conf import parse_exp_conf
from luolib.utils import DataKey
from mbs.utils.enums import SegClass
import monai
from monai.data import MetaTensor

from mbs.conf import MBSegPredConf
from mbs.datamodule import MBSegDataModule, load_split
from mbs.models import MBSegModel

SEG_PROB_FILENAME = 'seg-prob.pt'

class MBSegPredictor(pl.LightningModule):
    def __init__(self, conf: MBSegPredConf):
        super().__init__()
        self.conf = conf
        self.split = load_split()
        self.models: Mapping[str, MBSegModel] = nn.ModuleDict({
            # pytorch forces model name to be string
            f'{seed} {fold_id}': MBSegModel.load_from_checkpoint(
                self.conf.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt',
                strict=True,
                conf=conf,
            )
            for seed in conf.p_seeds for fold_id in range(conf.num_folds)
        })
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=True),
        ])

    def predict_step(self, batch: dict[str, ...], *args, **kwargs):
        conf = self.conf
        case: str = batch[DataKey.CASE][0]
        split = self.split[case]
        if split not in ['train', 'test']:
            assert split in range(conf.num_folds)
        img: MetaTensor = batch[DataKey.IMG]

        pred_prob = None
        if (prob_save_path := MBSegPredConf.get_case_save_dir(conf, case) / SEG_PROB_FILENAME).exists() and not conf.overwrite:
            pred_prob = torch.load(prob_save_path, map_location=self.device)
        else:
            cnt = 0
            for model_name, model in self.models.items():
                seed, fold_id = map(int, model_name.split())
                if split == fold_id:
                    continue
                cnt += 1
                prob = model.infer(img, progress=False).sigmoid()
                if pred_prob is None:
                    pred_prob = prob
                else:
                    pred_prob += prob
            pred_prob /= cnt
            prob_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pred_prob, prob_save_path)

        pred = pred_prob[0] > conf.th
        for i, seg_class in enumerate(SegClass):
            class_pred = pred[i]
            save_path = MBSegPredConf.get_save_path(conf, case, seg_class, False)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(class_pred, save_path)
            class_pred_post = self.post_transform(class_pred[None])[0]
            post_save_path = MBSegPredConf.get_save_path(conf, case, seg_class, True)
            post_save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(class_pred_post, post_save_path)

class MBSegPredictionDataModule(MBSegDataModule):
    conf: MBSegPredConf

    def predict_data(self) -> Sequence:
        conf = self.conf
        all_data = list(itertools.chain(*self.split_cohort.values()))
        return all_data[conf.l:conf.r]

def main():
    torch.set_float32_matmul_precision('high')
    conf = parse_exp_conf(MBSegPredConf)
    MBSegPredConf.default_pred_output_dir(conf)
    conf.log_dir = conf.p_output_dir
    conf.p_output_dir.mkdir(exist_ok=True, parents=True)
    MBSegPredConf.save_conf_as_file(conf)
    predictor = MBSegPredictor(conf)
    trainer = pl.Trainer(
        logger=False,
        num_nodes=1,
        accelerator='gpu',
        devices=1,
        precision=conf.precision,
        benchmark=True,
    )
    datamodule = MBSegPredictionDataModule(conf)
    trainer.predict(predictor, datamodule=datamodule)

if __name__ == '__main__':
    main()
