from __future__ import annotations

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
from mbs.utils import SEG_PROB_FILENAME

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
        img: MetaTensor = batch[DataKey.IMG]
        pred_prob = None
        case_dir = conf.p_output_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)

        prob_save_path = self.conf.p_output_dir / case / SEG_PROB_FILENAME
        if prob_save_path.exists() and not conf.overwrite:
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

        pred = (pred_prob[0] > conf.th).long()
        output_dir = case_dir / f'th{conf.th}'
        output_dir.mkdir(exist_ok=True)
        post_output_dir = case_dir / f'th{conf.th}-post'
        post_output_dir.mkdir(exist_ok=True)
        for i, seg_class in enumerate(SegClass):
            class_pred = pred[i]
            torch.save(class_pred, output_dir / f'{seg_class}.pt')
            class_pred_post = self.post_transform(class_pred[None])[0]
            torch.save(class_pred_post, post_output_dir / f'{seg_class}.pt')

class MBSegPredictionDataModule(MBSegDataModule):
    conf: MBSegPredConf

    def predict_data(self) -> Sequence:
        conf = self.conf
        return self.all_data()[conf.l:conf.r]

def main():
    torch.set_float32_matmul_precision('high')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    conf = parse_exp_conf(MBSegPredConf)
    if conf.p_output_dir is None:
        suffix = f'sw{conf.sw_overlap}'
        if conf.do_tta:
            suffix += '+tta'
        conf.p_output_dir = conf.output_dir / f'predict-{"+".join(map(str, conf.p_seeds))}' / suffix
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
