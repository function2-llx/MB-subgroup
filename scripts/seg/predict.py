import itertools
from collections.abc import Sequence, Mapping

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch import nn

from luolib.conf import parse_cli
from luolib.models import SegInferer
from luolib.utils import DataKey
import monai
from monai.data import MetaTensor

from mbs.conf import MBSegPredConf
from mbs.datamodule import MBSegDataModule, load_split, MBM2FDataModule
from mbs.models import MBSegModel, MBM2FModel

SEG_PROB_FILENAME = 'seg-prob.pt'

class MBSegPredictor(pl.LightningModule):
    @property
    def exp_conf(self):
        return self.conf.exp_conf

    def __init__(self, conf: MBSegPredConf):
        super().__init__()
        self.conf = conf
        assert conf.inferer_cls in [MBSegModel.__name__, MBM2FModel.__name__]
        import mbs.models
        inferer_cls: type[SegInferer] = getattr(mbs.models, conf.inferer_cls)
        self.split = load_split()
        self.models: Mapping[str, SegInferer] = nn.ModuleDict({
            # pytorch forces model name to be string
            f'{seed} {fold_id}': inferer_cls.load_from_checkpoint(
                self.exp_conf.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt',
                strict=True,
                conf=self.exp_conf,
            )
            for seed in conf.p_seeds for fold_id in range(conf.exp_conf.num_folds)
        })
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=True),
        ])

    def predict_step(self, batch: dict[str, ...], *args, **kwargs):
        conf = self.conf
        case: str = batch[DataKey.CASE][0]
        split = self.split[case]
        if split not in ['train', 'test']:
            assert split in range(self.exp_conf.num_folds)
        img: MetaTensor = batch[DataKey.IMG]

        pred_prob = None
        if (prob_save_path := MBSegPredConf.get_case_save_dir(conf, case) / SEG_PROB_FILENAME).exists() and not conf.overwrite:
            pred_prob = torch.load(prob_save_path, map_location=self.device)
        else:
            cnt = 0
            for model_name, model in self.models.items():
                seed, fold_id = map(int, model_name.split())
                if not conf.all_folds and split == fold_id:
                    continue
                cnt += 1
                prob = model.infer(img, progress=False)[0]
                if model.conf.output_logit:
                    prob = prob.sigmoid()
                if pred_prob is None:
                    pred_prob = prob
                else:
                    pred_prob += prob
            pred_prob /= cnt
            prob_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pred_prob, prob_save_path)

        pred = pred_prob > conf.th
        for i, seg_class in enumerate(conf.exp_conf.seg_classes):
            class_pred = pred[i]
            for do_post in [False, True]:
                save_path = MBSegPredConf.get_save_path(conf, case, seg_class, do_post)
                save_path.parent.mkdir(exist_ok=True, parents=True)
                if do_post:
                    class_pred = self.post_transform(class_pred[None])[0]
                torch.save(class_pred, save_path)

class MBSegPredDataModule(pl.LightningDataModule):
    def __init__(self, conf: MBSegPredConf):
        super().__init__()
        self.conf = conf
        assert conf.datamodule_cls in [MBSegDataModule.__name__, MBM2FDataModule.__name__]
        import mbs.datamodule
        datamodule_cls = getattr(mbs.datamodule, conf.datamodule_cls)
        self.inner: MBSegDataModule = datamodule_cls(conf.exp_conf)
        self.inner.predict_data = self.predict_data

    def predict_data(self) -> Sequence:
        conf = self.conf
        all_data = list(itertools.chain(*self.inner.split_cohort.values()))
        return all_data[conf.l:conf.r]

    def predict_dataloader(self):
        return self.inner.predict_dataloader()

def main():
    torch.set_float32_matmul_precision('high')
    conf, _ = parse_cli(MBSegPredConf)
    MBSegPredConf.load_exp_conf(conf)
    MBSegPredConf.default_pred_output_dir(conf)
    conf.p_output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(conf, conf.p_output_dir / 'pred-conf.yaml')
    predictor = MBSegPredictor(conf)
    trainer = pl.Trainer(
        logger=False,
        num_nodes=1,
        accelerator='gpu',
        devices=1,
        precision=conf.exp_conf.precision,
        benchmark=True,
    )
    datamodule = MBSegPredDataModule(conf)
    trainer.predict(predictor, datamodule=datamodule)

if __name__ == '__main__':
    main()
