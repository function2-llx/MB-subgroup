from collections.abc import Mapping, Sequence
from pathlib import Path

import cytoolz
from jsonargparse import ArgumentParser
import torch
from torch import nn
from ruamel.yaml import YAML

from luolib.lightning import LightningModule

from mbs.datamodule import MBSegDataModule
from .model import MBSegModel
from .utils import SlidingWindowInferenceConf

__all__ = [
    'MBSegPredictor',
]

yaml = YAML()

class MBSegPredictor(LightningModule):
    datamodule: MBSegDataModule

    @staticmethod
    def load_model(parser: ArgumentParser, run_dir: Path, config_filename: str = 'conf.yaml') -> tuple[MBSegModel, int]:
        conf = parser.parse_object({'model': yaml.load(run_dir / config_filename)['model']})
        model: MBSegModel = parser.instantiate_classes(conf).model
        model.load_state_dict(torch.load(run_dir / 'checkpoint' / 'last.ckpt')['state_dict'])
        # run_dir be like: .../fold-*/seed-*/run-*
        fold_id = int(run_dir.parts[-3][5:])
        return model, fold_id

    def __init__(
        self,
        run_dirs: list[Path],
        sw: SlidingWindowInferenceConf,
        tta: bool = True,
        prob_th: float = 0.5,
        output_root: Path | None = None,
        use_all_folds: bool = False,
        overwrite: bool = False,
    ):
        """
        Args:
            overwrite: whether to overwrite existing probability files
            use_all_folds: for prediction on training sets, use models trained from all folds
        """
        super().__init__()
        parser = ArgumentParser()
        parser.add_subclass_arguments(MBSegModel, 'model')
        models = {}
        for run_dir in run_dirs:
            model, fold_id = self.load_model(parser, run_dir)
            model.sw_conf = sw
            # https://github.com/pytorch/pytorch/issues/11714
            # I won't use dict.setdefault as it will create a new instance every time
            if fold_id not in models:
                models[fold_id] = nn.ModuleList()
            models[fold_id].append(model)
        self.models = models
        self.tta = tta
        assert output_root is not None
        conf_str = f'{sw.window_size}+{sw.overlap}+{sw.blend_mode}'
        if tta:
            conf_str += '+tta'
        self.output_dir = output_root / conf_str
        self.prob_th = prob_th
        self.use_all_folds = use_all_folds
        self.overwrite = overwrite

        # self.post_transform = monai.transforms.Compose([
        #     monai.transforms.KeepLargestConnectedComponent(is_onehot=True),
        # ])

    @property
    def models(self) -> dict[int, nn.ModuleList]:
        return {
            int(fold_id): model
            for fold_id, model in self._models.items()
        }

    @models.setter
    def models(self, models: dict[int, nn.ModuleList]):
        self._models = nn.ModuleDict({
            str(i): model
            for i, model in models.items()
        })

    def get_models(self, case: str) -> Sequence[MBSegModel]:
        fold_id = self.datamodule.case_split[case]
        if fold_id == 'test' or self.use_all_folds:
            return [*cytoolz.concat(self.models.values())]
        else:
            return self.models[fold_id]

    def infer(self, models: Sequence[MBSegModel], img: torch.Tensor) -> torch.Tensor:
        img = img[None]
        ensemble_prob = None
        for model in models:
            if self.tta:
                logits = model.tta_sw_infer(img, softmax=False)
            else:
                logits = model.sw_infer(img, softmax=False)
            prob = logits.sigmoid()
            if ensemble_prob is None:
                ensemble_prob = prob
            else:
                ensemble_prob += prob
        ensemble_prob /= len(models)
        return ensemble_prob[0]

    def predict_step(self, batch, *args, **kwargs):
        case: str = batch['case'][0]
        img = batch['img'][0]
        output_dir = self.output_dir / case
        output_dir.mkdir(parents=True, exist_ok=True)

        if (prob_save_path := output_dir / 'prob.pt').exists() and not self.overwrite:
            prob = torch.load(prob_save_path, map_location=self.device)
        else:
            prob = self.infer(self.get_models(case), img)
            prob_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prob, prob_save_path)

        pred = prob > self.prob_th
        pred_save_dir = output_dir / f'th:{self.prob_th}'
        pred_save_dir.mkdir(exist_ok=True)
        for i, name in zip(range(pred.shape[0]), ['ST', 'AT']):
            torch.save(pred[i], pred_save_dir / f'{name}.pt')
