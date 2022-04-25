from collections.abc import Sequence
import itertools

import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import StratifiedKFold

from monai.data import DataLoader, Dataset

from cls_unet import ClsUNetArgs
from mbs.utils import Intersection
from mbs.utils.args import TrainingArgs

class CrossValidationDataModule(LightningDataModule):
    def __init__(
        self,
        args: Intersection[TrainingArgs, ClsUNetArgs],
    ):
        super().__init__()
        self.args = args
        self._val_fold_id = -1
        self.folds = self.load_cohort()

    @property
    def val_fold_id(self) -> int:
        return self._val_fold_id

    @val_fold_id.setter
    def val_fold_id(self, value: int):
        self._val_fold_id = value

    def load_cohort(self) -> Sequence[Sequence]:
        cohort = pd.read_excel(self.args.cohort_path).set_index('subject')
        cohort = cohort[cohort[self.args.cls_key].isin(self.args.cls_labels)]
        for subject in cohort.index:
            for modality in self.args.modalities:
                cohort.loc[subject, modality] = self.args.img_dir / subject / f'{modality}.nii.gz'
            for seg_label in self.args.seg_labels:
                cohort.loc[subject, seg_label] = self.args.seg_dir / subject / f'{seg_label}.nii.gz'

        train_cohort = cohort[cohort['split'] == 'train']
        skf = StratifiedKFold(n_splits=self.args.num_folds, shuffle=True, random_state=self.args.seed)
        return [
            train_cohort.iloc[fold_indices, :].to_dict('records')
            for fold_id, (_, fold_indices) in enumerate(skf.split(train_cohort.index, train_cohort['cls']))
        ]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_set = list(itertools.chain(*self.folds[:self.val_fold_id], *self.folds[self.val_fold_id + 1:]))
        # memory issue when using multiple workers: https://github.com/pytorch/pytorch/issues/13246
        return DataLoader(
            Dataset(train_set, self.args.train_transform),
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_set = self.folds[self.val_fold_id]
        return DataLoader(
            Dataset(val_set, self.args.eval_transform),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
        )
