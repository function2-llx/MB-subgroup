from collections.abc import Callable, Sequence
import itertools

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import TrainingArguments

from monai.data import DataLoader, Dataset

class KFoldDataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainingArguments,
        folds: Sequence[Sequence[dict]],
        train_transform: Callable,
        val_transform: Callable,
    ):
        super().__init__()
        self.args = args
        self.folds = folds
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.val_fold_id = -1

    def set_fold_id(self, fold_id: int):
        self.val_fold_id = fold_id

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_set = list(itertools.chain(*self.folds[:self.val_fold_id], *self.folds[self.val_fold_id + 1:]))
        return DataLoader(
            Dataset(train_set, self.train_transform),
            batch_size=self.args.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_set = self.folds[self.val_fold_id]
        return DataLoader(
            Dataset(val_set, self.val_transform),
            batch_size=self.args.eval_batch_size,
            shuffle=True,
        )
