# adapt from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py

from copy import deepcopy
from functools import cached_property
from typing import Any, Optional

from pytorch_lightning.loops import FitLoop, Loop
from pytorch_lightning.trainer.states import TrainerFn

from utils.data import CrossValidationDataModule

class CrossValidationLoop(Loop):
    def __init__(self, num_folds: int) -> None:
        super().__init__()

        self.num_folds = num_folds
        self.val_fold_id: int = 0

        self.fit_loop: Optional[FitLoop] = None

    @property
    def done(self) -> bool:
        return self.val_fold_id >= self.num_folds

    def connect(self, *, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    @cached_property
    def cv_datamodule(self) -> CrossValidationDataModule:
        datamodule = self.trainer.datamodule
        assert isinstance(datamodule, CrossValidationDataModule)
        return datamodule

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        # assert isinstance(self.trainer.datamodule, CrossValidationDataModule)
        # self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.val_fold_id}")
        self.cv_datamodule.val_fold_id = self.val_fold_id

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()
        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()

    def on_advance_end(self) -> None:
        """save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.val_fold_id}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)

        self.val_fold_id += 1  # increment fold tracking number.

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> dict[str, int]:
        return {'val_fold': self.val_fold_id}

    def on_load_checkpoint(self, state_dict: dict) -> None:
        self.val_fold_id = state_dict['val_fold']

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_val_dataloaders()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
