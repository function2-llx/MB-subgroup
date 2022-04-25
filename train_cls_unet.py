from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cls_unet import ClsUNet, ClsUNetArgs
from mbs.utils.argparse import ArgParser
from mbs.utils.args import TrainingArgs
from mbs.utils.cv_loop import CrossValidationLoop
from mbs.utils.data import CrossValidationDataModule

@dataclass
class Args(ClsUNetArgs, TrainingArgs):
    def __post_init__(self):
        super().__post_init__()

# cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12172/
class MyWandbLogger(WandbLogger):
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name

def get_cv_trainer(args: Args) -> pl.Trainer:
    trainer = pl.Trainer(
        logger=MyWandbLogger(name=args.exp_name, save_dir=str(args.output_root / args.exp_name)),
        gpus=args.n_gpu,
        precision=args.precision,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        callbacks=[
            ModelCheckpoint(monitor='cls_loss', save_last=True),
        ],
        num_sanity_val_steps=0,
        strategy=None,
        weights_save_path=str(args.output_root),
    )
    cv_loop = CrossValidationLoop(args.num_folds)
    cv_loop.connect(fit_loop=trainer.fit_loop)
    trainer.fit_loop = cv_loop
    return trainer

def main():
    parser = ArgParser((Args, ), exit_on_error=False)
    args: Args = parser.parse_args_into_dataclasses()[0]
    # trainer = Trainer.from_argparse_args(args)
    model = ClsUNet(args)
    trainer = get_cv_trainer(args)
    datamodule = CrossValidationDataModule(args)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
