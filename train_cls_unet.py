from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cls_unet import ClsUNet, ClsUNetArgs
from utils.argparse import ArgParser
from utils.args import TrainingArgs
from utils.cv_loop import CrossValidationLoop
from utils.data import CrossValidationDataModule

@dataclass
class Args(ClsUNetArgs, TrainingArgs):
    def __post_init__(self):
        super().__post_init__()

def get_cv_trainer(args: Args) -> pl.Trainer:
    trainer = pl.Trainer(
        gpus=args.n_gpu,
        precision=16 if args.amp else 32,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        callbacks=[
            ModelCheckpoint()
        ],
        num_sanity_val_steps=0,
        checkpoint_callback=None,
        strategy=None,
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
