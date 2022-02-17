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
    pass

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
    parser = ArgParser((Args, ))
    args: Args = parser.parse_args_into_dataclasses()[0]
    # trainer = Trainer.from_argparse_args(args)
    model = ClsUNet(args)
    trainer = get_cv_trainer(args)
    data_module = CrossValidationDataModule(args)

    trainer.fit(model, data_module)

    callbacks = []
    checkpoint_callback = None
    if args.do_train:
        model = NNUnet(args)
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{dice_mean:.2f}", monitor="dice_mean", mode="max", save_last=True
        )
        checkpoint_callback.last_model_path
        callbacks.append(checkpoint_callback)
    else:  # Evaluation or inference
        if ckpt_path is not None:
            model = NNUnet.load_from_checkpoint(ckpt_path)
        else:
            model = NNUnet(args)
    trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop
    if args.do_train:
        trainer.fit(model, data_module)
    if args.do_eval:
        model.args = args
        trainer.validate()
        trainer.test(model, test_dataloaders=data_module.val_dataloader())

if __name__ == "__main__":
    main()
