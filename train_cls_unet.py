from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
import os

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import monai
from monai.utils import InterpolateMode

from cls_unet import ClsUNet, ClsUNetArgs
from utils.argparse import ArgParser
from utils.args import TrainingArgs
from utils.data import KFoldDataModule
from utils.transforms import CreateForegroundMaskD

@dataclass
class Args(ClsUNetArgs, TrainingArgs):
    pass

def main():
    parser = ArgParser((Args, ))
    args: Args = parser.parse_args_into_dataclasses()[0]
    # trainer = Trainer.from_argparse_args(args)
    model = ClsUNet(args)
    data_module = KFoldDataModule(args)
    trainer = Trainer(
        gpus=args.n_gpu,
        precision=16 if args.amp else 32,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        callbacks=callbacks,
        num_sanity_val_steps=0,
        # default_root_dir=args.output_dir,
        strategy=None,
    )
    trainer.early_stopping_callbacks
    for fold_id in range(args.num_folds):
        data_module.val_fold_id = fold_id



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
    Trainer.from_argparse_args()

    if args.do_train:
        trainer.fit(model, data_module)
    if args.do_eval
        model.args = args
        trainer.test(model, test_dataloaders=data_module.val_dataloader())
    elif args.exec_mode == "predict":
        if args.save_preds:
            ckpt_name = "_".join(args.ckpt_path.split("/")[-1].split(".")[:-1])
            dir_name = f"predictions_{ckpt_name}"
            dir_name += f"_task={model.args.task}_fold={model.args.fold}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        model.args = args
        trainer.test(model, test_dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    main()
