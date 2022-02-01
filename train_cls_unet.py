from dataclasses import dataclass
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cls_unet import ClsUNet, ClsUNetArgs
from utils.argparse import ArgParser
from utils.args import TrainingArgs
from utils.data import KFoldDataModule
from utils.data.datasets.tiantan.load import load_cohort

@dataclass
class Args(ClsUNetArgs, TrainingArgs):
    pass

def main():
    parser = ArgParser((Args, ))
    args: Args = parser.parse_args_into_dataclasses()[0]

    load_cohort

    trainer = Trainer.from_argparse_args(args)
    model = ClsUNet(args)
    data_module = KFoldDataModule(args, )

    callbacks = None
    model_ckpt = None
    if args.exec_mode == "train":
        model = NNUnet(args)
        early_stopping = EarlyStopping(monitor="dice_mean", patience=args.patience, verbose=True, mode="max")
        callbacks = [early_stopping]
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(
                filename="{epoch}-{dice_mean:.2f}", monitor="dice_mean", mode="max", save_last=True
            )
            callbacks.append(model_ckpt)
    else:  # Evaluation or inference
        if ckpt_path is not None:
            model = NNUnet.load_from_checkpoint(ckpt_path)
        else:
            model = NNUnet(args)

    trainer = Trainer(
        precision=16 if args.amp else 32,
        benchmark=True,
        deterministic=False,
        max_epochs=int(args.num_train_epochs),
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        resume_from_checkpoint=ckpt_path,
        accelerator="ddp" if args.gpus > 1 else None,
        checkpoint_callback=args.save_ckpt,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )

    if args.benchmark:
        if args.exec_mode == "train":
            trainer.fit(model, train_dataloader=data_module.train_dataloader())
        else:
            # warmup
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
            # benchmark run
            trainer.current_epoch = 1
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
    elif args.exec_mode == "train":
        trainer.fit(model, data_module)
    elif args.exec_mode == "evaluate":
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
