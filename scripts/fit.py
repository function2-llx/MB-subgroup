import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.utils import UMeIParser

from mbs.args import MBArgs
from mbs.datamodule import MBDataModule
from mbs.model import MBModel

task_name = 'mb'
args: MBArgs
datamodule: MBDataModule

def fit_or_eval():
    log_dir = None
    test_outputs = []
    if args.do_eval:
        log_dir = args.output_dir / f'run-{args.seed}' / 'eval'
        eval_suffix = ''
        if args.do_tta:
            eval_suffix += '-tta'
        log_dir /= eval_suffix

    for val_id in args.fold_ids:
        datamodule.val_id = val_id
        pl.seed_everything(args.seed)
        output_dir = args.output_dir / f'run-{args.seed}' / f'fold-{val_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        if args.do_train:
            log_dir = output_dir
        log_dir.mkdir(exist_ok=True, parents=True)
        print('real output dir:', output_dir)
        print('log dir:', log_dir)
        trainer = pl.Trainer(
            logger=WandbLogger(
                project=f'{task_name}-eval' if args.do_eval else task_name,
                name=str(output_dir.relative_to(args.output_root)),
                save_dir=str(log_dir),
                group=args.exp_name,
                offline=args.log_offline,
                resume=args.resume_log,
            ),
            callbacks=[
                ModelCheckpoint(
                    dirpath=output_dir,
                    filename=f'ep{{epoch}}-{args.monitor.replace("/", " ")}={{{args.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    verbose=True,
                    save_last=True,
                    save_on_train_epoch_end=False,
                ),
                LearningRateMonitor(logging_interval='epoch'),
                ModelSummary(max_depth=2),
            ],
            num_nodes=args.num_nodes,
            accelerator='gpu',
            devices=torch.cuda.device_count(),
            precision=args.precision,
            benchmark=True,
            max_epochs=int(args.num_train_epochs),
            num_sanity_val_steps=args.num_sanity_val_steps,
            log_every_n_steps=10,
            check_val_every_n_epoch=args.eval_epochs,
            strategy=DDPStrategy(find_unused_parameters=args.ddp_find_unused_parameters),
            # limit_train_batches=0.1,
            # limit_val_batches=0.2,
            # limit_test_batches=1,
        )
        model = MBModel(args)
        seg_ckpt_path = output_dir / 'seg' / 'last.ckpt'
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(output_dir / 'seg' / 'last.ckpt')['state_dict'],
            strict=False,
        )
        assert len(unexpected_keys) == 0
        assert set(missing_keys) == {'cls_head.weight', 'cls_head.bias'}
        print(f'load seg model weights from {seg_ckpt_path}')

        last_ckpt_path = args.ckpt_path
        if last_ckpt_path is None:
            last_ckpt_path = output_dir / 'last.ckpt'
            if not last_ckpt_path.exists() or args.no_resume:
                last_ckpt_path = None
        if args.do_train:
            if trainer.is_global_zero:
                conf_save_path = output_dir / 'conf.yml'
                if conf_save_path.exists():
                    conf_save_path.rename(output_dir / 'conf-old.yml')
                UMeIParser.save_args_as_conf(args, conf_save_path)
            trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
        if args.do_eval:
            trainer.test(model, ckpt_path=last_ckpt_path, dataloaders=datamodule.val_dataloader())
            for x in model.test_outputs:
                x['fold'] = val_id
            test_outputs.extend(model.test_outputs)
            # partial results
            pd.DataFrame.from_records(test_outputs).to_excel(
                args.output_dir / f'run-{args.seed}' / 'results.xlsx',
                index=False,
            )
        wandb.finish()
    if args.do_eval:
        pd.DataFrame.from_records(test_outputs).to_excel(log_dir / 'results.xlsx', index=False)
        pd.DataFrame.from_records(test_outputs).to_csv(log_dir / 'results.csv', index=False)

def main():
    global args, datamodule
    parser = UMeIParser((MBArgs, ), use_conf=True)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    datamodule = MBDataModule(args)
    assert args.do_train ^ args.do_eval
    fit_or_eval()

if __name__ == '__main__':
    main()
