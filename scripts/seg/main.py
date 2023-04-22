from copy import deepcopy

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from luolib.conf import parse_exp_conf

from mbs.conf import MBSegConf
from mbs.datamodule import MBSegDataModule
from mbs.models import MBSegModel

task_name = 'mb-seg'

def do_train(conf: MBSegConf, datamodule: MBSegDataModule, val_id: int):
    conf = deepcopy(conf)
    pl.seed_everything(conf.seed)
    conf.output_dir /= f'fold-{val_id}'
    if OmegaConf.is_missing(conf, 'log_dir'):
        conf.log_dir = conf.output_dir

    conf.output_dir.mkdir(parents=True, exist_ok=True)
    conf.log_dir.mkdir(exist_ok=True, parents=True)
    print('real output dir:', conf.output_dir)
    print('log dir:', conf.log_dir)
    MBSegConf.save_conf_as_file(conf)
    datamodule.val_id = val_id

    trainer = pl.Trainer(
        logger=WandbLogger(
            project=f'{task_name}-eval' if conf.do_eval else task_name,
            name=str(conf.output_dir.relative_to(conf.output_root)),
            save_dir=str(conf.log_dir),
            group=conf.exp_name,
            offline=conf.log_offline,
            resume=conf.resume_log,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=conf.output_dir,
                filename=f'step{{step}}-{conf.monitor.replace("/", "_")}={{{conf.monitor}:.3f}}',
                auto_insert_metric_name=False,
                monitor=conf.monitor,
                mode=conf.monitor_mode,
                verbose=True,
                save_last=True,
                save_top_k=conf.save_top_k,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval='step'),
            ModelSummary(max_depth=2),
        ],
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        num_nodes=conf.num_nodes,
        precision=conf.precision,
        benchmark=True,
        max_epochs=conf.max_epochs,
        max_steps=conf.max_steps,
        num_sanity_val_steps=conf.num_sanity_val_steps,
        val_check_interval=conf.val_check_interval,
        check_val_every_n_epoch=None,
    )
    model = MBSegModel(conf)
    trainer.fit(model, datamodule=datamodule, ckpt_path=MBSegConf.get_last_ckpt_path(conf))
    wandb.finish()

def do_eval(conf: MBSegConf, val_id: int):
    pass
    # log_dir = None
    # test_outputs = []
    # if conf.do_eval:
    #     log_dir = conf.output_dir / f'run-{conf.seed}' / 'eval'
    #     eval_suffix = f'sw{conf.sw_overlap}-{conf.sw_blend_mode}'
    #     if conf.do_tta:
    #         eval_suffix += '-tta'
    #     if conf.do_post:
    #         eval_suffix += '-post'
    #     log_dir /= eval_suffix
    #
    # for val_id in conf.fold_ids:
    #     datamodule.val_id = val_id
    #     pl.seed_everything(conf.seed)
    #     output_dir = conf.output_dir / f'run-{conf.seed}' / f'fold-{val_id}'
    #     output_dir.mkdir(exist_ok=True, parents=True)
    #     if conf.do_train:
    #         log_dir = output_dir
    #     log_dir.mkdir(exist_ok=True, parents=True)
    #     print('real output dir:', output_dir)
    #     print('log dir:', log_dir)
    #     trainer = pl.Trainer(
    #         logger=WandbLogger(
    #             project=f'{task_name}-eval' if conf.do_eval else task_name,
    #             name=str(output_dir.relative_to(conf.output_root)),
    #             save_dir=str(log_dir),
    #             group=conf.exp_name,
    #             offline=conf.log_offline,
    #             resume=conf.resume_log,
    #         ),
    #         callbacks=[
    #             ModelCheckpoint(
    #                 dirpath=output_dir,
    #                 filename=f'ep{{epoch}}-{conf.monitor.replace("/", " ")}={{{conf.monitor}:.3f}}',
    #                 auto_insert_metric_name=False,
    #                 monitor=conf.monitor,
    #                 mode=conf.monitor_mode,
    #                 verbose=True,
    #                 save_last=True,
    #                 save_on_train_epoch_end=False,
    #             ),
    #             LearningRateMonitor(logging_interval='epoch'),
    #             ModelSummary(max_depth=2),
    #         ],
    #         num_nodes=conf.num_nodes,
    #         accelerator='gpu',
    #         devices=torch.cuda.device_count(),
    #         precision=conf.precision,
    #         benchmark=True,
    #         max_epochs=int(conf.num_train_epochs),
    #         num_sanity_val_steps=conf.num_sanity_val_steps,
    #         log_every_n_steps=10,
    #         check_val_every_n_epoch=conf.eval_epochs,
    #         strategy=DDPStrategy(find_unused_parameters=conf.ddp_find_unused_parameters),
    #         # limit_train_batches=0.1,
    #         # limit_val_batches=0.2,
    #         # limit_test_batches=1,
    #     )
    #     model = MBSegModel(conf)
    #     last_ckpt_path = conf.ckpt_path
    #     if last_ckpt_path is None:
    #         last_ckpt_path = output_dir / 'last.ckpt'
    #         if not last_ckpt_path.exists():
    #             last_ckpt_path = None
    #     if conf.do_train:
    #         if trainer.is_global_zero:
    #             conf_save_path = output_dir / 'conf.yml'
    #             if conf_save_path.exists():
    #                 conf_save_path.rename(output_dir / 'conf-old.yml')
    #             UMeIParser.save_args_as_conf(conf, conf_save_path)
    #         trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
    #     if conf.do_eval:
    #         trainer.test(model, ckpt_path=last_ckpt_path, dataloaders=datamodule.val_dataloader())
    #         for x in model.test_outputs:
    #             x['fold'] = val_id
    #         test_outputs.extend(model.test_outputs)
    #         # partial results
    #         pd.DataFrame.from_records(test_outputs).to_excel(
    #             conf.output_dir / f'run-{conf.seed}' / 'results.xlsx',
    #             index=False,
    #         )
    #     wandb.finish()
    # if conf.do_eval:
    #     pd.DataFrame.from_records(test_outputs).to_excel(log_dir / 'results.xlsx', index=False)
    #     pd.DataFrame.from_records(test_outputs).to_csv(log_dir / 'results.csv', index=False)

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    conf = parse_exp_conf(MBSegConf)
    datamodule = MBSegDataModule(conf)

    conf.output_dir /= f'run-{conf.seed}'
    for val_id in conf.fold_ids:
        if conf.do_train:
            do_train(conf, datamodule, val_id)
        if conf.do_eval:
            do_eval(conf, val_id)

if __name__ == '__main__':
    main()
