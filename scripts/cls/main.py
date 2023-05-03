from copy import deepcopy

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from luolib.conf import parse_exp_conf
from mbs.conf import MBClsConf
from mbs.datamodule import MBClsDataModule
from mbs.models.lightning.cls_model import MBClsModel

task_name = 'mbs'

def do_train(conf: MBClsConf, datamodule: MBClsDataModule, val_id: int):
    conf = deepcopy(conf)
    pl.seed_everything(conf.seed)
    if val_id == -1:
        conf.output_dir /= f'all'
    else:
        conf.output_dir /= f'fold-{val_id}'
    if OmegaConf.is_missing(conf, 'log_dir'):
        conf.log_dir = conf.output_dir

    conf.output_dir.mkdir(parents=True, exist_ok=True)
    conf.log_dir.mkdir(exist_ok=True, parents=True)
    print('real output dir:', conf.output_dir)
    print('log dir:', conf.log_dir)
    if val_id == -1:
        datamodule.val_id = None
    else:
        datamodule.val_id = val_id

    trainer = pl.Trainer(
        logger=WandbLogger(
            project=task_name,
            name=str(conf.output_dir.relative_to(conf.output_root)),
            save_dir=str(conf.log_dir),
            group=conf.exp_name,
            offline=conf.log_offline,
            resume=conf.resume_log,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=conf.output_dir,
                filename=f'step{{step}}-{conf.monitor.replace("/", "_")}={{{conf.monitor}:.4f}}',
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
        log_every_n_steps=10,
    )
    if conf.pretrain_cv_dir is not None and conf.backbone.ckpt_path is None:
        conf.backbone.ckpt_path = conf.pretrain_cv_dir / f'fold-{val_id}/last.ckpt'
    else:
        print('train from sctrach!')
    model = MBClsModel(conf)
    MBClsConf.save_conf_as_file(conf)
    trainer.fit(model, datamodule=datamodule, ckpt_path=MBClsConf.get_last_ckpt_path(conf))
    wandb.finish()

def main():
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    conf = parse_exp_conf(MBClsConf)

    conf.output_dir /= f'run-{conf.seed}'
    for val_id in conf.fold_ids:
        datamodule = MBClsDataModule(conf)
        if conf.do_train:
            do_train(conf, datamodule, val_id)

if __name__ == '__main__':
    main()
