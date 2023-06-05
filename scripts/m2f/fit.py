from copy import deepcopy

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from luolib.conf import parse_exp_conf

from mbs.conf import MBM2FConf
from mbs.datamodule import MBM2FDataModule
from mbs.models import MBM2FModel

task_name = 'mb-m2f'

def do_train(conf: MBM2FConf, datamodule: MBM2FDataModule, val_id: int):
    conf = deepcopy(conf)
    pl.seed_everything(conf.seed)
    conf.output_dir /= f'fold-{val_id}'
    if OmegaConf.is_missing(conf, 'log_dir'):
        conf.log_dir = conf.output_dir

    conf.output_dir.mkdir(parents=True, exist_ok=True)
    conf.log_dir.mkdir(exist_ok=True, parents=True)
    print('real output dir:', conf.output_dir)
    print('log dir:', conf.log_dir)
    MBM2FConf.save_conf_as_file(conf)
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
    )
    model = MBM2FModel(conf)
    trainer.fit(model, datamodule=datamodule, ckpt_path=MBM2FConf.get_last_ckpt_path(conf))
    wandb.finish()

def main():
    # torch.multiprocessing.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    conf = parse_exp_conf(MBM2FConf)
    conf.do_train = True
    datamodule = MBM2FDataModule(conf)
    conf.output_dir /= f'run-{conf.seed}'
    for val_id in conf.fold_ids:
        do_train(conf, datamodule, val_id)

if __name__ == '__main__':
    main()
