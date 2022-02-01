from __future__ import annotations

import logging
import random
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import monai.transforms
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from monai.data import DataLoader, NiftiSaver
from monai.losses import DiceLoss, DiceFocalLoss
from monai.metrics import compute_meandice
from monai.utils import ImageMetaKey
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from finetuner_base import FinetunerBase, FinetuneArgs
from models import generate_model, Backbone
from models.segresnet import SegResNetOutput
from utils.argparse import ArgParser
from utils.data import MultimodalDataset
from utils.dicom_utils import ScanProtocol
from utils.report import Reporter

@dataclass
class EvalOutput:
    cls_loss: float = 0
    seg_loss: float = 0
    # N_examples * N_channels
    meandice: torch.FloatTensor = field(default_factory=list)

class Finetuner(FinetunerBase):
    args: FinetuneArgs

    def run_fold(self, val_id: int):
        logging.info(f'run cross validation on fold {val_id}')
        fold_output_dir = Path(self.args.output_dir) / f'val-{val_id}'
        fold_output_dir.mkdir(exist_ok=True, parents=True)
        best_checkpoint_path = fold_output_dir / 'checkpoint-best.pth.tar'
        latest_checkpoint_path = fold_output_dir / 'checkpoint-latest.pth.tar'

        if self.args.do_train:
            latest_checkpoint = None
            if not self.args.overwrite_output_dir and latest_checkpoint_path.exists():
                latest_checkpoint = torch.load(latest_checkpoint_path)
            writer = SummaryWriter(log_dir=str(Path(f'runs') / f'fold{val_id}' / fold_output_dir))
            model = generate_model(
                self.args,
                in_channels=self.args.in_channels,
                pretrain=self.args.model_name_or_path is not None,
                num_seg=len(self.args.segs),
                num_classes=len(self.args.subgroups),
                num_pretrain_seg=self.args.num_pretrain_seg,
            )
            from torch.optim import AdamW
            optimizer = AdamW(model.finetune_parameters(self.args))
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.args.lr_reduce_factor,
                patience=self.args.patience,
                verbose=True,
            )
            val_outputs = []
            best_loss = float('inf')
            if latest_checkpoint is None:
                epoch = 0
                logging.info('start training')
            else:
                model.load_state_dict(latest_checkpoint['model'])
                optimizer.load_state_dict(latest_checkpoint['optimizer'])
                scheduler.load_state_dict(latest_checkpoint['scheduler'])
                best_loss = scheduler.best
                val_outputs, self.epoch_reporters, self.reporters = latest_checkpoint['results']
                logging.info(f'resume training at epoch {scheduler.last_epoch}')
                epoch = scheduler.last_epoch + 1

            train_set, val_set = self.prepare_fold(val_id)
            logging.info(f'{torch.cuda.device_count()} GPUs available\n')
            # model = nn.DataParallel(model)
            train_loader = DataLoader(train_set, batch_size=self.args.train_batch_size, shuffle=True)
            # loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.conf.weight_strategy).to(self.conf.device))
            cls_loss_fn = CrossEntropyLoss()
            if self.args.use_focal:
                seg_loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
            else:
                seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
            # recons_fn = nn.MSELoss()
            step = 0
            plot_skip_first = True

            for epoch in range(epoch, int(self.args.num_train_epochs) + 1):
                # evaluate zero-shot performance when epoch = 0
                if epoch > 0:
                    model.train()
                    for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                        imgs = data['img']
                        labels = data['label']
                        seg_ref = data['seg']
                        outputs: SegResNetOutput = model.forward(imgs, permute=True)
                        logits = outputs.cls_logit
                        cls_loss = cls_loss_fn(logits, labels)
                        seg_loss = seg_loss_fn(outputs.seg_logit, seg_ref)
                        combined_loss = self.combine_loss(cls_loss, seg_loss, outputs.vae_loss)
                        writer.add_scalar('loss/train', combined_loss.item(), step)
                        writer.add_scalar('cls-loss/train', cls_loss.item(), step)
                        writer.add_scalar('seg-loss/train', seg_loss.item(), step)
                        if outputs.vae_loss is not None:
                            writer.add_scalar('vae-loss/train', outputs.vae_loss.item(), step)
                        optimizer.zero_grad()
                        combined_loss.backward()
                        optimizer.step()
                        step += 1
                    # if self.is_world_master() == 0:
                    writer.flush()

                val_output = self.run_eval(
                    model,
                    val_set,
                    reporter=self.epoch_reporters[epoch]['cross-val'],
                    plot_dir=self.epoch_reporters[epoch]['cross-val'].report_dir,
                    plot_num=3,
                )
                val_loss = self.combine_loss(val_output.cls_loss, val_output.seg_loss)
                if epoch > 0:
                    scheduler.step(val_loss)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')
                if not plot_skip_first or epoch > 0:
                    val_outputs.append(val_output)

                save_states = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'results': (val_outputs, self.epoch_reporters, self.reporters)
                }
                torch.save(save_states, latest_checkpoint_path)
                logging.info(f'epoch {epoch}: save latest checkpoint to {latest_checkpoint_path}\n')
                if val_loss < best_loss:
                    best_loss = val_loss
                    shutil.copy2(latest_checkpoint_path, best_checkpoint_path)
                    logging.info(f'copy best checkpoint to {best_checkpoint_path}\n')

            pd.DataFrame({
                f'epoch{epoch}': epoch_reporters['cross-val'].digest()
                for epoch, epoch_reporters in self.epoch_reporters.items()
            }).transpose().to_csv(Path(self.args.output_dir) / 'train-digest.csv')
            fig, ax = plt.subplots()
            if self.args.cls_factor != 0:
                ax.plot([metric.cls_loss for metric in val_outputs], label='cls loss')
            if self.args.seg_factor != 0:
                ax.plot([metric.seg_loss for metric in val_outputs], label='seg loss')
            for i, seg_name in enumerate(self.args.segs):
                ax.plot([metric.meandice[:, i].mean().item() for metric in val_outputs], label=f'{seg_name} mean DICE')
            ax.legend()
            fig.savefig(fold_output_dir / 'val-loss.pdf')
            plt.show()
            plt.close()

        if self.args.do_eval:
            val_set = self.prepare_val_fold(val_id)
            logging.info('run validation')
            model = generate_model(
                self.args,
                in_channels=self.args.in_channels,
                pretrain=False,
                num_seg=len(self.args.segs),
                num_classes=len(self.args.subgroups),
                num_pretrain_seg=self.args.num_pretrain_seg,
            )
            model.load_state_dict(torch.load(best_checkpoint_path)['model'])
            self.run_eval(
                model,
                val_set,
                self.reporters['cross-val'],
                plot_num=len(val_set),
                plot_dir=Path(self.args.output_dir) / 'seg-outputs',
                save_dir=Path(self.args.output_dir) / 'seg-outputs',
            )

    def run_eval(
        self,
        models: Backbone | list[Backbone],
        eval_dataset: MultimodalDataset,
        reporter: Optional[Reporter] = None,
        plot_num=0,
        plot_dir: Path = None,
        save_dir: Path = None,
    ) -> EvalOutput:
        if not isinstance(models, list):
            models = [models]
        for model in models:
            model.eval()

        cls_post_trans = monai.transforms.Compose([
            monai.transforms.EnsureType(),
            # Activations(softmax=True),
            monai.transforms.MeanEnsemble(),
        ])
        applied_labels = 1 if len(self.args.segs) == 1 else range(len(self.args.segs))
        seg_post_trans = monai.transforms.Compose([
            monai.transforms.EnsureType(),
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.MeanEnsemble(),
            monai.transforms.AsDiscrete(threshold=0.5),
            monai.transforms.KeepLargestConnectedComponent(applied_labels=applied_labels),
            monai.transforms.FillHoles(applied_labels=applied_labels),
        ])
        plot_idx = random.sample(range(len(eval_dataset)), plot_num)
        if plot_num > 0:
            assert plot_dir is not None
            plot_dir.mkdir(parents=True, exist_ok=True)

        cls_loss_fn = CrossEntropyLoss()
        seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        ret = EvalOutput()
        if plot_num > 0:
            assert plot_dir is not None
            plot_dir.mkdir(parents=True, exist_ok=True)
        meandices = []
        with torch.no_grad():
            step = 0
            for idx, data in enumerate(tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating')):
                patient = data['patient'][0]
                img = data['img'].to(self.args.device)
                cls_label = data['label'].to(self.args.device)
                seg_ref: torch.LongTensor = data['seg'].to(self.args.device)
                outputs: list[SegResNetOutput] = [model.forward(img, permute=True) for model in models]
                cls_prob = cls_post_trans(torch.stack([
                    torch.softmax(output.cls_logit[0], dim=0) for output in outputs
                ]))[None]
                seg_pred = seg_post_trans(torch.stack([output.seg_logit[0] for output in outputs]))[None]
                for output in outputs:
                    ret.cls_loss += cls_loss_fn(output.cls_logit, cls_label).item()
                    ret.seg_loss += seg_loss_fn(output.seg_logit, seg_ref).item()

                meandice = compute_meandice(seg_pred, seg_ref)[0]
                if reporter is not None:
                    reporter.append(data, cls_prob[0], meandice)
                meandices.append(meandice)
                step += 1
                if idx in plot_idx:
                    patient_plot_dir = plot_dir / patient
                    patient_plot_dir.mkdir(exist_ok=True)
                    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
                    fig: Figure
                    ax = {
                        protocol: ax[i]
                        for i, protocol in enumerate(self.args.protocols)
                    }

                    # select a slice to plot segmentation result
                    idx = seg_ref[0].sum(dim=(0, 1, 2)).argmax().item()
                    for protocol in self.args.protocols:
                        img = data[protocol][0, 0]
                        # idx = img.shape[2] // 2
                        ax[protocol].imshow(np.rot90(img[:, :, idx]), cmap='gray')
                        seg_t = {
                            ScanProtocol.T2: 'AT',
                            ScanProtocol.T1c: 'CT',
                        }.get(protocol, None)
                        if seg_t is None or seg_t not in self.args.segs:
                            continue
                        seg_id = self.args.segs.index(seg_t)
                        from matplotlib.colors import listedColormap
                        cur_seg_ref = seg_ref[0, seg_id, :, :, idx].cpu().numpy()
                        cur_seg_pred = seg_pred[0, seg_id, :, :, idx].int().cpu().numpy()
                        ax[protocol].imshow(np.rot90(cur_seg_pred), vmin=0, vmax=1, cmap=listedColormap(['none', 'red']), alpha=0.5)
                        ax[protocol].imshow(np.rot90(cur_seg_ref), vmin=0, vmax=1, cmap=listedColormap(['none', 'green']), alpha=0.5)
                    fig.savefig(patient_plot_dir / 'plot.pdf', dpi=300)
                    plt.show()
                    plt.close()

                if save_dir is not None:
                    patient_save_dir = save_dir / patient
                    saver = NiftiSaver(
                        output_dir=patient_save_dir,
                        output_postfix='',
                        output_dtype=np.int8,
                        resample=False,
                        separate_folder=False,
                        print_log=False,
                    )
                    seg_ref_keys = {
                        'AT': ScanProtocol.T2,
                        'CT': ScanProtocol.T1c,
                    }
                    for i, seg in enumerate(self.args.segs):
                        cur_seg_pred = seg_pred[0, [seg_id]].int().cpu().numpy()
                        seg_ref_key = seg_ref_keys[seg]
                        saver.save(cur_seg_pred, {
                            ImageMetaKey.FILENAME_OR_OBJ: seg,
                            'affine': data[f'{str(seg_ref_key)}_meta_dict']['affine'][0],
                        })
        ret.cls_loss /= step
        ret.seg_loss /= step
        ret.meandice = torch.stack(meandices)
        if reporter is not None:
            reporter.report()
        return ret

def main():
    from utils.data.datasets.tiantan.load import load_cohort
    # conf = get_conf()
    parser = ArgParser([FinetuneArgs])
    args, = parser.parse_args_into_dataclasses()
    args: FinetuneArgs
    cohort_folds = load_cohort(args, split_folds=True)
    output_dir = Path(args.output_dir)
    rng = np.random.RandomState(args.seed)
    for run in range(args.n_runs):
        run_args = deepcopy(args)
        run_args.output_dir = str(output_dir / f'run-{run}')
        run_args.seed = rng.randint(23333)
        runner = Finetuner(run_args, cohort_folds)
        runner.run()
    if args.do_ensemble:
        ensemble_runner = Finetuner(args, cohort_folds)
        for val_id in range(len(cohort_folds)):
            val_set = ensemble_runner.prepare_val_fold(val_id)
            models = []
            for run in range(args.n_runs):
                model = generate_model(
                    args,
                    in_channels=args.in_channels,
                    pretrain=False,
                    num_seg=len(args.segs),
                    num_classes=len(args.subgroups),
                    num_pretrain_seg=args.num_pretrain_seg,
                )
                model.load_state_dict(
                    torch.load(output_dir / f'run-{run}' / f'val-{val_id}' / 'checkpoint-best.pth.tar')['model']
                )
                models.append(model)

            ensemble_runner.run_eval(
                models=models,
                eval_dataset=val_set,
                reporter=ensemble_runner.reporters['cross-val'],
                plot_num=len(val_set),
                plot_dir=output_dir / 'seg-outputs',
                save_dir=output_dir / 'seg-outputs',
            )

if __name__ == '__main__':
    main()
