import logging
import random
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from monai.data import DataLoader
from monai.losses import DiceLoss
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import generate_model
from finetuner_base import FinetunerBase
from utils.conf import Conf
from utils.data import MultimodalDataset

@dataclass
class EvalMetric:
    cls_loss: float
    seg_loss: float

class Finetuner(FinetunerBase):
    def run_fold(self, val_id: int):
        output_path: Path = self.conf.output_dir / f'checkpoint-{val_id}.pth.tar'
        plot_dir = self.conf.output_dir / f'plot-fold{val_id}'
        plot_dir.mkdir(exist_ok=True, parents=True)
        if self.conf.do_train and (self.conf.force_retrain or not output_path.exists()):
            tmp_output_path: Path = self.conf.output_dir / f'checkpoint-{val_id}-tmp.pth.tar'
            # if self.is_world_master():
            writer = SummaryWriter(log_dir=str(Path(f'runs') / f'fold{val_id}' / self.conf.output_dir))
            logging.info(f'run cross validation on fold {val_id}')
            model = generate_model(self.conf, pretrain=self.conf.pretrain_name is not None)
            from torch.optim import AdamW
            optimizer = AdamW(model.finetune_parameters(self.conf))
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.conf.lr_reduce_factor, patience=3, verbose=True)
            train_set, val_set = self.prepare_fold(val_id)
            logging.info(f'''{torch.cuda.device_count()} GPUs available\n''')
            model = nn.DataParallel(model)
            train_loader = DataLoader(train_set, batch_size=self.conf.batch_size, shuffle=True)
            # loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.conf.weight_strategy).to(self.conf.device))
            cls_loss_fn = CrossEntropyLoss()
            seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
            # recons_fn = nn.MSELoss()
            best_loss = float('inf')
            patience = 0
            step = 0

            val_metrics = []
            for epoch in range(1, self.conf.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    imgs = data['img'].to(self.conf.device)
                    labels = data['label'].to(self.conf.device)
                    segs = data['seg'].to(self.conf.device)
                    outputs = model.forward(imgs)
                    logits = outputs['linear']
                    cls_loss = cls_loss_fn(logits, labels)
                    seg_loss = seg_loss_fn(outputs['seg'], segs)
                    loss = self.conf.cls_factor * cls_loss + self.conf.seg_factor * seg_loss
                    # if self.conf.recons:
                    #     recons_loss = recons_fn(imgs, outputs['recons'])
                    # if self.is_world_master():
                    writer.add_scalar('loss/train', loss.item(), step)
                    # if self.conf.recons:
                    #     writer.add_scalar('recons/train', recons_loss.item(), step)

                    optimizer.zero_grad()
                    # if self.conf.recons:
                    #     loss += recons_loss
                    loss.backward()
                    optimizer.step()
                    step += 1
                # if self.is_world_master() == 0:
                writer.flush()

                metric = self.run_eval(model.module, val_set)
                val_loss = self.combine_loss(metric.cls_loss, metric.seg_loss)
                scheduler.step(val_loss)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')
                val_metrics.append(metric)

                if self.conf.patience == 0:
                    torch.save(model.module.state_dict(), tmp_output_path)
                    logging.info(f'model updated, saved to {tmp_output_path}\n')
                else:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.module.state_dict(), tmp_output_path)
                        logging.info(f'model updated, saved to {tmp_output_path}\n')
                        patience = 0
                    else:
                        patience += 1
                        logging.info(f'patience {patience}/{self.conf.patience}\n')
                        if patience == self.conf.patience:
                            logging.info('run out of patience')
                            break

            # if self.is_world_master():
            tmp_output_path.rename(output_path)
            logging.info(f'move checkpoint permanently to {output_path}\n')
            plt.plot([metric.cls_loss for metric in val_metrics], label='cls loss')
            plt.plot([metric.seg_loss for metric in val_metrics], label='seg loss')
            plt.legend()
            plt.savefig(plot_dir / 'val-loss.pdf')
            plt.show()
        else:
            if self.conf.do_train:
                print('skip train')
            val_set = self.prepare_val_fold(val_id)

        model = generate_model(self.conf, pretrain=False)
        model.load_state_dict(torch.load(output_path))
        self.run_eval(model, val_set, 'cross-val', plot_num=3, plot_dir=plot_dir)

    def run_eval(
        self,
        model: nn.Module,
        eval_dataset: MultimodalDataset,
        test_name: Optional[str] = None,
        plot_num=0,
        plot_dir: Path = None,
    ) -> EvalMetric:
        model.eval()
        if plot_num > 0:
            assert plot_dir is not None
            plot_dir.mkdir(parents=True, exist_ok=True)

        cls_loss_fn = CrossEntropyLoss()
        seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        eval_cls_loss = 0
        eval_seg_loss = 0

        plot_idx = random.sample(range(len(eval_dataset)), plot_num)
        with torch.no_grad():
            step = 0
            for idx, data in enumerate(tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating')):
                img = data['img'].to(self.conf.device)
                cls_label = data['label'].to(self.conf.device)
                seg = data['seg'].to(self.conf.device)
                output = model.forward(img)
                logit = output['linear']
                if test_name:
                    self.reporters[test_name].append(data, logit[0])

                cls_loss = cls_loss_fn(logit, cls_label)
                seg_loss = seg_loss_fn(output['seg'], seg)
                eval_cls_loss += cls_loss.item()
                eval_seg_loss += seg_loss.item()
                step += 1

                if idx not in plot_idx:
                    continue

                seg = output['seg'].sigmoid()
                fig, ax = plt.subplots(1, 3, figsize=(16, 5))
                fig: Figure
                ax = {
                    protocol: ax[i]
                    for i, protocol in enumerate(self.conf.protocols)
                }
                from utils.dicom_utils import ScanProtocol
                for protocol in self.conf.protocols:
                    img = data[protocol][0, 0]
                    idx = img.shape[2] // 2
                    ax[protocol].imshow(np.rot90(img[:, :, idx]), cmap='gray')
                    seg_t = {
                        ScanProtocol.T2: 'AT',
                        ScanProtocol.T1c: 'CT',
                    }.get(protocol, None)
                    if seg_t is not None:
                        seg_id = self.conf.segs.index(seg_t)
                        from matplotlib.colors import ListedColormap
                        cur_seg = seg[0, seg_id, :, :, idx] > 0.5
                        seg_ref = data['seg'][0, seg_id, :, :, idx]
                        cur_seg = cur_seg.int().cpu().numpy()
                        ax[protocol].imshow(np.rot90(cur_seg), vmin=0, vmax=1, cmap=ListedColormap(['none', 'red']), alpha=0.5)
                        ax[protocol].imshow(np.rot90(seg_ref), vmin=0, vmax=1, cmap=ListedColormap(['none', 'green']), alpha=0.5)
                fig.savefig(plot_dir / f"{data['patient'][0]}.pdf", dpi=300)
                plt.show()

        return EvalMetric(
            cls_loss=eval_cls_loss / step,
            seg_loss=eval_seg_loss / step,
        )

def get_model_output_root(args):
    return args.output_root \
        / args.targets \
        / (f'{args.model}{args.model_depth}-scratch' if args.pretrain_name is None else args.pretrain_name) \
        / '{aug_list},bs{batch_size},lr{lr},wd{weight_decay},{weight_strategy},{sample_size}x{sample_slices}'.format(
            aug_list='+'.join(args.aug) if args.aug else 'no',
            **args.__dict__
        )

def parse_args(search=False):
    from argparse import ArgumentParser
    from utils.dicom_utils import ScanProtocol
    import utils.args
    import models

    parser = ArgumentParser(parents=[utils.args.parser, models.args.parser])
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='invsqrt')
    parser.add_argument('--targets', choices=['all', 'G3G4', 'WS'], default='all')
    parser.add_argument('--n_folds', type=int, choices=[3, 4], default=3)
    protocol_names = [value.name for value in ScanProtocol]
    parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--features', type=str)

    args = parser.parse_args()
    args = utils.args.process_args(args)
    args = models.args.process_args(args)
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'WS': ['WNT', 'SHH'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    # for output
    if args.weight_decay == 0.0:
        args.weight_decay = 0
    assert len(args.protocols) in [1, 3]
    if len(args.protocols) == 1:
        args.protocols = [args.protocols[0] for _ in range(3)]
    args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))
    args.n_classes = len(args.target_names)
    if not search:
        args.model_output_root = get_model_output_root(args)
        print('output root:', args.model_output_root)
    return args

def finetune(conf: Conf, folds):
    runner = Finetuner(conf, folds)
    runner.run()

def main():
    from utils.data.datasets.tiantan.load import load_cohort
    from utils.conf import get_conf
    conf = get_conf()
    conf.do_train = True
    # conf.force_retrain = True
    finetune(conf, load_cohort(conf))

if __name__ == '__main__':
    main()
