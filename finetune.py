import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import IntervalStrategy

from finetuner_base import FinetunerBase
from models import generate_model
from models.segresnet import SegResNetOutput
from utils.args import FinetuneArgs, ArgumentParser
from utils.data import MultimodalDataset

@dataclass
class EvalOutput:
    cls_loss: float = 0
    seg_loss: float = 0
    # N_examples * N_channels
    meandice: torch.FloatTensor = field(default_factory=list)

class Finetuner(FinetunerBase):
    args: FinetuneArgs

    def run_fold(self, val_id: int):
        fold_output_dir = Path(self.args.output_dir) / f'val-{val_id}'
        fold_output_dir.mkdir(exist_ok=True, parents=True)
        # output_path: Path = output_dir / f'checkpoint-{val_id}.pth.tar'
        # plot_dir = output_dir / f'plot-fold{val_id}'
        # plot_dir.mkdir(exist_ok=True, parents=True)
        if self.args.do_train and (self.args.overwrite_output_dir or not fold_output_dir.exists()):
            # tmp_output_path: Path = output_dir / f'checkpoint-{val_id}-tmp.pth.tar'
            # tmp_output_path: Path = output_dir / f'checkpoint-{val_id}/'
            writer = SummaryWriter(log_dir=str(Path(f'runs') / f'fold{val_id}' / fold_output_dir))
            logging.info(f'run cross validation on fold {val_id}')
            model = generate_model(
                self.args,
                pretrain=self.args.model_name_or_path is not None,
                num_seg=len(self.args.segs),
                num_classes=len(self.args.subgroups),
                num_pretrain_seg=self.args.num_pretrain_seg,
            )
            from torch.optim import AdamW
            optimizer = AdamW(model.finetune_parameters(self.args))
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.args.learning_rate, patience=3, verbose=True)
            train_set, val_set = self.prepare_fold(val_id)
            logging.info(f'''{torch.cuda.device_count()} GPUs available\n''')
            # model = nn.DataParallel(model)
            train_loader = DataLoader(train_set, batch_size=self.args.train_batch_size, shuffle=True)
            # loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.conf.weight_strategy).to(self.conf.device))
            cls_loss_fn = CrossEntropyLoss()
            seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
            # recons_fn = nn.MSELoss()
            best_loss = float('inf')
            step = 0
            val_outputs = []
            for epoch in range(1, int(self.args.num_train_epochs) + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    imgs = data['img'].to(self.args.device)
                    labels = data['label'].to(self.args.device)
                    seg_ref = data['seg'].to(self.args.device)
                    outputs: SegResNetOutput = model.forward(imgs)
                    # logits = outputs['linear']
                    logits = outputs.cls
                    cls_loss = cls_loss_fn(logits, labels)
                    # seg_loss = seg_loss_fn(outputs['seg'], segs)
                    seg_loss = seg_loss_fn(outputs.seg, seg_ref)
                    combined_loss = self.combine_loss(cls_loss, seg_loss, outputs.vae_loss)
                    # loss = self.args.cls_factor * cls_loss + self.args.seg_factor * seg_loss
                    # if self.conf.recons:
                    #     recons_loss = recons_fn(imgs, outputs['recons'])
                    # if self.is_world_master():
                    # self.args.process_index
                    writer.add_scalar('loss/train', combined_loss.item(), step)
                    writer.add_scalar('cls-loss/train', cls_loss.item(), step)
                    writer.add_scalar('seg-loss/train', seg_loss.item(), step)
                    if outputs.vae_loss is not None:
                        writer.add_scalar('vae-loss/train', outputs.vae_loss.item(), step)
                    # if self.conf.recons:
                    #     writer.add_scalar('recons/train', recons_loss.item(), step)

                    optimizer.zero_grad()
                    # if self.conf.recons:
                    #     loss += recons_loss
                    combined_loss.backward()
                    optimizer.step()
                    step += 1
                # if self.is_world_master() == 0:
                writer.flush()

                val_output = self.run_eval(model, val_set)
                val_loss = self.combine_loss(val_output.cls_loss, val_output.seg_loss)
                scheduler.step(val_loss)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')
                val_outputs.append(val_output)

                # if self.args.patience == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                if self.args.save_strategy is IntervalStrategy.EPOCH:
                    checkpoint_save_dir = fold_output_dir / f'checkpoint-ep{epoch}'
                else:
                    raise ValueError
                checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
                save_states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(save_states, checkpoint_save_dir / 'states.pth')
                logging.info(f'save checkpoint to {checkpoint_save_dir}\n')

                # else:
                #     if val_loss < best_loss:
                #         best_loss = val_loss
                #         torch.save(model.module.state_dict(), tmp_output_path)
                #         logging.info(f'model updated, saved to {tmp_output_path}\n')
                #         patience = 0
                #     else:
                #         patience += 1
                #         logging.info(f'patience {patience}/{self.args.patience}\n')
                #         if patience == self.args.patience:
                #             logging.info('run out of patience')
                #             break

            torch.save(model.state_dict(), fold_output_dir / 'checkpoint.pth.tar')
            # if self.is_world_master():
            # tmp_output_path.rename(output_path)
            # logging.info(f'move checkpoint permanently to {output_path}\n')
            fig, ax = plt.subplots()
            ax.plot([metric.cls_loss for metric in val_outputs], label='cls loss')
            ax.plot([metric.seg_loss for metric in val_outputs], label='seg loss')
            ax.plot([metric.meandice[:, 0].mean().item() for metric in val_outputs], label='AT mean DICE')
            ax.plot([metric.meandice[:, 1].mean().item() for metric in val_outputs], label='CT mean DICE')

            ax.legend()
            fig.savefig(fold_output_dir / 'val-loss.pdf')
            plt.show()
        else:
            if self.args.do_train:
                print('skip train')
            val_set = self.prepare_val_fold(val_id)

        model = generate_model(
            self.args,
            pretrain=False,
            num_seg=len(self.args.segs),
            num_classes=len(self.args.subgroups),
            num_pretrain_seg=self.args.num_pretrain_seg,
        )
        model.load_state_dict(torch.load(fold_output_dir / 'checkpoint.pth.tar'))
        self.run_eval(model, val_set, 'cross-val', plot_num=3, plot_dir=fold_output_dir)

    def run_eval(
        self,
        model: nn.Module,
        eval_dataset: MultimodalDataset,
        test_name: Optional[str] = None,
        plot_num=0,
        plot_dir: Path = None,
    ) -> EvalOutput:
        model.eval()
        post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )
        if plot_num > 0:
            assert plot_dir is not None
            plot_dir.mkdir(parents=True, exist_ok=True)

        cls_loss_fn = CrossEntropyLoss()
        seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        ret = EvalOutput()
        # plot_idx = random.sample(range(len(eval_dataset)), plot_num)
        meandices = []
        with torch.no_grad():
            step = 0
            for idx, data in enumerate(tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating')):
                img = data['img'].to(self.args.device)
                cls_label = data['label'].to(self.args.device)
                seg_ref = data['seg'].to(self.args.device)
                output = model.forward(img)
                logit = output['linear']
                seg_pred = torch.stack([post_trans(i) for i in decollate_batch(output['seg'])])
                cls_loss = cls_loss_fn(logit, cls_label)
                seg_loss = seg_loss_fn(output['seg'], seg_ref)
                ret.cls_loss += cls_loss.item()
                ret.seg_loss += seg_loss.item()
                meandice = compute_meandice(seg_pred, seg_ref)[0]
                if test_name:
                    self.reporters[test_name].append(data, logit[0], meandice)
                meandices.append(meandice)
                step += 1

                fig, ax = plt.subplots(1, 3, figsize=(16, 5))
                fig: Figure
                ax = {
                    protocol: ax[i]
                    for i, protocol in enumerate(self.args.protocols)
                }
                from utils.dicom_utils import ScanProtocol
                for protocol in self.args.protocols:
                    img = data[protocol][0, 0]
                    idx = img.shape[2] // 2
                    ax[protocol].imshow(np.rot90(img[:, :, idx]), cmap='gray')
                    seg_t = {
                        ScanProtocol.T2: 'AT',
                        ScanProtocol.T1c: 'CT',
                    }.get(protocol, None)
                    if seg_t is None:
                        continue
                    seg_id = self.args.segs.index(seg_t)
                    from matplotlib.colors import ListedColormap
                    cur_seg_ref = seg_ref[0, seg_id, :, :, idx].cpu().numpy()
                    cur_seg_pred = seg_pred[0, seg_id, :, :, idx]
                    cur_seg_pred = cur_seg_pred.int().cpu().numpy()
                    ax[protocol].imshow(np.rot90(cur_seg_pred), vmin=0, vmax=1, cmap=ListedColormap(['none', 'red']), alpha=0.5)
                    ax[protocol].imshow(np.rot90(cur_seg_ref), vmin=0, vmax=1, cmap=ListedColormap(['none', 'green']), alpha=0.5)
                fig.savefig(plot_dir / f"{data['patient'][0]}.pdf", dpi=300)
                plt.show()

        ret.cls_loss /= step
        ret.seg_loss /= step
        ret.meandice = torch.stack(meandices)
        return ret

# def get_model_output_root(args):
#     return args.output_root \
#         / args.targets \
#         / (f'{args.model}{args.model_depth}-scratch' if args.pretrain_name is None else args.pretrain_name) \
#         / '{aug_list},bs{batch_size},lr{lr},wd{weight_decay},{weight_strategy},{sample_size}x{sample_slices}'.format(
#             aug_list='+'.join(args.aug) if args.aug else 'no',
#             **args.__dict__
#         )

# def parse_args(search=False):
#     from argparse import ArgumentParser
#     from utils.dicom_utils import ScanProtocol
#     import utils.args
#     import models
#
#     parser = ArgumentParser(parents=[utils.args.parser, models.args.parser])
#     parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='invsqrt')
#     parser.add_argument('--targets', choices=['all', 'G3G4', 'WS'], default='all')
#     parser.add_argument('--n_folds', type=int, choices=[3, 4], default=3)
#     protocol_names = [value.name for value in ScanProtocol]
#     parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)
#     parser.add_argument('--output_root', type=Path, required=True)
#     parser.add_argument('--features', type=str)
#
#     args = parser.parse_args()
#     args = utils.args.process_args(args)
#     args = models.args.process_args(args)
#     args.target_names = {
#         'all': ['WNT', 'SHH', 'G3', 'G4'],
#         'WS': ['WNT', 'SHH'],
#         'G3G4': ['G3', 'G4'],
#     }[args.targets]
#     args.target_dict = {name: i for i, name in enumerate(args.target_names)}
#     # for output
#     if args.weight_decay == 0.0:
#         args.weight_decay = 0
#     assert len(args.protocols) in [1, 3]
#     if len(args.protocols) == 1:
#         args.protocols = [args.protocols[0] for _ in range(3)]
#     args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))
#     args.n_classes = len(args.target_names)
#     if not search:
#         args.model_output_root = get_model_output_root(args)
#         print('output root:', args.model_output_root)
#     return args

def finetune(args: FinetuneArgs, folds):
    runner = Finetuner(args, folds)
    runner.run()

def main():
    from utils.data.datasets.tiantan.load import load_cohort
    # conf = get_conf()
    parser = ArgumentParser([FinetuneArgs])
    args, = parser.parse_args_into_dataclasses()
    # conf.force_retrain = True
    finetune(args, load_cohort(args))

if __name__ == '__main__':
    main()
