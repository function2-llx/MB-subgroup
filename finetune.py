import logging
from pathlib import Path
from typing import Optional

import torch
from matplotlib import pyplot as plt
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

class Finetuner(FinetunerBase):
    def run_fold(self, val_id: int):
        output_path: Path = self.conf.model_output_root / f'checkpoint-{val_id}.pth.tar'
        if self.conf.do_train and (self.conf.force_retrain or not output_path.exists()):
            tmp_output_path: Path = self.conf.model_output_root / f'checkpoint-{val_id}-tmp.pth.tar'
            # if self.is_world_master():
            writer = SummaryWriter(log_dir=Path(f'runs') / f'fold{val_id}' / self.conf.model_output_root)
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
                    loss = seg_loss
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

                val_loss = self.run_eval(model.module, val_set, cls_loss_fn)
                scheduler.step(val_loss)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')

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
        else:
            if self.conf.do_train:
                print('skip train')
            val_set = self.prepare_val_fold(val_id)

        model = generate_model(self.conf, pretrain=False)
        model.load_state_dict(torch.load(output_path))
        self.run_eval(model, val_set, CrossEntropyLoss(), 'cross-val')

    def run_eval(self, model: nn.Module, eval_dataset: MultimodalDataset, loss_fn: CrossEntropyLoss, test_name: Optional[str] = None) -> float:
        plot = True
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            step = 0
            for data in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.conf.device)
                label = data['label'].to(self.conf.device)
                output = model.forward(img)
                logit = output['linear']
                if test_name:
                    self.reporters[test_name].append(data, logit[0])

                loss = loss_fn(logit, label)
                eval_loss += loss.item()
                step += 1

                if not plot:
                    continue
                seg = output['seg']
                fig, ax = plt.subplots(1, 3)
                ax = {
                    protocol: ax[i]
                    for i, protocol in enumerate(self.conf.protocols)
                }
                from utils.dicom_utils import ScanProtocol
                for protocol in self.conf.protocols:
                    img = data[protocol][0, 0]
                    idx = img.shape[2] // 2
                    ax[protocol].imshow(img[:, :, idx], cmap='gray')
                    seg_t = {
                        ScanProtocol.T2: 'AT',
                        ScanProtocol.T1c: 'CT',
                    }.get(protocol, None)
                    if seg_t is not None:
                        seg_id = self.conf.segs.index(seg_t)
                        from matplotlib.colors import ListedColormap
                        cur_seg = seg[0, seg_id, :, :, idx] > 0.5
                        cur_seg = cur_seg.int().cpu().numpy()
                        ax[protocol].imshow(cur_seg, vmin=0, vmax=1, cmap=ListedColormap(['none', 'green']), alpha=0.5)
                plt.show()

        return eval_loss / step

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
    conf.force_retrain = True
    finetune(conf, load_cohort(conf))

if __name__ == '__main__':
    main()
