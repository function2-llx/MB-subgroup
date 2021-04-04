import logging
from pathlib import Path
from typing import Optional

import torch
from monai.data import DataLoader
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resnet_3d.model import generate_model
from runner_base import FinetunerBase
from utils.data import MultimodalDataset
from utils.data.datasets.tiantan.load import load_folds
from utils.dicom_utils import ScanProtocol

class Finetuner(FinetunerBase):
    def run_fold(self, val_id: int):
        output_path = self.args.model_output_root / f'checkpoint-{val_id}.pth.tar'
        if self.args.train and (self.args.force_retrain or not output_path.exists()):
            tmp_output_path: Path = self.args.model_output_root / f'checkpoint-{val_id}-tmp.pth.tar'
            if self.args.rank == 0:
                writer = SummaryWriter(log_dir=Path(f'runs/fold{val_id}') / self.args.model_output_root)
            logging.info(f'run cross validation on fold {val_id}')
            model = generate_model(self.args, pretrain=self.args.pretrain_name is not None)
            from torch.optim import AdamW
            optimizer = AdamW(model.finetune_parameters(self.args))
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.args.lr_reduce_factor, patience=3, verbose=True)
            train_set, val_set = self.prepare_fold(val_id)
            logging.info(f'''{torch.cuda.device_count()} GPUs available\n''')
            model = nn.DataParallel(model)
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.args.weight_strategy).to(self.args.device))
            best_loss = float('inf')
            patience = 0
            step = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    imgs = data['img'].to(self.args.device)
                    labels = data['label'].to(self.args.device)
                    logits = model.forward(imgs)['linear']
                    loss = loss_fn(logits, labels)
                    if self.args.rank == 0:
                        writer.add_scalar('loss/train', loss.item(), step)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                if self.args.rank == 0:
                    writer.flush()

                val_loss = self.run_eval(model.module, val_set, loss_fn)
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
                    logging.info(f'patience {patience}/{self.args.patience}\n')
                    if patience == self.args.patience:
                        logging.info('run out of patience\n')
                        break
            tmp_output_path.rename(output_path)
        else:
            if self.args.train:
                print('skip train')
            val_set = self.prepare_val_fold(val_id)

        model = generate_model(self.args, pretrain=False)
        model.load_state_dict(torch.load(output_path))
        self.run_eval(model, val_set, CrossEntropyLoss(), 'cross-val')

    def run_eval(self, model: nn.Module, eval_dataset: MultimodalDataset, loss_fn: CrossEntropyLoss, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].to(self.args.device)
                logit = model.forward(img)['linear']
                if test_name:
                    self.reporters[test_name].append(logit[0], label.item())

                loss = loss_fn(logit, label)
                eval_loss += loss.item()
        return eval_loss

def parse_args():
    from argparse import ArgumentParser
    import utils.args
    import resnet_3d.model

    parser = ArgumentParser(parents=[utils.args.parser, resnet_3d.model.parser])
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='inv')
    parser.add_argument('--targets', choices=['all', 'G3G4', 'WS'], default='all')
    protocol_names = [value.name for value in ScanProtocol]
    parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)
    parser.add_argument('--output_root', type=Path, required=True)

    args = parser.parse_args()
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'WS': ['WNT', 'SHH'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    if args.weight_decay == 0.0:
        args.weight_decay = 0

    args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))

    return args

def main(args, folds):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_output_root = args.output_root \
        / args.targets \
        / (f'{args.model}{args.model_depth}-scratch' if args.pretrain_name is None else args.pretrain_name) \
        / '{aug},bs{batch_size},lr{lr},wd{weight_decay},{weight_strategy},{sample_size}x{sample_slices}'.format(**args.__dict__)
    print('output root:', args.model_output_root)
    args.n_classes = len(args.target_names)
    args.rank = 0
    runner = Finetuner(args, folds)
    runner.run()

if __name__ == '__main__':
    args = parse_args()
    main(args, load_folds(args))
