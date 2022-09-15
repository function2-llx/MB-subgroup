import pandas as pd
import torch
from torchmetrics import Recall

from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric
from umei.utils import DataKey, DataSplit, UMeIParser
from umei.utils.index_tracker import IndexTracker

from mbs.utils.enums import MBDataKey, Modality, SegClass
from mbs.args import MBSegArgs
from mbs.datamodule import MBSegDataModule, DATA_DIR, load_cohort
from mbs.model import MBSegModel

torch.multiprocessing.set_sharing_strategy('file_system')
# plot_case = None
plot_case = '470127wangwenkang'
val_id = None
dice_metric = DiceMetric()
recall_metric = Recall(num_classes=1, multiclass=False).cuda()

def plot(img, pred, seg):
    dice = dice_metric(pred, seg)
    recall = recall_metric(pred.view(-1), seg.view(-1).long())
    print(dice.item(), recall.item())
    IndexTracker(img[0, 0].cpu(), pred[0, 0].cpu())

def main():
    global val_id
    cohort = pd.read_excel(DATA_DIR / 'plan-split.xlsx', sheet_name='Sheet1')
    cohort.set_index('name', inplace=True)
    if val_id is None:
        val_id = int(cohort.loc[plot_case, 'split'])
    parser = UMeIParser((MBSegArgs, ), use_conf=True)
    args: MBSegArgs = parser.parse_args_into_dataclasses()[0]
    dm = MBSegDataModule(args)
    dm.val_id = val_id
    output_dir = args.output_dir / f'run-{args.seed}' / f'fold-{val_id}'
    ckpt_path = output_dir / 'last.ckpt'
    model: MBSegModel = MBSegModel.load_from_checkpoint(ckpt_path, args=args).cuda().eval()
    print(f'load from {ckpt_path}')

    for data in DataLoader(Dataset(
        [{
            MBDataKey.CASE: plot_case,
            **{
                img_type: DATA_DIR / 'image' / plot_case / f'{img_type}.nii'
                for img_type in list(Modality) + list(SegClass)
            }
        }],
        transform=dm.val_transform,
    )):
        seg = data[DataKey.SEG].cuda()
        img = data[DataKey.IMG].cuda()
        with torch.no_grad():
            for flip_idx in [[]] + model.tta_flips:
                print(flip_idx)
                flip_img = img.flip(flip_idx)
                pred = model.infer(flip_img)
                pred = (pred.sigmoid() > 0.5).long()
                plot(flip_img, pred, seg.flip(flip_idx))
                # pred = model.post_transform(pred[0])[None]
                # plot(img, pred, seg)

if __name__ == '__main__':
    main()
