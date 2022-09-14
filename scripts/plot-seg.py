import torch

from mbs.args import MBSegArgs
from mbs.datamodule import MBSegDataModule
from mbs.model import MBSegModel
from mbs.utils.enums import MBDataKey
from monai.metrics import DiceMetric
from umei.utils import DataKey, DataSplit, UMeIParser
from umei.utils.index_tracker import IndexTracker

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = UMeIParser((MBSegArgs, ), use_conf=True)
    args: MBSegArgs = parser.parse_args_into_dataclasses()[0]
    dm = MBSegDataModule(args)
    val_id = 0
    dm.val_id = val_id
    output_dir = args.output_dir / f'run-{args.seed}' / f'fold-{val_id}'
    model: MBSegModel = MBSegModel.load_from_checkpoint(output_dir / 'last.ckpt', args=args).cuda().eval()
    dice_metric = DiceMetric()
    for data in dm.val_dataloader():
        # data = next(iter(dm.val_dataloader()))
        data = data[DataSplit.VAL]
        case = data[MBDataKey.CASE][0]
        if case != '425605huangxiaoyi':
            continue
        print(case)
        seg = data[DataKey.SEG].cuda()
        img = data[DataKey.IMG].cuda()
        with torch.no_grad():
            pred = model.sw_infer(img)
        pred = (pred.sigmoid() > 0.5).long()
        if args.use_post:
            pred = model.post_transform(pred[0])[None]
        dice = dice_metric(pred, seg)
        print(dice.item())
        IndexTracker(img[0, 0], pred[0, 0])

if __name__ == '__main__':
    main()
