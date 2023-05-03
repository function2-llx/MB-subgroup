import dill as pickle
import toolz
import torch
from tqdm import tqdm

from luolib.conf import parse_exp_conf
from luolib.utils import DataKey, DataSplit
from mbs.models import MBClsModel
from monai.data import DataLoader, Dataset

from mbs.conf import MBClsConf
from mbs.datamodule import MBClsDataModule

def main():
    torch.multiprocessing.set_start_method('fork')
    torch.multiprocessing.set_sharing_strategy('file_system')
    conf = parse_exp_conf(MBClsConf)
    conf.num_cls_classes = 4
    datamodule = MBClsDataModule(conf)
    model = MBClsModel(conf).cuda().eval()
    split_cohort = datamodule.split_cohort
    test_cohort = split_cohort[DataSplit.TEST]
    train_cohort = list(toolz.concat([data for split, data in split_cohort.items() if split != DataSplit.TEST]))
    train_cohort = sorted(train_cohort, key=lambda x: x[DataKey.CASE])
    test_cohort = sorted(test_cohort, key=lambda x: x[DataKey.CASE])
    dataset = Dataset(
        train_cohort + test_cohort,
        transform=datamodule.test_transform(),
    )
    data_loader = DataLoader(dataset, num_workers=8, batch_size=16, pin_memory=True, persistent_workers=True)
    feature_dict = {}
    with torch.no_grad():
        for epoch in range(10):
            for batch in tqdm(data_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                feature = model.cal_feature(batch)
                for i, case in enumerate(batch[DataKey.CASE]):
                    feature_dict[case] = feature[i]
    torch.save(feature_dict, 'feature.pt')

if __name__ == '__main__':
    main()
