import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ruamel.yaml import YAML
from transformers import HfArgumentParser

yaml = YAML()

@dataclass
class DataTrainingArgs:
    sample_size: int = field(default=None)
    sample_slices: int = field(default=None)
    aug: List[str] = field(default=None)
    subjects: int = field(default=None)
    input_fg_mask: bool = field(default=False)
    use_focal: bool = field(default=False)

@dataclass
class ModelArgs:
    model: str = field(default=None)
    model_name_or_path: str = field(default=None)
    resnet_shortcut: str = field(default='B')
    conv1_t_size: int = 7
    model_depth: int = 34
    conv1_t_stride: int = 1
    resnet_widen_factor: float = 1
    no_max_pool: bool = False
    cls_factor: float = None
    seg_factor: float = None
    vae_factor: float = None

class ArgumentParser(HfArgumentParser):
    def __init__(self, *args, use_conf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_conf = use_conf

    def parse_args_into_dataclasses(self, **kwargs):
        from sys import argv

        if not self.use_conf:
            return super().parse_args_into_dataclasses(**kwargs)
        conf_path = Path(argv[1])
        if conf_path.suffix in ['.yml', '.yaml', '.json']:
            conf = yaml.load(conf_path)
        else:
            raise ValueError(f'不支持的参数配置格式：{conf_path.suffix}')
        args = argv[2:]
        # 手动修复检查 required 不看 namespace 的问题
        if 'output_dir' in conf:
            args = ['--output_dir', conf['output_dir']] + args

        args, _ = self.parse_known_args(args=args, namespace=Namespace(**conf))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args_dict = vars(args)
        yaml.dump(args_dict, output_dir / 'conf.yml')
        args = self.parse_dict(args_dict)
        return args
