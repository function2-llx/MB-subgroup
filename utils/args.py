import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from argparse import Namespace
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class DataTrainingArgs:
    sample_size: int = field(default=None)
    sample_slices: int = field(default=None)
    aug: List[str] = field(default=None)
    subjects: int = field(default=None)

@dataclass
class ModelArgs:
    model: str = field(default=None)
    resnet_shortcut: str = 'B'
    conv1_t_size: int = 7
    model_depth: int = 34
    conv1_t_stride: int = 1
    resnet_widen_factor: float = 1
    no_max_pool: bool = False
    cls_factor: float = None
    seg_factor: float = None

@dataclass
class PretrainArgs(DataTrainingArgs, ModelArgs, TrainingArguments):
    pass

class ArgumentParser(HfArgumentParser):
    def __init__(self, *args, use_conf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_conf = use_conf

    def parse_args_into_dataclasses(self, **kwargs):
        from sys import argv

        if not self.use_conf:
            return super().parse_args_into_dataclasses(**kwargs)
        conf_path = Path(argv[1])
        if conf_path.suffix in ['.yml', '.yaml']:
            with open(conf_path) as f:
                conf = yaml.safe_load(f)
        elif conf_path.suffix == '.json':
            with open(conf_path) as f:
                conf = json.load(f)
        else:
            raise ValueError(f'不支持的参数配置格式：{conf_path.suffix}')
        args = argv[2:]
        # 手动修复检查 required 不看 namespace 的问题
        if 'output_dir' in conf:
            args.extend(['--output_dir', conf['output_dir']])

        args, _ = self.parse_known_args(args=args, namespace=Namespace(**conf))
        return self.parse_dict(vars(args))
