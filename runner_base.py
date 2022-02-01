import logging
import random
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from transformers import TrainingArguments

from utils import Intersection
from utils.args import DataTrainingArgs, ModelArgs

class RunnerBase(ABC):
    def __init__(self, args: Intersection[TrainingArguments, DataTrainingArgs, ModelArgs]):
        self.args = args
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        # yaml = YAML()
        # yaml.dump(vars(self.args), output_dir / 'conf.yml')

        if args.do_train:
            self.set_determinism()

    def combine_loss(self, cls_loss, seg_loss, vae_loss=None) -> torch.FloatTensor:
        ret = self.args.cls_factor * cls_loss + self.args.seg_factor * seg_loss
        if vae_loss is not None:
            ret += vae_loss * self.args.vae_factor
        return ret

    def set_determinism(self):
        # harm the speed
        # monai.utils.set_determinism(seed)
        # if self.is_world_master():
        seed = self.args.seed
        logging.info(f'set random seed of {seed}\n')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # not supported currently, throw RE for AdaptiveAvgPool
        # torch.use_deterministic_algorithms(True)

    def is_world_master(self) -> bool:
        return self.args.process_index == 0

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        if args.do_train:
            handlers.append(logging.FileHandler(Path(args.output_dir) / 'train.log', mode='a'))
            logging.basicConfig(
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt=logging.Formatter.default_time_format,
                level=logging.INFO,
                handlers=handlers,
                force=True,
            )
