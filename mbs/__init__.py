from pathlib import Path

from omegaconf import OmegaConf

conf_root = Path(__file__).parent.parent / 'conf'
OmegaConf.register_new_resolver('mbs.conf_root', lambda: conf_root)
