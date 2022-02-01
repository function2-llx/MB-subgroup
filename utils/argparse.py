from argparse import Namespace
from pathlib import Path

from transformers import HfArgumentParser
from ruamel.yaml import YAML

yaml = YAML()

class ArgParser(HfArgumentParser):
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
            raise ValueError(f'format not supported for conf: {conf_path.suffix}')
        args = argv[2:]
        # manually fix problem that error is raised when required arguments are not found in command line
        # even if they already present in the `namespace` object
        if 'output_dir' in conf:
            args = ['--output_dir', conf['output_dir']] + args

        args, _ = self.parse_known_args(args=args, namespace=Namespace(**conf))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args_dict = vars(args)
        for k, v in args_dict.items():
            if isinstance(v, Path):
                args_dict[k] = str(v)
        yaml.dump(args_dict, output_dir / 'conf.yml')
        args = self.parse_dict(args_dict)
        return args
