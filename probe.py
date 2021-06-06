if __name__ == '__main__':
    from models import generate_model

    def parse_args():
        import utils.args
        import models
        from argparse import ArgumentParser
        parser = ArgumentParser(parents=[utils.args.parser, models.args.parser])

        args = parser.parse_args()
        return args

    args = parse_args()
    args.target_names = ['WNT', 'SHH', 'G3', 'G4']
    args.target_dict = {
        name: i
        for i, name in enumerate(args.target_names)
    }
    model = generate_model(args, pretrain=True)
    from utils.data.datasets.tiantan import load_cohort
    cohort = load_cohort(args)
    
