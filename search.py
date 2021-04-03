from finetune import main, parse_args, Runner
from utils.data import load_folds

if __name__ == '__main__':
    args = parse_args()
    for sample_size in [112, 224]:
        folds = load_folds(args)
        for lr in [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
            args.lr = lr
            for wd in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
                args.weight_decay = wd
                for weight_strategy in ['invsqrt', 'equal', 'inv']:
                    args.sample_size = sample_size
                    main(args, folds)
