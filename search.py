from finetune import main, parse_args, get_model_output_root
from utils.data.datasets.tiantan import load_folds

if __name__ == '__main__':
    args = parse_args(search=True)
    for sample_size in [112, 224]:
        args.sample_size = sample_size
        folds = load_folds(args)
        for lr in [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
            args.lr = lr
            for wd in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
                args.weight_decay = wd
                for weight_strategy in ['invsqrt', 'inv']:
                    args.weight_strategy = weight_strategy
                    args.model_output_root = get_model_output_root(args)
                    print('output root:', args.model_output_root)
                    main(args, folds)
