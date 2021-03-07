from run_3d import main, get_args

if __name__ == '__main__':
    args = get_args()
    for batch_size in [2, 3, 4, 5]:
        args.batch_size = batch_size
        for lr in [1e-5, 2e-5, 3e-5, 4e-5]:
            args.lr = lr
            for weight_strategy in ['equal', 'invsqrt']:
                args.weight_strategy = weight_strategy
                for sample_size in [112, 150, 188, 224]:
                    args.sample_size = sample_size
                    for sample_slices in [16, 18, 20]:
                        args.sample_slices = sample_slices
                        main(args)
