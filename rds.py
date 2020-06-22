import time
import argparse
from torchRDS.RDS import RDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforced Data Sampling")
    parser.add_argument("--data", type=str, default="datasets/madelon.csv",
                help="path to dataset file (default: datasets/madelon.csv)")
    parser.add_argument("--target", type=int, nargs="+", default=[0],
                help="column indexes for target variables (default: 0 - first column)")
    parser.add_argument("--data-loader", "-loader", type=str, default=None,
                help="the loader class for a dataset e.g., datasets.MNIST (default: None)")
    parser.add_argument("--exp-id", "-id", type=str, default=None,
                help="experiment id (default: timestamp)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/",
                help="output directory (default: outputs/)")
    parser.add_argument("--task", "-t", type=str, default="classification",
                help="task type: classification, regression (default: classification)")
    parser.add_argument("--measure", "-m", type=str, default="auc",
                help="measure type: cross_entropy (binary, multiclass), mse (regression), \
                auc (binary), f1_micro(classification), r2 (regression) (default: auc)")
    parser.add_argument("--envs", "-e", type=str, nargs="+", default=["models.LR"],
                help="learning environment (default: models.LR)")
    parser.add_argument("--learning", "-l", type=str, default="deterministic",
                help="deterministic or stochastic (default: deterministic)")
    parser.add_argument("--iterations", "-iters", type=int, default=1000,
                help="number of total iterations to run (default: 1000)")
    parser.add_argument("--burn-in-iterations", "-burnin", type=int, default=30,
                help="number of total iterations to burn (default: 30)")
    parser.add_argument("--episodes", "-eps", type=int, default=3,
                help="number of total episodes to run (default: 3)")
    parser.add_argument("--sampling-ratio", "-ratio", type=float, default=0.6,
                help="sampling ratio (default: 0.6)")
    parser.add_argument("--delta", "-d", type=float, default=0.01,
                help="sampling delta for saving (default: 0.01)")
    parser.add_argument("--weight-perf", "-wp", type=float, default=1.0,
                help="weight factor for model performance (default: 1.0)")
    parser.add_argument("--weight-ratio", "-wr", type=float, default=0.9,
                help="weight factor for sampling ratio (default: 0.9)")
    parser.add_argument("--weight-iid", "-wi", type=float, default=0.1,
                help="weight factor for class ratios in classification (default: 0.1)")
    parser.add_argument("--weight-kl", "-wk", type=float, default=0.1,
                help="weight factor for distributional divergence in regression (default: 0.1)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001,
                help="initial learning rate (default: 0.001)")
    parser.add_argument("--hidden-dim", "-hd", type=int, default=256,
                help="number of nodes in the hidden layer (default: 256)")
    parser.add_argument("--device", "-dev", type=str, default="cuda",
                help="device to run: cuda, cpu (default: cuda)")
    parser.add_argument("--verbose", "-v", type=int, default=1,
                help="verbose: 0 - no printing, 1 - printing (default: 1)")

    args = parser.parse_args()
    exp_id = str(int(time.time())) if args.exp_id is None else args.exp_id
    opt = {"exp": exp_id,
           "data_file": args.data,
           "target": args.target,
           "data_loader": args.data_loader,
           "task": args.task,
           "measure": args.measure,
           "model_classes": args.envs,
           "learn": args.learning,
           "iters": args.iterations,
           "burnin": args.burn_in_iterations,
           "eps": args.episodes if args.episodes > 0 else len(args.envs),
           "ratio": args.sampling_ratio,
           "delta": args.delta,
           "weight_perf": args.weight_perf,
           "weight_ratio": args.weight_ratio,
           "weight_iid": args.weight_iid,
           "weight_kl": args.weight_kl,
           "learning_rate": args.learning_rate,
           "hidden_dim": args.hidden_dim,
           "dev": args.device,
           "sample_file": "{}/{}.npy".format(args.output_dir, exp_id),
           "checkpoint_file": "{}/{}.pkl".format(args.output_dir, exp_id),
           "report_file": "{}/{}.txt".format(args.output_dir, exp_id),
           "verbose": args.verbose
           }

    print("Reinforced Data Sampling")
    print("\n".join("{}: {}".format(k, v) for k, v in opt.items()))
    trainer = RDS(opt)
    trainer.train()
