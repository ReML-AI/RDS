import argparse
import numpy as np
from util import *

parser = argparse.ArgumentParser(description="RDS Evaluator")
parser.add_argument("--data", type=str, default="datasets/madelon.csv",
                    help="path to dataset file (default: datasets/madelon.csv)")
parser.add_argument("--target", type=int, nargs="+", default=[0],
                    help="column indexes for response variables (default: 0 - first column)")
parser.add_argument('--data-loader', '-loader', type=str, default=None,
                    help='the loader class for a dataset e.g., datasets.MNIST (default: None)')
parser.add_argument("--sample", type=str, default="samples/MDL_RANDOM.npy.npy",
                    help="path to sampling file (default: samples/MDL_RANDOM.npy.npy)")
parser.add_argument('--task', '-t', type=str, default='classification',
                    help='task type: classification, regression (default: classification)')
parser.add_argument("--measure", "-m", type=str, default="auc",
                    help="measure type: cross_entropy (binary, multiclass), mse (regression), auc (binary), r2 (regression) (default: cross_entropy)")
parser.add_argument("--envs", "-e", type=str, nargs="+", default=["models.LR"],
                    help="learning environment (default: models.LR)")


if __name__ == "__main__":
    args = parser.parse_args()
    opt = {"data": args.data,
           "target": args.target,
           'loader': args.data_loader,
           "sample": args.sample,
           "task": args.task,
           "measure": args.measure,
           "envs": args.envs
           }
    print("Evaluator")
    print("\n".join("{}:     \t{}".format(k, v) for k, v in opt.items()))

    data_x, data_y = load_data(opt["data"], opt["target"], opt["task"] == "classification", opt["loader"])
    selection = np.load(opt["sample"])

    models =[]
    for env in opt["envs"]:
        models.append(load_lib(env))

    preds = []
    perfs = []
    for i in range(len(models)):
        y_pred = models[i].run((data_x, data_y), selection)
        preds.append(y_pred)
        perf = evaluate(data_y[selection == 0,1], y_pred[:,1], opt["measure"])
        perfs.append(perf)
        print("{}: {}".format(opt["envs"][i].split(".")[-1], perf))

    e_pred = np.mean(np.asarray(preds), axis=0)
    e_perf = evaluate(data_y[selection == 0], e_pred, opt["measure"])
    print("Ensemble: {}".format(e_perf))
