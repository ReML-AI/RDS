import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from rds import RDSUtil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDS Sampler")
    parser.add_argument("--data", type=str, default="datasets/madelon.csv", 
                        help="path to dataset file (default: datasets/madelon.csv)")
    parser.add_argument("--target", type=int, nargs="+", default=[0], 
                        help="column indexes for response variables (default: 0 - first column)")
    parser.add_argument('--data-loader', '-loader', type=str, default=None,
                        help='the loader class for a dataset e.g., datasets.MNIST (default: None)')
    parser.add_argument("--sample", type=str, default="samples/MDL_TEST.npy", 
                        help="path to sampling file (default: samples/MDL_TEST.npy)")
    parser.add_argument('--task', '-t', type=str, default='classification',
                        help='task type: classification, regression (default: classification)')
    parser.add_argument("--sample-type", "-s", type=str, default="random", 
                        help="sample type: random, stratified, sequence (default: random)")
    parser.add_argument("--sampling-ratio", "-ratio", type=float, default=0.6, 
                        help="sampling ratio (default: 0.6)")
    args = parser.parse_args()
    opt = {
        "data": args.data,
        "target": args.target,
        'loader': args.data_loader,
        "sample": args.sample,
        "task": args.task,
        "type": args.sample_type,
        "ratio": args.sampling_ratio
    }
    print("Simple Sampler")
    print("\n".join("{}:     \t{}".format(k, v) for k, v in opt.items()))

    data_x, data_y = RDSUtil.load_data(opt["data"], opt["target"], opt["task"] == "classification", opt["loader"])

    if opt["type"] == "sequence":
        selection = RDSUtil.sequence(data_x, data_y, opt["ratio"])
    elif opt["type"] == "stratified":
        selection = RDSUtil.stratification(data_x, data_y, opt["ratio"])
    else: #random
        selection = RDSUtil.simplerandom(data_x, data_y, opt["ratio"])
    np.save(opt["sample"], selection)
