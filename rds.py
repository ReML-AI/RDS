import importlib
import pickle
import os
import re
import random
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class RDS:
    def __init__(self, opt):
        self.opt = opt
        self.env = LearningEnv(opt)
        self.opt["device"] = torch.device(self.opt["dev"])

    def initialise_policy(self, policy, x, y, epoch=30):
        optimizer = optim.RMSprop(policy.parameters(), lr=self.opt["lr"])
        for e in tqdm(range(epoch)):
            probs = policy(x)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.Tensor(rewards).to(self.opt["device"]))
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (3 * rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss).to(self.opt["device"])
        print("policy loss:{}".format(policy_loss.data.cpu()))
        print("reg ratio:{}".format(self.reg_ratio.data.cpu()))
        print("reg iid:{}".format(self.reg_iid.data.cpu()))
        (policy_loss + self.reg_ratio + self.reg_iid).backward()
        self.optimizer.step()

    def initialise_action(self, data_y):
        a = np.zeros((len(data_y), 2)).astype("float32")
        a[:, 0] = 1 - self.opt["ratio"] # Test sub-set
        a[:, 1] = self.opt["ratio"] # Train sub-set
        return a

    def sample_action(self, probs):
        if not isinstance(probs, Variable):
            probs = Variable(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.data.cpu().numpy(), m.log_prob(action).mean().to(self.opt["device"])

    def train(self):
        data_x, data_y = RDSUtil.load_data(self.opt["data"], self.opt["target"], self.opt["task"] == "classification", self.opt["loader"])

        self.policy = RDSPolicy(input_dim=data_x.shape[1] + data_y.shape[1], hidden_dim=self.opt["hdim"])
        self.policy.to(self.opt["device"])
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.opt["lr"])

        s = Variable(torch.cat([torch.from_numpy(data_x).float(), torch.from_numpy(data_y).float()], dim=1)).to(self.opt["device"])
        a = Variable(torch.from_numpy(self.initialise_action(data_y)).to(self.opt["device"]))
        y = torch.from_numpy(data_y).float().to(self.opt["device"])

        self.initialise_policy(self.policy, s, a)

        self.opt["epoch"] = 0
        self.opt["episode"] = 0
        self.opt["best_reward"] = 0.0
        self.opt["best_perf"] = -np.inf
        self.checkpoints = {"it_results": [], "epoch_results": []}
        while True:
            epoch_time = time.time()
            
            action_probs = self.policy(s)
            self.reg_ratio = self.opt["w_ratio"] * RDSUtil.reg_ratio(action_probs, self.opt["ratio"])
            self.reg_iid = self.opt["w_iid"] * RDSUtil.reg_iid(action_probs, y) if self.opt["task"] == "classification" else torch.tensor(0.0)
            self.checkpoints["epoch_results"].append([action_probs.data.cpu().numpy(), self.reg_ratio.item(), self.reg_iid.item()])

            log_probs = []
            rewards = []
            perfs = []
            self.env.reset()
            for _ in tqdm(range(self.opt["eps"])):
                action, log_prob = self.sample_action(action_probs)
                result = {"reg_ratio": self.reg_ratio.item(), "reg_iid": self.reg_iid.item()}
                result.update(self.env.get_reward((data_x, data_y), action))
                self.env.report(result)
                
                log_probs.append(log_prob)
                rewards.append(result["reward"])
                perfs.append(result["perf"])

                if self.opt["episode"] >= self.opt["burnin"] and result["perf"] > self.opt["best_perf"] and abs(result["ratio"] - self.opt["ratio"]) < self.opt["delta"]:
                    self.opt["best_perf"] = result["perf"]
                    np.save("{}/{}.npy".format(self.opt["out"], self.opt["exp"]), action)

                self.checkpoints["it_results"].append([result["reward"], action])
                self.opt["episode"] += 1

            self.update_policy(log_probs, rewards)

            print("Reward: {} in epoch: {} ".format(np.mean(rewards), self.opt["epoch"]))
            print("Perf: {} in epoch: {} ".format(np.mean(perfs), self.opt["epoch"]))
            
            if self.opt["episode"] >= self.opt["burnin"]:
                if self.opt["best_reward"] < np.mean(rewards):
                    self.opt["best_reward"] = np.mean(rewards)
                print("Best reward: {}".format(self.opt["best_reward"]))
            
            print("Epoch:{} Time:{:0.2f}s".format(self.opt["epoch"], time.time() - epoch_time))
            self.opt["epoch"] += 1

            self.save_checkpoints()

            if self.opt["iters"] > 0 and self.opt["episode"] > self.opt["iters"]:
                return

    def save_checkpoints(self):
        try:
            self.checkpoints["pkl_file"] = "{}/{}.pkl".format(self.opt["out"], self.opt["exp"])
            pickle.dump(self.checkpoints, open(self.checkpoints["pkl_file"], "wb"))
        except:
            print("Saving error")


class RDSPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, activations=None):
        super(RDSPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(0))
        out = out.view(out.size()[1], out.size(2))
        out = nn.Softmax(dim=1)(self.linear(out))
        return out


class LearningEnv:
    def __init__(self, opt):
        self.opt = opt
        self.models = opt.get("models", [])
        for model in opt.get("envs", []):
            self.models.append(RDSUtil.load_lib(model))

    def reset(self):
        if self.opt["learn"] == "stochastic" and self.opt["epoch"] % len(self.models) == 0:
            self.steps = random.sample(list(range(len(self.models))), len(self.models))  # stochastic sampling

    def evaluate(self, state, action):
        eval = {}
        _, data_y = state
        test_y = data_y[action == 0]

        eval["step"] = self.opt["episode"]
        eval["sample"] = np.sum(action)
        eval["ratio"] = np.sum(action) / len(action)

        if self.opt["learn"] == "deterministic":
            m_preds = []
            m_perfs = []
            for i in range(len(self.models)):
                m_pred = self.models[i].run(state, action)
                m_preds.append(m_pred)
                m_perf = RDSUtil.evaluate(test_y, m_pred,self.opt["measure"])
                m_perfs.append(m_perf)

            e_pred = np.mean(np.asarray(m_preds), axis=0)
            eval["perf"] = RDSUtil.evaluate(test_y, e_pred,self.opt["measure"])  # Ensemble performance
            eval["perf_learners"] = m_perfs
            eval["model"] = len(self.models)

        else:  # stochastic
            eval["model"] = self.steps[self.opt["epoch"] % len(self.steps)]
            m = self.models[eval["model"]]
            y_pred = m.run(state, action)
            eval["perf"] = RDSUtil.evaluate(test_y, y_pred,self.opt["measure"])

        return eval

    def get_reward(self, state, action):
        tqdm.write("\rSampling: {}".format(np.sum(action)))

        eval = self.evaluate(state, action)
        eval["reward"] = self.opt["w_perf"] * eval["perf"]
        
        if self.opt["task"] == "regression":
            eval["reg_iidr"] = self.opt["w_kl"] * RDSUtil.reg_iidr(state, action)
            eval["reward"] += eval["reg_iidr"]

        return eval

    def report(self, result):
        rpt = [result[k] for k in ["step", "sample", "ratio", "reward", "model", "perf", "reg_ratio"]]
        rpt += [result.get("reg_iid", 0), result.get("reg_iidr", 0)]
        rpt += result.get("perf_learners", [])

        # Reporting
        with open("{}/{}.txt".format(self.opt["out"], self.opt["exp"]), "a") as f:
            f.write(",".join(map(str, rpt)))
            f.write("\n")


class RDSUtil:
    @staticmethod
    def evaluate(y, y_pred, measure=None):
        if measure == "auc":
            return roc_auc_score(y, y_pred)
        elif measure == "mse":
            return -mean_squared_error(y.ravel(), y_pred.ravel())
        elif measure == "cross_entropy":
            return -log_loss(y, y_pred)
        elif measure == "f1_micro":
            return f1_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average='micro')
        else:
            return r2_score(y.ravel(), y_pred.ravel())

    @staticmethod
    def encode_onehot(data_y):
        return np.eye(np.max(data_y) + 1)[data_y.astype("int32")]

    @staticmethod
    def load_data(file, target, to_onehot = False, loader=None):
        if loader is not None:
            lib = RDSUtil.load_lib(loader)
            return lib.load()

        df = pd.read_csv(file)
        df_x = df.drop(df.columns[target], axis=1)
        df_y = df.iloc[:, target]

        data_x = df_x.to_numpy()
        data_y = df_y.to_numpy()

        if to_onehot:
            data_y = RDSUtil.encode_onehot(data_y[:, 0])

        return data_x, data_y

    @staticmethod
    def load_lib(model_name, path = None):
        module_name = model_name if path is None else re.sub(r"[\/\\]", ".", os.path.splitext(path)[0])
        module = importlib.import_module(module_name)
        model = getattr(module, re.search(r"[^\.]+$", model_name).group())
        return model()

    @staticmethod
    def cumulative_kl(x, y, fraction=0.5):
        # Implementation based on Pérez-Cruz (2008) - Kullback-Leibler divergence estimation of continuous distributions
        def ecdf(x):
            x = np.sort(x)
            u, c = np.unique(x, return_counts=True)
            n = len(x)
            y = (np.cumsum(c) - 0.5) / n

            def interpolate_(x_):
                yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
                return yinterp

            return interpolate_

        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        ex = np.min(dx)
        ey = np.min(dy)
        e = np.min([ex, ey]) * fraction
        n = len(x)
        m = len(y)
        P = ecdf(x)
        Q = ecdf(y)
        KL = (1.0 / n) * np.sum(np.log((P(x) - P(x - e)) / (Q(x) - Q(x - e))))
        return KL

    @staticmethod
    def reg_ratio(action_probs, ratio):
        return torch.abs(torch.mean(action_probs[:, 1]) - ratio)

    @staticmethod
    def reg_iid(action_probs, y):
        y_prob = torch.matmul(action_probs.transpose(0, 1), y)
        p_train = y_prob[1] / torch.sum(y_prob[1])
        p_test = y_prob[0] / torch.sum(y_prob[0])
        return torch.sum(p_test * torch.log(p_test / p_train))

    @staticmethod
    def reg_iidr(state, action):
        data_x, data_y = state
        train_y = data_y[action == 1].ravel()
        test_y = data_y[action == 0].ravel()

        train_kl = RDSUtil.cumulative_kl(train_y, data_y)
        test_kl = RDSUtil.cumulative_kl(test_y, data_y)
        objective_kl = train_kl + test_kl
        return np.exp(-objective_kl)

    @staticmethod
    def simplerandom(data_x, data_y, ratio):
        idx_train, idx_test = train_test_split(np.asarray(range(0, len(data_x))), train_size=ratio)
        idx = np.zeros(len(data_x))
        idx[idx_train] = 1
        return idx.astype(int)

    @staticmethod
    def stratification(data_x, data_y, ratio):
        idx_train, idx_test = train_test_split(np.asarray(range(0, len(data_x))), train_size=ratio, stratify=data_y)
        idx = np.zeros(len(data_x))
        idx[idx_train] = 1
        return idx.astype(int)

    @staticmethod
    def sequence(data_x, data_y, ratio):
        idx = np.zeros(len(data_x))
        idx[: int(ratio * len(data_x))] = 1
        return idx.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforced Data Sampling')
    parser.add_argument('--data', type=str, default='datasets/madelon.csv',
                help='path to dataset file (default: datasets/madelon.csv)')
    parser.add_argument('--target', type=int, nargs='+', default=[0],
                help='column indexes for target variables (default: 0 - first column)')
    parser.add_argument('--data-loader', '-loader', type=str, default=None,
                help='the loader class for a dataset e.g., datasets.MNIST (default: None)')
    parser.add_argument('--exp-id', '-id', type=str, default=None,
                help='experiment id (default: timestamp)')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs/',
                help='output directory (default: outputs/)')
    parser.add_argument('--task', '-t', type=str, default='classification',
                help='task type: classification, regression (default: classification)')
    parser.add_argument('--measure', '-m', type=str, default='cross_entropy',
                help='measure type: cross_entropy (binary, multiclass), mse (regression), \
                auc (binary), f1_micro(classification), r2 (regression) (default: cross_entropy)')
    parser.add_argument('--envs', '-e', type=str, nargs='+', default=['models.LR'],
                help='learning environment (default: models.LR)')
    parser.add_argument('--learning', '-l', type=str, default='deterministic',
                help='deterministic or stochastic (default: deterministic)')
    parser.add_argument('--iterations', '-iters', type=int, default=1000,
                help='number of total iterations to run (default: 1000)')
    parser.add_argument('--burn-in-iterations', '-burnin', type=int, default=30,
                help='number of total burn-in iterations to run (default: 30)')
    parser.add_argument('--episodes', '-eps', type=int, default=3,
                help='number of total episodes to run (default: 3)')
    parser.add_argument('--sampling-ratio', '-ratio', type=float, default=0.6,
                help='sampling ratio (default: 0.6)')
    parser.add_argument("--delta", "-d", type=float, default=0.01,
                help="sampling delta for saving (default: 0.01)")
    parser.add_argument('--weight-perf', '-wp', type=float, default=1.0,
                help='weight factor for model performance (default: 1.0)')
    parser.add_argument('--weight-ratio', '-wr', type=float, default=0.9,
                help='weight factor for sampling ratio (default: 0.9)')
    parser.add_argument('--weight-iid', '-wi', type=float, default=0.1,
                help='weight factor for class ratios in classification (default: 0.1)')
    parser.add_argument('--weight-kl', '-wk', type=float, default=0.1,
                help='weight factor for distributional divergence in regression (default: 0.1)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                help='initial learning rate (default: 0.001)')
    parser.add_argument('--hidden-dim', '-hd', type=int, default=256,
                help='number of nodes in the hidden layer (default: 256)')
    parser.add_argument('--device', '-dev', type=str, default="cuda",
                help='device to run: cuda, cpu (default: cuda)')

    args = parser.parse_args()
    opt = {'exp': str(int(time.time())) if args.exp_id is None else args.exp_id,
           'data': args.data,
           'target': args.target,
           'loader': args.data_loader,
           'out': args.output_dir,
           'task': args.task,
           'measure': args.measure,
           'envs': args.envs,
           'learn': args.learning,
           'iters': args.iterations,
           'burnin': args.burn_in_iterations,
           'eps': args.episodes if args.episodes > 0 else len(args.envs),
           'ratio': args.sampling_ratio,
           "delta": args.delta,
           'w_perf': args.weight_perf,
           'w_ratio': args.weight_ratio,
           'w_iid': args.weight_iid,
           'w_kl': args.weight_kl,
           'lr': args.learning_rate,
           'hdim': args.hidden_dim,
           "dev": args.device
           }

    print("Reinforced Data Sampling")
    print("\n".join("{}:     \t{}".format(k, v) for k, v in opt.items()))
    trainer = RDS(opt)
    trainer.train()
