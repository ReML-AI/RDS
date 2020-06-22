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
    def __init__(self, opt=None, exp="latest", data_file=None,target=None, X=None, Y=None, data_loader=None,
        sample_file=None, task="classification", measure="auc", model_classes=[], models=[], 
        learn="stochastic", iters=500, burnin=30, eps=3, ratio=0.6, delta=0.01, 
        weight_perf=1.0, weight_ratio=0.9, weight_iid=0.1, weight_kl=0.1, 
        learning_rate=0.001, hidden_dim=256, device="cuda", report_file=None, checkpoint_file=None, verbose=1):
        """ Reinforced Data Sampling - Initialization

        Args:
            opt (dict, optional): The option dict. Defaults to None.
            exp (str, optional): The experiment id. Defaults to "latest".
            data_file (str, optional): The path to data file. Defaults to None.
            target (array, optional): The target columns. Defaults to None.
            X (array, optional): The input variables. Defaults to None.
            Y (array, optional): The output variables. Defaults to None.
            data_loader (str, optional): The data loader class. Defaults to None.
            sample_file (str, optional): The output sample file. Defaults to None.
            task (str, optional): The task type (e.g., classification, regression). Defaults to "classification".
            measure (str, optional): The evaluation measure (e.g. cross_entropy, mse, auc , f1_micro, r2). Defaults to "auc".
            model_classes (list, optional): The list of model class names. Defaults to [].
            models (list, optional): The list of models. Defaults to [].
            learn (str, optional): The learning type (e.g., deterministic or stochastic). Defaults to "stochastic".
            iters (int, optional): The number of total iterations to run. Defaults to 500.
            burnin (int, optional): The number of total iterations to burn. Defaults to 30.
            eps (int, optional): The number of total episodes per epoch. Defaults to 3.
            ratio (float, optional): The sampling ratio. Defaults to 0.6.
            delta (float, optional): The sampling delta for return. Defaults to 0.01.
            weight_perf (float, optional): The weight factor for model performance. Defaults to 1.0.
            weight_ratio (float, optional): The weight factor for sampling ratio. Defaults to 0.9.
            weight_iid (float, optional): The weight factor for class ratios in classification. Defaults to 0.1.
            weight_kl (float, optional): The weight factor for distributional divergence in regression. Defaults to 0.1.
            learning_rate (float, optional): The initial learning rate. Defaults to 0.001.
            hidden_dim (int, optional): The number of nodes of the hidden layer in policy. Defaults to 256.
            device (str, optional): The device to run (e.g., cuda, cpu). Defaults to "cuda".
            report_file (str, optional): The report file. Defaults to None.
            checkpoint_file (str, optional): The checkpoint file. Defaults to None.
            verbose (int, optional): The verbose level (0 - no printing, 1 - printing). Defaults to 1.
        """
        self.opt = {}
        self.opt.update({"exp": exp, "data_file": data_file, "target": target, "X": X, "Y": Y, "data_loader": data_loader,
           "sample_file": sample_file, "task": task, "measure": measure, "model_classes": model_classes, "models": models, 
           "learn": learn, "iters": 500, "burnin": burnin, "eps": eps, "ratio": ratio, "delta": delta,
           "weight_perf": weight_perf, "weight_ratio": weight_ratio, "weight_iid": weight_iid, "weight_kl": weight_kl,
           "learning_rate": learning_rate, "hidden_dim": hidden_dim, "dev": device, 
           "report_file": report_file, "checkpoint_file": checkpoint_file, "verbose": verbose})
        if opt is not None:
            self.opt.update(opt)
        self.env = RDSEnv(self.opt)
        self.opt["device"] = torch.device(self.opt["dev"])
        self.verboseprint = print if self.opt["verbose"] > 0 else lambda *a, **k: None

    def initialize_policy(self, policy, x, y, epoch=30):
        self.verboseprint("Policy Init: ", end="")
        time_init = time.time()
        optimizer = optim.RMSprop(policy.parameters(), lr=self.opt["learning_rate"])
        for e in range(epoch):
            self.verboseprint("|", end="")
            probs = policy(x)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.verboseprint(" [{:0.2f}s]".format(time.time() - time_init))

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.Tensor(rewards).to(self.opt["device"]))
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (3 * rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss).to(self.opt["device"])
        self.verboseprint(" Loss={:0.6f}".format(policy_loss.data.cpu()), end="")
        self.verboseprint(" Reg_Ratio={:0.6f}".format(self.reg_ratio.data.cpu()), end="")
        (policy_loss + self.reg_ratio + self.reg_iid).backward()
        self.optimizer.step()

    def initialize_action(self, data_y):
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
        data_x, data_y = RDSUtil.load_data(self.opt["data_file"], self.opt["target"], self.opt["task"] == "classification", self.opt["data_loader"])

        self.policy = RDSPolicy(input_dim=data_x.shape[1] + data_y.shape[1], hidden_dim=self.opt["hidden_dim"])
        self.policy.to(self.opt["device"])
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.opt["learning_rate"])

        s = Variable(torch.cat([torch.from_numpy(data_x).float(), torch.from_numpy(data_y).float()], dim=1)).to(self.opt["device"])
        a = Variable(torch.from_numpy(self.initialize_action(data_y)).to(self.opt["device"]))
        y = torch.from_numpy(data_y).float().to(self.opt["device"])

        self.initialize_policy(self.policy, s, a)

        self.opt["epoch"] = 0
        self.opt["episode"] = 0
        self.opt["best_reward"] = 0.0
        self.opt["best_perf"] = -np.inf
        self.opt["best_action"] = None
        self.checkpoints = {"it_results": [], "epoch_results": []}
        while True:
            time_epoch = time.time()
            
            action_probs = self.policy(s)
            self.reg_ratio = self.opt["weight_ratio"] * RDSUtil.reg_ratio(action_probs, self.opt["ratio"])
            self.reg_iid = self.opt["weight_iid"] * RDSUtil.reg_iid(action_probs, y) if self.opt["task"] == "classification" else torch.tensor(0.0)
            self.checkpoints["epoch_results"].append([action_probs.data.cpu().numpy(), self.reg_ratio.item(), self.reg_iid.item()])

            log_probs = []
            rewards = []
            perfs = []
            self.env.reset()
            for _ in range(self.opt["eps"]):
                self.verboseprint("#{}".format(self.opt["episode"]), end="")
                time_episode = time.time()
                action, log_prob = self.sample_action(action_probs)
                self.verboseprint(" Sampling={}".format(np.sum(action)), end="")

                result = {"reg_ratio": self.reg_ratio.item(), "reg_iid": self.reg_iid.item()}
                result.update(self.env.get_reward((data_x, data_y), action))
                self.env.report(result)
                
                log_probs.append(log_prob)
                rewards.append(result["reward"])
                perfs.append(result["perf"])
                self.verboseprint(" Reward={:0.6f}".format(result["reward"]), end="")

                if self.opt["episode"] >= self.opt["burnin"] and result["perf"] > self.opt["best_perf"] and abs(result["ratio"] - self.opt["ratio"]) < self.opt["delta"]:
                    self.opt["best_perf"] = result["perf"]
                    self.opt["best_action"] = action
                    self.verboseprint(" !", end="")
                    self.save_sample(action)

                self.checkpoints["it_results"].append([result["reward"], action])
                self.opt["episode"] += 1

                self.verboseprint(" [{:0.2f}s]".format(time.time() - time_episode))

            self.verboseprint("*Epoch {}".format(self.opt["epoch"]), end="")

            self.update_policy(log_probs, rewards)

            self.verboseprint(" Reward={:0.6f}".format(np.mean(rewards)), end="")
            self.verboseprint(" Perf={:0.6f}".format(np.mean(perfs)), end="")
            
            if self.opt["episode"] >= self.opt["burnin"]:
                if self.opt["best_reward"] < np.mean(rewards):
                    self.opt["best_reward"] = np.mean(rewards)
                self.verboseprint(" Best_Reward={:0.6f}".format(self.opt["best_reward"]), end="")
            
            self.verboseprint(" [{:0.2f}s]".format(time.time() - time_epoch))
            self.opt["epoch"] += 1

            self.save_checkpoints()

            if self.opt["iters"] > 0 and self.opt["episode"] > self.opt["iters"]:
                return
        
        return self.opt["best_action"]

    def save_sample(self, action):
        try:
            if self.opt.get("sample_file") is not None:
                np.save(self.opt.get("sample_file"), action)
        except:
            pass

    def save_checkpoints(self):
        try:
            if self.opt.get("checkpoint_file") is not None:
                pickle.dump(self.checkpoints, open(self.opt.get("checkpoint_file"), "wb"))
        except:
            pass


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


class RDSEnv:
    def __init__(self, opt):
        self.opt = opt
        self.models = opt.get("models", [])
        for model in opt.get("model_classes", []):
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
        eval = self.evaluate(state, action)
        eval["reward"] = self.opt["weight_perf"] * eval["perf"]
        
        if self.opt["task"] == "regression":
            eval["reg_iidr"] = self.opt["weight_kl"] * RDSUtil.reg_iidr(state, action)
            eval["reward"] += eval["reg_iidr"]

        return eval

    def report(self, result):
        if self.opt.get("report_file") is None:
            return

        rpt = [result[k] for k in ["step", "sample", "ratio", "reward", "model", "perf", "reg_ratio"]]
        rpt += [result.get("reg_iid", 0), result.get("reg_iidr", 0)]
        rpt += result.get("perf_learners", [])

        # Reporting
        with open(self.opt.get("report_file"), "a") as f:
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
            return f1_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average="micro")
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
