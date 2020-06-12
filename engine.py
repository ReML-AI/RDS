import pickle
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm
from util import *


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
        data_x, data_y = load_data(self.opt["data"], self.opt["target"], self.opt["task"] == "classification", self.opt["loader"])

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
            self.reg_ratio = self.opt["w_ratio"] * reg_ratio(action_probs, self.opt["ratio"])
            self.reg_iid = self.opt["w_iid"] * reg_iid(action_probs, y) if self.opt["task"] == "classification" else torch.tensor(0.0)
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
        self.models = []
        for model in self.opt["envs"]:
            self.models.append(load_lib(model))

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
                m_perf = evaluate(test_y, m_pred,self.opt["measure"])
                m_perfs.append(m_perf)

            e_pred = np.mean(np.asarray(m_preds), axis=0)
            eval["perf"] = evaluate(test_y, e_pred,self.opt["measure"])  # Ensemble performance
            eval["perf_learners"] = m_perfs
            eval["model"] = len(self.models)

        else:  # stochastic
            eval["model"] = self.steps[self.opt["epoch"] % len(self.steps)]
            m = self.models[eval["model"]]
            y_pred = m.run(state, action)
            eval["perf"] = evaluate(test_y, y_pred,self.opt["measure"])

        return eval

    def get_reward(self, state, action):
        tqdm.write("\rSampling: {}".format(np.sum(action)))

        eval = self.evaluate(state, action)
        eval["reward"] = self.opt["w_perf"] * eval["perf"]
        
        if self.opt["task"] == "regression":
            eval["reg_iidr"] = self.opt["w_kl"] * reg_iidr(state, action)
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

