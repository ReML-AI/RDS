# RDS

Implementation of [Reinforced Data Sampling for Model Diversification](https://arxiv.org/pdf/2006.07100).

This work provides a method to learn how to sample data effectively on the search for useful models and meaningful insights. By employing diverse base learners such as neural networks, decision trees, or support vector machines, RDS aims to maximize the learning potentials and optimum allocation of data sampling to disentangle dataset shift and evidence ambiguity. In the hope of saving a massive amount of computational resources and time, we design RDS as a viable alternative to simple randomization and stratification in *train_test_split* for various machine learning tasks such as classification and regression.

### Machine Learning Tasks

This repository supports multiple machine learning tasks on multivariate, textual and visual data:

- **Binary Classification**
- **Multi-Class Classification**
- **Regression**

### Requirements

- numpy
- torch
- scikit-learn
- pandas
- tqdm

### Installation

```
pip install torchRDS
```

### Usage

```python
from torchRDS.RDS import RDS

trainer = RDS(data_file="datasets/madelon.csv", target=[0], task="classification", measure="auc", 
              model_classes=["models.MDL_RF", "models.MDL_MLP", "models.MDL_LR"], 
              learn="deterministic", ratio=0.7695, iters=100)
sample = trainer.train()

print("No of observations in training set: ", sum(sample))
```

### Real-World Use Cases

Please contact us if you want to be listed here for real-world competitions or use cases.

### Experiment Results

Experiments have been conducted on four datasets as the following.

| Dataset | Task                      | Challenge                             | Size of Data                     | Evaluation | Year |
| :------ | :------------------------ | :------------------------------------ | :------------------------------- | :--------- | ---: |
| MADELON | Binary Classification     | NIPS 2013 Feature Selection Challenge | 2,600 x 500 (multivariate)       | AUC        | 2003 |
| DR      | Regression                | Drug Reviews (Kaggle Hackathon)       | 215,063 x 6 (multivariate, text) | R^2        | 2018 |
| MNIST   | Multiclass Classification | Hand Written Digit Recognition        | 70,000 x 28 x 28 (image)         | Micro-F1   | 1998 |
| KLP     | Binary Classification     | Kalapa Credit Scoring Challenge       | 50,000 x 64 (multivariate, text) | AUC        | 2020 |

#### MADELON - Results

| Sampling   | \#Sample |      | Class Ratio |         |        LR |        RF |       MLP | Ensemble  |  Public   |
| :--------- | -------: | ---: | ----------: | ------: | --------: | --------: | --------: | :-------: | :-------: |
|            |    Train | Test |       Train |    Test |           |           |           |           |           |
| Preset     |     2000 |  600 |     1\.0000 | 1\.0000 |     .6019 | **.8106** |     .5590 |   .6783   |   .9063   |
| Random     |     2000 |  600 |       .9920 | 1\.0270 |     .5742 |     .7729 |     .5774 |   .6453   |   .9002   |
| Stratified |     2000 |  600 |     1\.0000 | 1\.0000 |     .5673 |     .7470 |     .6153 |   .6360   |   .8828   |
| RDS^{DET}  |     2001 |  599 |     1\.0375 |   .9137 | **.6192** |     .8050 | **.6228** | **.6973** |   .8915   |
| RDS^{STO}  |     2021 |  579 |     1\.0010 |   .9966 | **.6192** |     .8050 |     .6050 |   .6947   | **.9106** |

#### DR - Results

| Sampling  |  Train  |  Test  |   Ridge   |    MLP    |    CNN    | Ensemble  |  Public   |
| :-------- | :-----: | :----: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Preset    | 161,297 | 53,766 |   .4580   | **.5787** |   .7282   |   .6660   |   .7637   |
| Random    | 161,297 | 53,766 |   .4597   |   .4179   |   .7353   |   .6485   |   .7503   |
| RDS^{DET} | 162,070 | 52,993 |   .4646   |   .5776   |   .7355   | **.6692** | **.7649** |
| RDS^{STO} | 161,944 | 53,119 | **.4647** |   .5370   | **.7509** |   .6562   |   .7600   |

#### MNIST - Results

| Sampling   | \#Sample |       | Class Ratio |       |    LR     |    RF     |    CNN    | Ensemble  |  Public   |
| :--------- | :------: | :---: | :---------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: |
|            |  Train   | Test  |    Train    | Test  |           |           |           |           |           |
| Preset     |  60000   | 10000 |    .8571    | .1429 |   .9647   | **.9524** |   .9824   |   .9819   |   .9917   |
| Random     |  59500   | 10500 |    .8500    | .1500 |   .9603   |   .9465   |   .9779   |   .9768   |   .9914   |
| Stratified |  59500   | 10500 |    .8500    | .1500 | **.9625** |   .9510   |   .9795   |   .9792   |   .9901   |
| RDS^{DET}  |  59938   | 10062 |    .8562    | .1438 |   .9495   |   .9382   |   .9757   |   .9769   |   .9927   |
| RDS^{STO}  |  59496   | 10504 |    .8499    | .1501 |   .9583   |   .9486   | **.9851** | **.9830** | **.9931** |

#### KLP - Results

| Sampling   | \#Sample |       | Class Ratio |       |    LR     |    RF     |    MLP    | Ensemble  |  Public   |
| :--------- | :------: | :---: | :---------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: |
|            |  Train   | Test  |    Train    | Test  |           |           |           |           |           |
| Preset     |  30000   | 20000 |    .0165    | .0186 |   .5799   |   .5517   |   .5635   |   .5723   |   .5953   |
| Simple     |  30000   | 20000 |    .0169    | .0179 |   .5886   |   .5374   |   .5914   |   .5856   |   .6042   |
| Stratified |  30000   | 20000 |    .0173    | .0173 |   .5952   | **.5608** |   .5780   |   .5983   |   .6014   |
| RDS^{DET}  |  29999   | 20001 |    .0180    | .0163 | **.6045** |   .5350   |   .5802   |   .6057   |   .5362   |
| RDS^{STO}  |  30031   | 19969 |    .0172    | .0174 |   .5997   |   .5491   | **.6354** | **.6072** | **.6096** |

## Citing this work

Please consider citing us if this work is useful in your research:

```
@misc{nguyen2020reinforced,
    title={Reinforced Data Sampling for Model Diversification},
    author={Hoang D. Nguyen and Xuan-Son Vu and Quoc-Tuan Truong and Duc-Trong Le},
    year={2020},
    eprint={2006.07100},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## References

- Lee, S., Prakash, S.P.S., Cogswell, M., Ranjan, V., Crandall, D. and Batra, D., 2016. Stochastic multiple choice learning for training diverse deep ensembles. In *Advances in Neural Information Processing Systems* (pp. 2119-2127).
- Peng, M., Zhang, Q., Xing, X., Gui, T., Huang, X., Jiang, Y.G., Ding, K. and Chen, Z., 2019, July. Trainable undersampling for class-imbalance learning. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, pp. 4707-4714).
- Gong, Z., Zhong, P. and Hu, W., 2019. Diversity in machine learning. *IEEE Access*, *7*, pp.64323-64350.

