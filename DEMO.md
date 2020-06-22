# RDS

Implementation of [Reinforced Data Sampling for Model Diversification](https://arxiv.org/pdf/2006.07100).

### Demos

#### Madelon - Binary Classification

Binary Classification with Deterministic Ensemble

```
python rds.py --data datasets/madelon.csv --target 0 -id MDL_DET --learning deterministic --sampling-ratio 0.7695 --envs models.MDL_RF models.MDL_MLP models.MDL_LR
```

Binary Classification with Stochastic Choice

```
python rds.py --data datasets/madelon.csv --target 0 -id MDL_STO --learning stochastic --sampling-ratio 0.7695 --envs models.MDL_RF models.MDL_MLP models.MDL_LR
```

Evaluating with Public Benchmarking

```
python evaluator.py --data datasets/madelon.csv --target 0 --sample outputs/MDL_DET.npy --task classification --measure auc --envs models.MDL_PS
python evaluator.py --data datasets/madelon.csv --target 0 --sample outputs/MDL_STO.npy --task classification --measure auc --envs models.MDL_PS
```

#### Boston Housing - Regression

Regression with Deterministic Ensemble

```
python rds.py --data datasets/boston.csv --target 0 -id BOS_DET --task regression --learning deterministic --envs models.BOS_MLP models.BOS_Ridge models.BOS_SVM
```

Regression with Stochastic Choice

```
python rds.py --data datasets/boston.csv --target 0 -id BOS_STO --task regression --learning stochastic --envs models.BOS_MLP models.BOS_Ridge models.BOS_SVM
```

Evaluating with Ensemble Benchmarking

```
python evaluator.py --data datasets/boston.csv --target 0 --sample outputs/BOS_DET.npy --task regression --measure auc --envs models.BOS_MLP models.BOS_Ridge models.BOS_SVM
python evaluator.py --data datasets/boston.csv --target 0 --sample outputs/BOS_STO.npy --task regression --measure auc --envs models.BOS_MLP models.BOS_Ridge models.BOS_SVM
```

#### MNIST - Multi-class Classification

Regression with Deterministic Ensemble

```
python rds.py --data-loader datasets.MNIST -id MNIST_DET --task classification --learning deterministic --sampling-ratio 0.8572 --measure f1_micro --envs models.MNIST_CNN models.MNIST_RF models.MNIST_LR
```

Regression with Stochastic Choice

```
python rds.py --data-loader datasets.MNIST -id MNIST_STO --task classification --learning stochastic --sampling-ratio 0.8572 --measure f1_micro --envs models.MNIST_CNN models.MNIST_RF models.MNIST_LR
```

Evaluating with Ensemble Benchmarking

```
python evaluator.py --data-loader datasets.MNIST --sample outputs/MNIST_DET.npy --task classification --measure f1_micro --envs models.MNIST_CNN models.MNIST_RF models.MNIST_LR
python evaluator.py --data-loader datasets.MNIST --sample outputs/MNIST_STO.npy --task classification --measure f1_micro --envs models.MNIST_CNN models.MNIST_RF models.MNIST_LR
```
