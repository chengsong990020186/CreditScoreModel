# Project description

## CreditScoreModel 

## Installation
```
pip install CreditScoreModel

```

## Basic Usage
```
from CreditScoreModel.LogisticScoreCard import logistic_score_card #导入包

ls=logistic_score_card() #初始化参数

ls.fit(data) #模型训练

ls.score_card #制作好的评分卡

ls.logistic_auc_ks 模型准确性
```

