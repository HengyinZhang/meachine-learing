# 导入必要的Python库
import pandas as pd
import numpy as np
import re  # 字符串处理
import matplotlib.pyplot as plt
import seaborn as sns

# 模型相关
from sklearn.tree import DecisionTreeClassifier  # 决策树（ID3/CART近似）
from sklearn.svm import SVC  # 支持向量机
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.ensemble import StackingClassifier  # 堆叠集成
import xgboost as xgb  # XGBoost
import lightgbm as lgb  # LightGBM
from catboost import CatBoostClassifier  # CatBoost

# 数据处理与评估
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.pipeline import Pipeline  # 管道流
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold  # 分层K折交叉验证（去重合并）
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc  # 评估指标（去重合并）
)

#数据处理
# 1. 读取训练和测试数据
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 2. 预处理: 修改grade_subgrade的映射 (A=1, B=2, 等)
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

#  将非数值型变量转换为数值型 (排除grade_subgrade)



def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)
# 强制转换grade_subgrade为数值
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])

test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# 5. 划分数据集， 8：2

X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=123)

#五折交叉验证
N_SPLITS = 5
RANDOM_STATE = 123

skf = StratifiedKFold(
    n_splits=N_SPLITS, 
    shuffle=True, 
    random_state=RANDOM_STATE
)