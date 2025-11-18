# ================================================================
# train_xgb.py  XGBoost 默认 vs 优化  四指标对比
# ================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from xgboost import XGBClassifier   # 确保导入的是分类器

# ----------------------------------------------------------------
# 1. 读取数据
# ----------------------------------------------------------------
data_train = pd.read_csv(r"E:/Code/python/pre-payback/train.csv")
data_test  = pd.read_csv(r"E:/Code/python/pre-payback/test.csv")

# ----------------------------------------------------------------
# 2. 预处理（与你之前完全一致）
# ----------------------------------------------------------------
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade']  = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

def convert_to_numeric(df):
    for col in df.columns:
        if col not in {'grade_subgrade', 'id', 'loan_paid_back'} and \
           (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test  = convert_to_numeric(data_test)

data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade']  = pd.to_numeric(data_test['grade_subgrade'])

data_train = data_train.drop('id', axis=1)
data_test  = data_test.drop('id', axis=1)

# ----------------------------------------------------------------
# 3. 训练/验证划分
# ----------------------------------------------------------------
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123, stratify=y)

# ----------------------------------------------------------------
# 4. KS 计算函数
# ----------------------------------------------------------------
def calculate_ks(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    return (data['CGR'] - data['CBR']).abs().max()

# ----------------------------------------------------------------
# 5. 默认参数 XGBoost
# ----------------------------------------------------------------
xgb_default = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=123
)
xgb_default.fit(X_train, y_train)

y_pred_proba_def = xgb_default.predict_proba(X_valid)[:, 1]
y_pred_def       = xgb_default.predict(X_valid)

auc_def = roc_auc_score(y_valid, y_pred_proba_def)
acc_def = accuracy_score(y_valid, y_pred_def)
pre_def = precision_score(y_valid, y_pred_def, zero_division=0)
ks_def  = calculate_ks(y_valid, y_pred_proba_def)
'''
# ----------------------------------------------------------------
# 6. 网格搜索优化（精简参数，跑得快）
# ----------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=123
    ),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
y_pred_proba_best = best_xgb.predict_proba(X_valid)[:, 1]
y_pred_best       = best_xgb.predict(X_valid)

auc_best = roc_auc_score(y_valid, y_pred_proba_best)
acc_best = accuracy_score(y_valid, y_pred_best)
pre_best = precision_score(y_valid, y_pred_best, zero_division=0)
ks_best  = calculate_ks(y_valid, y_pred_proba_best)
'''
# ----------------------------------------------------------------
# 7. 对比输出
# ----------------------------------------------------------------
print("--------------- XGBoost 默认参数 ---------------")
print(f"Parameters: {xgb_default.get_params()}")
print(f"AUC: {auc_def:.4f}")
print(f"Accuracy: {acc_def:.4f}")
print(f"Precision: {pre_def:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks_def:.4f}")

'''
print("\n--------------- XGBoost 网格搜索最优 ---------------")
print(f"Best Parameters: {grid.best_params_}")
print(f"AUC: {auc_best:.4f}")
print(f"Accuracy: {acc_best:.4f}")
print(f"Precision: {pre_best:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks_best:.4f}")
'''