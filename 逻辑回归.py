# ================================================================
# 完整脚本：决策树默认 vs 网格搜索优化
# 输出：AUC / Accuracy / Precision / KS 对比 + 最优参数
# ================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

# ----------------------------------------------------------------
# 1. 读取数据
# ----------------------------------------------------------------
train_path = r"E:/Code/python/pre-payback/train.csv"
test_path  = r"E:/Code/python/pre-payback/test.csv"

data_train = pd.read_csv(train_path)
data_test  = pd.read_csv(test_path)

# ----------------------------------------------------------------
# 2. 预处理
# ----------------------------------------------------------------
# 2.1 grade_subgrade 映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade']  = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 2.2 其它非数值型 → 数值编码
def convert_to_numeric(df):
    for col in df.columns:
        if col not in {'grade_subgrade', 'id', 'loan_paid_back'} and \
           (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test  = convert_to_numeric(data_test)

# 2.3 强制转 grade_subgrade 为数值
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade']  = pd.to_numeric(data_test['grade_subgrade'])

# 2.4 去 ID
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
# 4. KS 计算函数（复用你之前实现）
# ----------------------------------------------------------------
def calculate_ks(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    return (data['CGR'] - data['CBR']).abs().max()

# ----------------------------------------------------------------
# 5. 优化前 —— 默认参数
# ----------------------------------------------------------------
dt_default = DecisionTreeClassifier(random_state=123)
dt_default.fit(X_train, y_train)

y_pred_proba_def = dt_default.predict_proba(X_valid)[:, 1]
y_pred_def       = dt_default.predict(X_valid)

auc_def = roc_auc_score(y_valid, y_pred_proba_def)
acc_def = accuracy_score(y_valid, y_pred_def)
pre_def = precision_score(y_valid, y_pred_def, zero_division=0)
ks_def  = calculate_ks(y_valid, y_pred_proba_def)

# ----------------------------------------------------------------
# 6. 优化后 —— 网格搜索
# ----------------------------------------------------------------
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=123),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
y_pred_proba_best = best_dt.predict_proba(X_valid)[:, 1]
y_pred_best       = best_dt.predict(X_valid)

auc_best = roc_auc_score(y_valid, y_pred_proba_best)
acc_best = accuracy_score(y_valid, y_pred_best)
pre_best = precision_score(y_valid, y_pred_best, zero_division=0)
ks_best  = calculate_ks(y_valid, y_pred_proba_best)

# ----------------------------------------------------------------
# 7. 对比输出
# ----------------------------------------------------------------
print("--------------- 决策树 默认参数 ---------------")
print(f"Parameters: {dt_default.get_params()}")
print(f"AUC: {auc_def:.4f}")
print(f"Accuracy: {acc_def:.4f}")
print(f"Precision: {pre_def:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks_def:.4f}")

print("\n--------------- 决策树 网格搜索最优 ---------------")
print(f"Best Parameters: {grid.best_params_}")
print(f"AUC: {auc_best:.4f}")
print(f"Accuracy: {acc_best:.4f}")
print(f"Precision: {pre_best:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks_best:.4f}")