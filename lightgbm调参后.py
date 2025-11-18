import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ----------------------数据处理（完全保留你的原始逻辑）----------------------
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 处理grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 非数值变量转数值（排除grade_subgrade）
def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df
data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 转换grade_subgrade为数值型，删除id列
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])
test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# ----------------------数据划分（保持7:2:1比例）----------------------
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']

# 先划分训练集（90%）和测试集（10%）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=123
)

# 再从90%中划分训练集（7/9≈78%）和验证集（2/9≈22%）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=2/9, stratify=y_train_val, random_state=123
)

# 类别不平衡比例（用于scale_pos_weight）
pos_count = sum(y == 1)
neg_count = sum(y == 0)
imbalance_ratio = neg_count / pos_count


# ----------------------使用最佳参数训练最终模型----------------------
# 最佳参数（基于调优结果）
best_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,  # 固定迭代轮数（可根据学习率调整，此处用默认值）
    'scale_pos_weight': imbalance_ratio,  # 不平衡处理
    
    # 树结构最佳参数
    'max_depth': 3,
    'min_child_samples': 50,
    
    # 采样最佳参数
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.9,
    
    # 正则化最佳参数
    'reg_alpha': 5.0,
    'reg_lambda': 0.5,
    
    'random_state': 123,
    'n_jobs': -1
}

# 训练最终模型
final_model = lgb.LGBMClassifier(** best_params)
final_model.fit(X_train, y_train)  # 兼容旧版本，仅用训练集拟合


# ----------------------计算四个评估指标（测试集）----------------------
# 预测概率和类别（使用最优阈值，沿用调优时的最佳阈值逻辑）
y_test_proba = final_model.predict_proba(X_test)[:, 1]

# 搜索最优阈值（基于验证集，保持与调优逻辑一致）
y_val_proba = final_model.predict_proba(X_val)[:, 1]
best_thresh = 0.5
best_f1_val = 0
for thresh in np.arange(0.1, 0.91, 0.05):
    y_val_pred = (y_val_proba >= thresh).astype(int)
    current_f1 = f1_score(y_val, y_val_pred)
    if current_f1 > best_f1_val:
        best_f1_val = current_f1
        best_thresh = thresh

# 测试集预测结果
y_test_pred = (y_test_proba >= best_thresh).astype(int)

# 计算四个指标
auc = roc_auc_score(y_test, y_test_proba)  # AUC（用概率计算）
accuracy = accuracy_score(y_test, y_test_pred)  # 准确率
precision = precision_score(y_test, y_test_pred)  # 精确率
# KS值计算：KS = max(TPR - FPR)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ks = max(tpr - fpr)  # KS值


# ----------------------输出结果----------------------
print("===== 最佳参数模型评估指标（测试集）=====")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"KS: {ks:.4f}")