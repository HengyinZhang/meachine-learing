import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ----------------------数据处理（完全保留原始逻辑）----------------------
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

# 转换grade_subgrade为数值型，保留测试集ID
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])
test_ids = data_test['id']  # 保存测试集ID用于输出
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# ----------------------数据划分（用于确定最优阈值）----------------------
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']

# 7:2:1划分（与之前一致，用于确定阈值）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=123
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=2/9, stratify=y_train_val, random_state=123
)

# 类别不平衡比例
pos_count = sum(y == 1)
neg_count = sum(y == 0)
imbalance_ratio = neg_count / pos_count


# ----------------------最佳参数模型训练----------------------
best_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'scale_pos_weight': imbalance_ratio,
    'max_depth': 3,
    'min_child_samples': 50,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.9,
    'reg_alpha': 5.0,
    'reg_lambda': 0.5,
    'random_state': 123,
    'n_jobs': -1
}

# 训练最终模型（用全部训练数据拟合，提升泛化能力）
final_model = lgb.LGBMClassifier(** best_params)
final_model.fit(X, y)  # 用完整训练集训练


# ----------------------确定最优阈值（基于验证集）----------------------
y_val_proba = final_model.predict_proba(X_val)[:, 1]
best_thresh = 0.5
best_f1_val = 0
for thresh in np.arange(0.1, 0.91, 0.05):
    y_val_pred = (y_val_proba >= thresh).astype(int)
    current_f1 = f1_score(y_val, y_val_pred)
    if current_f1 > best_f1_val:
        best_f1_val = current_f1
        best_thresh = thresh
print(f"最优预测阈值（基于验证集F1）：{best_thresh:.2f}")


# ----------------------预测测试集并输出结果----------------------
# 预测测试集概率和标签
test_proba = final_model.predict_proba(data_test)[:, 1]  # 正类概率
test_pred = (test_proba >= best_thresh).astype(int)  # 预测标签（0/1）

# 生成结果DataFrame（包含ID、预测标签、预测概率）
result = pd.DataFrame({
    'id': test_ids,  # 恢复测试集原始ID
    'loan_paid_back_pred': test_pred,  # 预测标签（是否还款）
    'probability': test_proba  # 预测为还款的概率
})

# 保存结果到CSV
result.to_csv("test_prediction_result.csv", index=False, encoding="utf-8-sig")
print("测试集预测结果已保存至：test_prediction_result.csv")
print("结果示例：")
print(result.head())