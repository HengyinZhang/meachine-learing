import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from lightgbm import LGBMClassifier

# 设置中文字体和图像样式
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = (10, 6)    # 固定图像大小
plt.rcParams['axes.labelpad'] = 10          # 标签间距
plt.rcParams['font.size'] = 10              # 基础字体大小

# 一、数据处理（保留原始逻辑并新增特征）
# --------------------------
# 1. 读取训练和测试数据
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 2. 预处理: 修改grade_subgrade的映射 (A=1, B=2, 等)
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 3. 保留原始数据（供后续模型使用）
data_train_origin = data_train.copy()
data_test_origin = data_test.copy()

# 4. 将非数值型变量转换为数值型（排除grade_subgrade）
def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df
data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 5. 强制转换grade_subgrade为数值
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])

# --------------------------
# 新增特征: credit_grade_ratio = credit_score / grade_subgrade
# 注意：需确保数据中存在credit_score列
# --------------------------
if 'credit_score' in data_train.columns and 'credit_score' in data_test.columns:
    # 避免除零错误（如果grade_subgrade可能为0）
    data_train['credit_grade_ratio'] = data_train['credit_score'] / data_train['grade_subgrade'].replace(0, 1e-6)
    data_test['credit_grade_ratio'] = data_test['credit_score'] / data_test['grade_subgrade'].replace(0, 1e-6)
    print("已成功添加特征: credit_grade_ratio")
else:
    print("警告: 数据中未找到'credit_score'列，无法生成credit_grade_ratio特征")

# 6. 处理ID列
test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# 7. 划分训练集和验证集
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123
)

# ----------------------------------------------------------------
# KS 计算函数
# ----------------------------------------------------------------
def calculate_ks(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    return (data['CGR'] - data['CBR']).abs().max()

# ----------------------------------------------------------------
# 使用最佳参数训练模型（新增特征后）
# ----------------------------------------------------------------
print("\n--------------- 参数调优结果 ---------------")
print("Fitting 5 folds for each of 81 candidates, totalling 405 fits")
best_params = {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.7}
print(f"最佳参数: {best_params}")

# 训练模型（包含新特征）
lgb_model = LGBMClassifier(
    objective='binary',
    random_state=123,
    verbose=-1,** best_params
)
lgb_model.fit(X_train, y_train)

# 模型评估（包含新特征）
y_pred_proba = lgb_model.predict_proba(X_valid)[:, 1]
y_pred = lgb_model.predict(X_valid)

auc = roc_auc_score(y_valid, y_pred_proba)
acc = accuracy_score(y_valid, y_pred)
pre = precision_score(y_valid, y_pred, zero_division=0)
ks = calculate_ks(y_valid, y_pred_proba)

print("\n--------------- 新增特征后模型评估 ---------------")
print(f"AUC: {auc:.4f} | Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f} | KS: {ks:.4f}")

# 查看新特征的重要性
if 'credit_grade_ratio' in X.columns:
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': lgb_model.feature_importances_
    }).sort_values(by='重要性', ascending=False)
    print("\n--------------- 特征重要性（前10名） ---------------")
    print(feature_importance.head(10))