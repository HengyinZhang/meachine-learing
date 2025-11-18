# ================================================================
# train_xgb.py  XGBoost 最优参数模型  四指标及特征重要性分析
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from xgboost import XGBClassifier, plot_importance

# 设置全局字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ----------------------------------------------------------------
# 1. 读取数据
# ----------------------------------------------------------------
data_train = pd.read_csv(r"E:/Code/python/pre-payback/train.csv")
data_test  = pd.read_csv(r"E:/Code/python/pre-payback/test.csv")

# ----------------------------------------------------------------
# 2. 预处理
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
# 5. 使用指定最优参数的XGBoost模型
# ----------------------------------------------------------------
best_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=123,
    learning_rate=0.2,
    max_depth=5,
    n_estimators=200,
    subsample=0.8
)
best_xgb.fit(X_train, y_train)

y_pred_proba_best = best_xgb.predict_proba(X_valid)[:, 1]
y_pred_best       = best_xgb.predict(X_valid)

auc_best = roc_auc_score(y_valid, y_pred_proba_best)
acc_best = accuracy_score(y_valid, y_pred_best)
pre_best = precision_score(y_valid, y_pred_best, zero_division=0)
ks_best  = calculate_ks(y_valid, y_pred_proba_best)

# 获取并打印最优参数模型的特征重要性（兼容新版本XGBoost）
print("\n--------------- 最优参数模型特征重要性 ---------------")
booster = best_xgb.get_booster()
feature_importance_best = booster.get_score(fmap='', importance_type='weight')
sorted_importance = sorted(feature_importance_best.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance:
    print(f"{feature}: {importance}")

# 绘制特征重要性图（优化样式）
plt.figure(figsize=(10, 8))

# 绘制特征重要性，关闭内置网格
ax = plot_importance(
    best_xgb,
    importance_type='weight',
    title='Feature Importance (weight)',  # 更清晰的标题
    height=0.8,
    grid=False  # 关闭网格线
)

# 进一步美化：调整标题和坐标轴字体大小
ax.set_title('Feature Importance (weight)', fontsize=14, pad=20)  # 标题字体大小和间距
ax.set_xlabel('F Score', fontsize=12, labelpad=10)  # x轴标签
ax.set_ylabel('Features', fontsize=12, labelpad=10)  # y轴标签

# 调整刻度字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 6. 输出最优模型结果
# ----------------------------------------------------------------
print("\n--------------- XGBoost 最优参数模型 ---------------")
print(f"使用参数: learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.8")
print(f"AUC: {auc_best:.4f}")
print(f"Accuracy: {acc_best:.4f}")
print(f"Precision: {pre_best:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks_best:.4f}")