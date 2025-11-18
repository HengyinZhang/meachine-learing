# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    roc_curve, 
    auc,
    precision_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp 
# 导入中文字体管理
import matplotlib as mpl
from matplotlib import font_manager

# --------------------------------------------------------------------
# 0. 中文字体设置
# --------------------------------------------------------------------
try:
    # 尝试设置中文字体（例如 SimHei，如果您的系统安装了）
    mpl.rcParams['font.sans-serif'] = ['SimHei'] 
    mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
except:
    print("警告：未找到 SimHei 字体，请确保已安装或修改为系统中存在的中文宋体/黑体字体。")

# --------------------------------------------------------------------
# 0. 辅助函数：IV/KS/ROC 定义
# --------------------------------------------------------------------

def calculate_woe_iv(df, feature, target='loan_paid_back'):
    """计算单个特征的 WOE 和 IV 值，适用于二分类目标。"""
    # ... (WOE/IV计算代码不变) ...
    grouped = df.groupby(feature)[target].agg(
        total_count='count',
        bad_count=lambda x: (x == 0).sum(),  # 坏样本数（未还款）
        good_count=lambda x: (x == 1).sum()  # 好样本数（已还款）
    ).reset_index()
    
    total_bad = grouped['bad_count'].sum()
    total_good = grouped['good_count'].sum()
    
    epsilon = 0.5 
    grouped['pct_bad'] = (grouped['bad_count'] + epsilon) / (total_bad + epsilon * grouped.shape[0])
    grouped['pct_good'] = (grouped['good_count'] + epsilon) / (total_good + epsilon * grouped.shape[0])
    
    grouped['WOE'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['IV_contribution'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['WOE']
    iv = grouped['IV_contribution'].sum()
    
    return iv

def calculate_ks(y_true, y_pred_proba):
    """计算 Kolmogorov-Smirnov (KS) 统计量"""
    # ... (KS计算代码不变) ...
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (~data['y_true'].astype(bool)).cumsum() / (~data['y_true'].astype(bool)).sum()
    ks_value = (data['CGR'] - data['CBR']).abs().max()
    return ks_value

# **【修改后的函数】**：绘制训练集和验证集的 ROC 曲线
def plot_dual_roc_curve(y_true_train, y_probas_train, y_true_valid, y_probas_valid, model_name="模型"):
    """绘制训练集和验证集的 ROC 曲线，并显示 AUC"""
    
    # 1. 计算训练集 ROC
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_probas_train)
    auc_train = auc(fpr_train, tpr_train)
    
    # 2. 计算验证集 ROC
    fpr_valid, tpr_valid, _ = roc_curve(y_true_valid, y_probas_valid)
    auc_valid = auc(fpr_valid, tpr_valid)
    
    # 3. 绘制 ROC 曲线
    plt.figure(figsize=(10, 7))
    
    # 绘制训练集曲线
    plt.plot(fpr_train, tpr_train, 
             label=f'训练集 AUC = {auc_train:.4f}',
             color='darkblue', linewidth=2)
    
    # 绘制验证集曲线
    plt.plot(fpr_valid, tpr_valid, 
             label=f'验证集 AUC = {auc_valid:.4f}',
             color='darkred', linewidth=2, linestyle='--')
    
    # 绘制对角线 (随机分类器)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='随机分类器 (AUC = 0.50)')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title(f'{model_name} - 训练集 vs 验证集 ROC 曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------------
# 1-3. 数据加载、k-1 编码、IV筛选、划分和缩放 (保持不变)
# --------------------------------------------------------------------
print("----- 1. 数据加载与 k-1 编码 -----")

# 假设 data_train_origin 和 data_test_origin 已经加载
data_train_origin = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test_origin = pd.read_csv("E:/Code/python/pre-payback/test.csv")

grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train_lr = data_train_origin.copy()
data_test_lr = data_test_origin.copy()

data_train_lr['grade_subgrade'] = data_train_lr['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test_lr['grade_subgrade'] = data_test_lr['grade_subgrade'].astype(str).replace(grade_map, regex=True)

test_ids = data_test_lr['id']
data_train_lr = data_train_lr.drop('id', axis=1)
data_test_lr = data_test_lr.drop('id', axis=1)

CATEGORICAL_COLS = data_train_lr.select_dtypes(include=['object', 'category']).columns.tolist()

data_train_lr = pd.get_dummies(data_train_lr, columns=CATEGORICAL_COLS, drop_first=True)
data_test_lr = pd.get_dummies(data_test_lr, columns=CATEGORICAL_COLS, drop_first=True)

train_cols = set(data_train_lr.columns) - {'loan_paid_back'}
missing_in_test = list(train_cols - set(data_test_lr.columns))
for col in missing_in_test:
    data_test_lr[col] = 0
data_test_lr = data_test_lr[data_train_lr.drop('loan_paid_back', axis=1).columns]

print("\n----- 2. IV 计算与特征筛选 (IV >= 0.1) -----")
X_iv = data_train_lr.drop('loan_paid_back', axis=1)
iv_results = {}
for col in X_iv.columns:
    iv_value = calculate_woe_iv(data_train_lr, col, 'loan_paid_back')
    iv_results[col] = iv_value

iv_df = pd.DataFrame(list(iv_results.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
IV_THRESHOLD = 0.1
features_to_keep = iv_df[iv_df['IV'] >= IV_THRESHOLD]['Feature'].tolist()

X_filtered = data_train_lr[features_to_keep]
X_test_filtered = data_test_lr[features_to_keep]

print(f"最终保留特征总数: {len(features_to_keep)}")

print("\n----- 3. 划分数据集和特征缩放 -----")
y = data_train_lr['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(X_filtered, y, train_size=0.8, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test_filtered)


# --------------------------------------------------------------------
# 4. 逻辑回归模型训练
# --------------------------------------------------------------------
print("\n----- 4. 训练 IV 筛选后的逻辑回归模型 ----- ")

lr_model_iv = LogisticRegression(
    solver='liblinear', 
    random_state=123, 
    n_jobs=-1,              
    max_iter=1000           
)
lr_model_iv.fit(X_train_scaled, y_train)

# --------------------------------------------------------------------
# 5. 模型预测和评估指标计算
# --------------------------------------------------------------------

# **【新增】** 预测训练集概率
y_pred_proba_train_iv = lr_model_iv.predict_proba(X_train_scaled)[:, 1] 

# 预测验证集概率
y_pred_proba_valid_iv = lr_model_iv.predict_proba(X_valid_scaled)[:, 1]  

print("\n----- 5. 验证集性能指标 (IV 筛选后) -----")
# (AUC/ACC/Precision/KS 计算代码不变)
auc_score = roc_auc_score(y_valid, y_pred_proba_valid_iv)
y_pred_iv = lr_model_iv.predict(X_valid_scaled) 
accuracy = accuracy_score(y_valid, y_pred_iv)
precision = precision_score(y_valid, y_pred_iv, zero_division=1)
ks = calculate_ks(y_valid, y_pred_proba_valid_iv)

print(f"AUC: {auc_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks:.4f}")

# --------------------------------------------------------------------
# 6. 绘制 ROC 曲线 (训练集 vs 验证集)
# --------------------------------------------------------------------
print("\n----- 6. 绘制训练集和验证集的 ROC 曲线 -----")
plot_dual_roc_curve(
    y_train, 
    y_pred_proba_train_iv, 
    y_valid, 
    y_pred_proba_valid_iv, 
    model_name=f"逻辑回归 (IV >= {IV_THRESHOLD})"
)