# 导入必要的Python库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
import re

# --- 1. 数据读取 ---
# 请根据您的文件路径修改此处
FILE_PATH = "E:/Code/python/pre-payback/"
data_train = pd.read_csv(FILE_PATH + "train.csv")
data_test = pd.read_csv(FILE_PATH + "test.csv")

# --- 2. 预处理和特征工程 ---
print("--- 1. 数据预处理和特征工程开始 ---")

# 2.1 修改 grade_subgrade 的映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 2.2 特征工程：创建新变量 (AD 和 LC)
print("   - 创建新特征 AD 和 LC...")
data_train['AD'] = data_train['annual_income'] * data_train['debt_to_income_ratio']
data_test['AD'] = data_test['annual_income'] * data_test['debt_to_income_ratio']

data_train['LC'] = data_train['loan_amount'] * data_train['credit_score']
data_test['LC'] = data_test['loan_amount'] * data_test['credit_score']

# 2.3 删除 grade_subgrade 列
print("   - 删除特征 grade_subgrade...")
data_train.drop('grade_subgrade', axis=1, inplace=True)
data_test.drop('grade_subgrade', axis=1, inplace=True)

# 2.4 保留原始数据副本用于 CatBoost 和 WOE/IV 计算
data_train_origin = data_train.copy() 
data_test_origin  = data_test.copy()

# 2.5 将非数值型变量转换为数值型 (使用类别编码)
def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 2.6 存储 ID 并删除 ID 列
test_ids = data_test['id']
data_train.drop('id', axis=1, inplace=True)
data_test.drop('id', axis=1, inplace=True)

print("--- 1. 数据预处理和特征工程完成 ---")

# --- 3. 划分数据集 (用于模型训练) ---
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123, stratify=y)

# 五折交叉验证设置
N_SPLITS = 5
RANDOM_STATE = 123
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# --- 4. WOE/IV 计算函数定义 ---
def calculate_woe_iv(df, feature, target, bins=10, special_values=None):
    """ 计算单个特征的WOE和IV值 """
    if special_values is None:
        special_values = []
    
    if pd.api.types.is_numeric_dtype(df[feature]) and feature not in special_values and feature != target:
        try:
            # 数值型特征分箱
            df['bin'] = pd.cut(df[feature], bins=bins, duplicates='drop')
        except ValueError:
            # 如果分箱失败（例如特征值唯一），则视为类别
            df['bin'] = df[feature].astype(str)
    else:
        # 类别型特征或特殊数值直接使用其值作为分箱
        df['bin'] = df[feature].astype(str)
    
    # 计算每个分箱的好坏样本数
    bin_stats = df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
    bin_stats.columns = ['total', 'bad']
    bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
    
    # 计算总体好坏样本数
    total_bad = bin_stats['bad'].sum()
    total_good = bin_stats['good'].sum()
    
    # 避免除以0，添加微小值 (1e-6)
    epsilon = 1e-6 
    bin_stats['bad_rate'] = bin_stats['bad'] / (total_bad + epsilon)
    bin_stats['good_rate'] = bin_stats['good'] / (total_good + epsilon)
    
    # 计算WOE
    bin_stats['woe'] = np.log((bin_stats['good_rate'] + epsilon) / (bin_stats['bad_rate'] + epsilon))
    
    # 计算IV
    bin_stats['iv'] = (bin_stats['good_rate'] - bin_stats['bad_rate']) * bin_stats['woe']
    iv = bin_stats['iv'].sum()
    
    # 清理临时列
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    
    return None, iv

def calculate_all_features_woe_iv(df, target, exclude_cols=None):
    """ 计算数据集中所有特征的WOE和IV值 """
    if exclude_cols is None:
        exclude_cols = [target]
    else:
        exclude_cols = exclude_cols + [target]
    
    features = [col for col in df.columns if col not in exclude_cols]
    iv_list = []
    
    for feature in features:
        _, iv = calculate_woe_iv(df.copy(), feature, target)
        iv_list.append({'feature': feature, 'iv': iv})
    
    # 按IV值排序
    iv_df = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    return iv_df

# --- 5. 执行 IV/WOE 计算、特征筛选和热力图绘制 ---

print("\n--- 2. WOE/IV 重新计算与特征筛选 ---")
if 'loan_paid_back' in data_train_origin.columns:
    # 重新计算 IV 值
    iv_df = calculate_all_features_woe_iv(
        df=data_train_origin,
        target='loan_paid_back',
        exclude_cols=['id'] 
    )
    
    print("\n特征IV值排序:")
    print(iv_df)
    
    # *** 关键修改: 筛选 IV > 0.01 的特征 ***
    IV_THRESHOLD = 0.01 
    useful_features = iv_df[iv_df['iv'] > IV_THRESHOLD]['feature'].tolist()
    if 'loan_paid_back' in useful_features:
        useful_features.remove('loan_paid_back')
    
    print(f"\nIV > {IV_THRESHOLD} 的有用特征 ({len(useful_features)}个): {useful_features}")
    
    # --- 6. 绘制相关系数热力图 ---
    
    # 目标变量加入特征列表
    plot_cols = useful_features + ['loan_paid_back']
    
    try:
        # 从数值化的训练数据 (data_train) 中选择这些特征
        data_for_heatmap = data_train[plot_cols].copy() 
        
        # 计算相关系数矩阵
        correlation_matrix = data_for_heatmap.corr()
        
        print("\n--- 3. 绘制相关系数热力图 ---")
        # 动态调整图形大小以适应特征数量
        fig_size = max(8, len(plot_cols) * 0.8)
        plt.figure(figsize=(fig_size, fig_size))
        
        sns.heatmap(
            correlation_matrix, 
            annot=True,          # 显示数值
            fmt=".2f",           # 格式化为两位小数
            cmap='coolwarm',     # 颜色映射
            linewidths=.5,       # 线宽
            cbar=True,           # 显示颜色条
            annot_kws={"size": 8} # 调整注解字体大小
        )
        
        plt.title(f'Correlation Heatmap ')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    except KeyError as e:
        print(f"\n❌ 错误: 在数值化后的训练集 (data_train) 中找不到列 {e}。")
        print("请检查特征工程和数值转换步骤是否正确，确保所有 IV > 0.01 的特征都在 data_train 中。")
        
else:
    print("\n❌ 错误: 目标变量 'loan_paid_back' 未在 data_train_origin 中找到。请检查数据读取和变量命名。")