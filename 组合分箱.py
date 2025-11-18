import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import xgboost as xgb
import re
import catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 原有代码保留 --------------------------
# 数据读取
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 保留原始数据（用于WOE/IV计算和组合分箱）
data_train_origin = data_train.copy()
data_test_origin = data_test.copy()

# 非数值型变量转换（排除grade_subgrade）
def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])

# 数据清理（修正原代码drop列语法错误）
test_ids = data_test['id']
data_train = data_train.drop(['id', 'grade_subgrade'], axis=1)  # 修正：用列表传入多列
data_test = data_test.drop(['id', 'grade_subgrade'], axis=1)

# 数据集划分
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123, stratify=y)

# 五折交叉验证配置
N_SPLITS = 5
RANDOM_STATE = 123
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# -------------------------- 核心新增：组合分箱函数 --------------------------
def create_combination_bins(df, is_train=True):
    """
    新增2个组合特征并分箱：
    1. annual_income × debt_to_income_ratio（收入×债务比）
    2. loan_amount × credit_score（贷款金额×信用分）
    """
    df_comb = df.copy()
    
    # 1. 收入×债务比组合（体现还款能力）
    # 先对原始特征做区间划分（业务逻辑分箱）
    df_comb['income_bin'] = pd.cut(
        df_comb['annual_income'],
        bins=[0, 10000, 30000, 60000, 110551.7, np.inf],  # 参考文档异常值上限+业务阈值
        labels=['极低收入', '低收入', '中等收入', '高收入', '极高收入']
    )
    df_comb['debt_bin'] = pd.cut(
        df_comb['debt_to_income_ratio'],
        bins=[0, 0.1, 0.2, 0.3, 0.63],  # 参考文档债务比最大值+业务阈值
        labels=['低债务', '中低债务', '中高债务', '高债务']
    )
    # 组合分箱（直接拼接区间标签，形成组合特征）
    df_comb['income_debt_comb'] = df_comb['income_bin'].astype(str) + '+' + df_comb['debt_bin'].astype(str)
    
    # 2. 贷款金额×信用分组合（体现风险匹配度）
    df_comb['loan_bin'] = pd.cut(
        df_comb['loan_amount'],
        bins=[0, 5000, 20000, 40000, 48959.95],  # 参考文档贷款金额范围+业务阈值
        labels=['小额贷款', '中等贷款', '大额贷款', '超大额贷款']
    )
    df_comb['credit_bin'] = pd.cut(
        df_comb['credit_score'],
        bins=[395, 600, 680, 750, 849],  # 参考文档信用分范围+行业阈值
        labels=['低信用', '中信用', '高信用', '极高信用']
    )
    df_comb['loan_credit_comb'] = df_comb['loan_bin'].astype(str) + '+' + df_comb['credit_bin'].astype(str)
    
    # 训练集：删除中间分箱列；测试集保留组合特征用于后续建模
    drop_cols = ['income_bin', 'debt_bin', 'loan_bin', 'credit_bin']
    df_comb = df_comb.drop(drop_cols, axis=1)
    
    return df_comb

# -------------------------- 修正WOE/IV计算函数（适配组合特征） --------------------------
def calculate_woe_iv(df, feature, target, bins=10, special_values=None):
    if special_values is None:
        special_values = []
    
    # 组合特征为字符串类型，直接作为分箱
    if feature in ['income_debt_comb', 'loan_credit_comb']:
        df['bin'] = df[feature].astype(str)
    elif pd.api.types.is_numeric_dtype(df[feature]) and feature not in special_values:
        df['bin'] = pd.cut(df[feature], bins=bins, duplicates='drop')
    else:
        df['bin'] = df[feature].astype(str)
    
    # 计算WOE和IV（原有逻辑保留）
    bin_stats = df.groupby('bin')[target].agg(['count', 'sum'])
    bin_stats.columns = ['total', 'bad']
    bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
    total_bad = bin_stats['bad'].sum()
    total_good = bin_stats['good'].sum()
    
    bin_stats['bad_rate'] = bin_stats['bad'] / (total_bad + 1e-10)
    bin_stats['good_rate'] = bin_stats['good'] / (total_good + 1e-10)
    bin_stats['woe'] = np.log((bin_stats['good_rate'] + 1e-10) / (bin_stats['bad_rate'] + 1e-10))
    bin_stats['iv'] = (bin_stats['good_rate'] - bin_stats['bad_rate']) * bin_stats['woe']
    iv = bin_stats['iv'].sum()
    
    woe_dict = dict(zip(bin_stats.index, bin_stats['woe']))
    df.drop('bin', axis=1, inplace=True)
    
    return woe_dict, iv

def calculate_all_features_woe_iv(df, target, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = [target]
    else:
        exclude_cols = exclude_cols + [target]
    
    features = [col for col in df.columns if col not in exclude_cols]
    iv_list = []
    woe_maps = {}
    
    for feature in features:
        print(f"计算特征 {feature} 的WOE和IV...")
        # 组合特征强制用类别分箱（忽略bins参数）
        if feature in ['income_debt_comb', 'loan_credit_comb']:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target, bins=None)
        else:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target)
        iv_list.append({'feature': feature, 'iv': iv})
        woe_maps[feature] = woe_dict
    
    iv_df = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    return iv_df, woe_maps

# -------------------------- 执行组合分箱+WOE/IV计算 --------------------------
if __name__ == "__main__":
    # 1. 对原始训练数据做组合分箱（保留原始特征，不影响后续建模）
    data_train_comb = create_combination_bins(data_train_origin, is_train=True)
    
    # 2. 计算所有特征（含2个组合特征）的WOE和IV
    iv_df, woe_maps = calculate_all_features_woe_iv(
        df=data_train_comb,
        target='loan_paid_back',
        exclude_cols=['id']
    )
    
    # 3. 打印结果（重点关注组合特征的IV值）
    print("\n=== 所有特征IV值排序（含组合特征）===")
    print(iv_df[['feature', 'iv']].round(4))
    
    # 4. 保存结果（含组合特征IV）
    iv_df.to_csv('feature_iv_values_with_comb.csv', index=False)
    print("\nIV值（含组合特征）已保存到 feature_iv_values_with_comb.csv")
    
    # 5. 筛选有用特征（IV>0.1），包含组合特征
    useful_features = iv_df[iv_df['iv'] > 0.1]['feature'].tolist()
    print(f"\nIV>0.1的有用特征（含组合）: {useful_features}")
    
    # -------------------------- 可选：用组合特征建模（延续你的后续流程） --------------------------
    # 对建模数据添加组合特征（需同步处理训练/测试集）
    data_train_comb_model = create_combination_bins(data_train_origin, is_train=True)
    data_test_comb_model = create_combination_bins(data_test_origin, is_train=False)
    
    # 转换组合特征为数值型（适配模型）
    for col in ['income_debt_comb', 'loan_credit_comb']:
        data_train_comb_model[col] = pd.Categorical(data_train_comb_model[col]).codes
        data_test_comb_model[col] = pd.Categorical(data_test_comb_model[col]).codes
    
    # 后续建模可使用 data_train_comb_model 和 data_test_comb_model
    print("\n组合特征已添加到建模数据，可直接用于后续模型训练")