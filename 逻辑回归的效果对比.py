import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# -------------------------- 数据读取与预处理 --------------------------
# 请确保文件路径正确
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 保留原始数据（用于组合特征和 WOE/IV 计算）
data_train_origin = data_train.copy()
data_test_origin = data_test.copy()

# 非数值型变量转换函数
def convert_to_numeric_for_clean(df):
    """将非数值型特征转换为数值型编码，用于原始特征模型的对比"""
    df_copy = df.copy()
    exclude_col = 'grade_subgrade'
    for col in df_copy.columns:
        if col != exclude_col and (df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category'):
            df_copy[col] = pd.Categorical(df_copy[col]).codes.astype(float)
    return df_copy

# 转换原始数据以用于对比实验的“原始特征模型”
data_train_numeric = convert_to_numeric_for_clean(data_train.copy())
data_test_numeric = convert_to_numeric_for_clean(data_test.copy())

# 将 grade_subgrade 转换为数值型
data_train_numeric['grade_subgrade'] = pd.to_numeric(data_train_numeric['grade_subgrade'])
data_test_numeric['grade_subgrade'] = pd.to_numeric(data_test_numeric['grade_subgrade'])

# 数据清理 (用于原始特征模型)
data_train_clean_orig = data_train_numeric.drop(['id', 'grade_subgrade'], axis=1)

# -------------------------- 组合分箱函数 --------------------------
def create_combination_bins(df, is_train=True):
    df_comb = df.copy()
    
    # 1. 收入×债务比组合
    df_comb['income_bin'] = pd.cut(
        df_comb['annual_income'],
        bins=[0, 10000, 30000, 60000, 110551.7, np.inf],
        labels=['极低收入', '低收入', '中等收入', '高收入', '极高收入'],
        duplicates='drop'
    )
    df_comb['debt_bin'] = pd.cut(
        df_comb['debt_to_income_ratio'],
        bins=[0, 0.1, 0.2, 0.3, 0.63],
        labels=['低债务', '中低债务', '中高债务', '高债务'],
        duplicates='drop'
    )
    df_comb['income_debt_comb'] = df_comb['income_bin'].astype(str) + '+' + df_comb['debt_bin'].astype(str)
    
    # 2. 贷款金额×信用分组合
    df_comb['loan_bin'] = pd.cut(
        df_comb['loan_amount'],
        bins=[0, 5000, 20000, 40000, 48959.95],
        labels=['小额贷款', '中等贷款', '大额贷款', '超大额贷款'],
        duplicates='drop'
    )
    df_comb['credit_bin'] = pd.cut(
        df_comb['credit_score'],
        bins=[395, 600, 680, 750, 849],
        labels=['低信用', '中信用', '高信用', '极高信用'],
        duplicates='drop'
    )
    df_comb['loan_credit_comb'] = df_comb['loan_bin'].astype(str) + '+' + df_comb['credit_bin'].astype(str)
    
    # 删除中间分箱列
    drop_cols = ['income_bin', 'debt_bin', 'loan_bin', 'credit_bin']
    df_comb = df_comb.drop(drop_cols, axis=1)
    
    return df_comb

# -------------------------- WOE/IV计算函数 --------------------------
def calculate_woe_iv(df, feature, target, bins=10, special_values=None):
    if special_values is None:
        special_values = []
    
    if feature in ['income_debt_comb', 'loan_credit_comb']:
        df['bin'] = df[feature].astype(str)
    elif pd.api.types.is_numeric_dtype(df[feature]) and feature not in special_values:
        df[feature] = pd.to_numeric(df[feature], errors='coerce') 
        df.dropna(subset=[feature], inplace=True)
        q_bins = min(bins, len(df[feature].unique()))
        if q_bins < 2:
             df['bin'] = df[feature].astype(str)
        else:
             df['bin'] = pd.qcut(df[feature], q=q_bins, duplicates='drop')
    else:
        df['bin'] = df[feature].astype(str)
    
    bin_stats = df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
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
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    
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
        if feature in ['income_debt_comb', 'loan_credit_comb']:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target, bins=None)
        else:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target)
        iv_list.append({'feature': feature, 'iv': iv})
        woe_maps[feature] = woe_dict
    
    iv_df = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    return iv_df, woe_maps

# -------------------------- WOE 转换函数 (已修正) --------------------------
# -------------------------- WOE 转换函数 (最终修正) --------------------------
def apply_woe_transformation(df, feature, woe_map):
    """将特征值替换为其对应的 WOE 值"""
    
    woe_keys = list(woe_map.keys()) 
    
    # 1. 映射值到分箱
    if woe_keys and isinstance(woe_keys[0], pd.Interval):
        interval_index = pd.IntervalIndex(woe_keys)
        df[feature] = pd.to_numeric(df[feature], errors='coerce') 
        df['bin'] = pd.cut(df[feature], bins=interval_index, duplicates='drop', include_lowest=True)
    else:
        df['bin'] = df[feature].astype(str)
        
    # 2. 映射分箱到 WOE 值
    df_woe = df['bin'].map(woe_map)
    
    # 3. 强制转换为浮点数类型，避免 Categorical 限制
    # 这是关键的修复步骤！
    df_woe = df_woe.astype(float) 
    
    # 4. 填充缺失值（现在 df_woe 是 float 类型，可以直接填充 0）
    df_woe = df_woe.fillna(0) 
    
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    return df_woe

# -------------------------- 模型训练与评估函数（逻辑回归） --------------------------
def train_evaluate_lr(X_train, X_valid, y_train, y_valid, features):
    """训练并评估逻辑回归模型"""
    
    X_train_sub = X_train[features]
    X_valid_sub = X_valid[features]
    
    # 1. 标准化：逻辑回归对尺度敏感，WOE特征需要标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_valid_scaled = scaler.transform(X_valid_sub)
    
    # 2. 训练模型
    model = LogisticRegression(
        random_state=123, 
        solver='liblinear',
        C=0.1, 
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    
    # 3. 预测
    y_pred_proba = model.predict_proba(X_valid_scaled)[:, 1]
    y_pred = np.round(y_pred_proba)
    
    # 4. 计算指标
    auc = roc_auc_score(y_valid, y_pred_proba)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0)
    ks = ks_2samp(y_pred_proba[y_valid == 1], y_pred_proba[y_valid == 0]).statistic
    
    return {
        'model': model,
        'metrics': {
            'AUC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'KS': ks
        }
    }

# -------------------------- 主执行流程 --------------------------
if __name__ == "__main__":
    # 1. 准备带组合特征的数据
    data_train_comb = create_combination_bins(data_train_origin, is_train=True)
    data_test_comb = create_combination_bins(data_test_origin, is_train=False)
    
    # 2. 计算所有特征IV值及 WOE 映射表
    print("开始计算特征的WOE和IV...")
    # 确保 data_train_comb 中的所有非数值列都存在（包括组合特征）
    iv_df, woe_maps = calculate_all_features_woe_iv(
        df=data_train_comb.copy(), 
        target='loan_paid_back',
        exclude_cols=['id']
    )
    
    # 3. 筛选IV>0.1的特征
    useful_features = iv_df[iv_df['iv'] > 0.1]['feature'].tolist()
    print(f"\nIV>0.1的有用特征: {useful_features}")
    
    # 4. 应用 WOE 转换到筛选后的特征集
    X_woe = pd.DataFrame()
    X_woe['loan_paid_back'] = data_train_comb['loan_paid_back']
    
    print("开始对筛选特征进行 WOE 转换...")
    for feature in useful_features:
        if feature in woe_maps:
            woe_map = woe_maps[feature]
            # 注意：传入 data_train_comb 的副本以避免 side effects
            X_woe[feature] = apply_woe_transformation(data_train_comb.copy(), feature, woe_map)

    # 5. 数据集划分
    
    # === 模型 1: 原始数值特征模型（用于对比） ===
    X_original = data_train_clean_orig.drop('loan_paid_back', axis=1)
    y_original = data_train_clean_orig['loan_paid_back']
    # 确保 X_original 的特征是数值型，否则标准化会报错
    X_original = X_original.apply(pd.to_numeric, errors='coerce').fillna(X_original.mean()) 
    
    X_train_orig, X_valid_orig, y_train_orig, y_valid_orig = train_test_split(
        X_original, y_original, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_original
    )
    
    # === 模型 2: WOE 转换 + IV 筛选特征模型 ===
    X_comb_woe = X_woe.drop('loan_paid_back', axis=1)
    y_comb_woe = X_woe['loan_paid_back']
    
    X_train_comb, X_valid_comb, y_train_comb, y_valid_comb = train_test_split(
        X_comb_woe, y_comb_woe, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_comb_woe
    )
    
    # 6. 训练并评估两个逻辑回归模型
    print("\n=== 原始数值特征逻辑回归模型评估 ===")
    original_features = X_original.columns.tolist()
    orig_result = train_evaluate_lr(
        X_train_orig, X_valid_orig, y_train_orig, y_valid_orig,
        original_features
    )
    for metric, value in orig_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== WOE转换+IV筛选特征逻辑回归模型评估 ===")
    woe_features = X_train_comb.columns.tolist()
    comb_result = train_evaluate_lr(
        X_train_comb, X_valid_comb, y_train_comb, y_valid_comb,
        woe_features
    )
    for metric, value in comb_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # 7. 对比结果
    print("\n=== 模型对比 ===")
    metrics_df = pd.DataFrame({
        '指标': orig_result['metrics'].keys(),
        '原始数值特征(LR)': orig_result['metrics'].values(),
        'WOE转换+IV筛选(LR)': comb_result['metrics'].values()
    })
    
    # 确保用于计算的列是浮点数类型
    metrics_df['原始数值特征(LR)'] = metrics_df['原始数值特征(LR)'].astype(float)
    metrics_df['WOE转换+IV筛选(LR)'] = metrics_df['WOE转换+IV筛选(LR)'].astype(float)
    
    # 输出原始对比结果
    print(metrics_df.round(4).to_markdown(index=False))
    
    # 8. 计算提升百分比
    # 仅选择 AUC 所在的行（即第 0 行）进行计算
    auc_orig = metrics_df['原始数值特征(LR)'].iloc[0]
    auc_comb = metrics_df['WOE转换+IV筛选(LR)'].iloc[0]
    
    # 明确计算 AUC 提升
    auc_improvement = ((auc_comb - auc_orig) / auc_orig) * 100
    
    # 创建包含最终结果的 DataFrame
    result_df = pd.DataFrame({
        '指标': ['AUC'],
        'AUC提升(%)': [auc_improvement]
    })
    
    print("\n=== AUC 提升百分比 ===")
    # 直接对包含百分比的列进行 round(2) 和输出
    result_df['AUC提升(%)'] = result_df['AUC提升(%)'].round(2)
    print(result_df.to_markdown(index=False))