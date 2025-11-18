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

# 注意：convert_to_numeric_for_clean 不再被 Model 1 使用，但保留用于数据探索或备用。
def convert_to_numeric_for_clean(df):
    """将非数值型特征转换为数值型编码（Label Encoding）"""
    df_copy = df.copy()
    exclude_col = 'grade_subgrade'
    for col in df_copy.columns:
        if col != exclude_col and (df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category'):
            df_copy[col] = pd.Categorical(df_copy[col]).codes.astype(float)
    return df_copy

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
    
    # 3. 强制转换为浮点数类型（解决 Categorical TypeError）
    df_woe = df_woe.astype(float) 
    
    # 4. 填充缺失值
    df_woe = df_woe.fillna(0) 
    
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    return df_woe

# -------------------------- 模型训练与评估函数（逻辑回归） --------------------------
def train_evaluate_lr(X_train, X_valid, y_train, y_valid, features):
    """训练并评估逻辑回归模型"""
    
    X_train_sub = X_train[features]
    X_valid_sub = X_valid[features]
    
    # 1. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_valid_scaled = scaler.transform(X_valid_sub)
    
    # 2. 训练模型 (C=0.1 意味着 L2 正则化强度增强)
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
    # 1. 准备带组合特征的数据 (用于 WOE 计算)
    data_train_comb = create_combination_bins(data_train_origin, is_train=True)
    
    # 2. 计算所有特征IV值及 WOE 映射表
    print("开始计算特征的WOE和IV...")
    iv_df, woe_maps = calculate_all_features_woe_iv(
        df=data_train_comb.copy(), 
        target='loan_paid_back',
        exclude_cols=['id']
    )
    
    # 3. 筛选特征列表
    useful_features = iv_df[iv_df['iv'] > 0.1]['feature'].tolist()
    all_features_for_woe = [col for col in data_train_comb.columns if col not in ['id', 'loan_paid_back']]

    print(f"\nIV>0.1的有用特征 ({len(useful_features)}个): {useful_features}")
    print(f"全部用于WOE的特征 ({len(all_features_for_woe)}个): {all_features_for_woe}")

    # 4. 应用 WOE 转换到不同特征集 (Model 2 & 3)
    
    # --- 4a. 准备 WOE + IV 筛选特征集 (Model 2) ---
    X_woe_filter = pd.DataFrame()
    X_woe_filter['loan_paid_back'] = data_train_comb['loan_paid_back']
    for feature in useful_features:
        if feature in woe_maps:
            woe_map = woe_maps[feature]
            X_woe_filter[feature] = apply_woe_transformation(data_train_comb.copy(), feature, woe_map)

    # --- 4b. 准备 WOE 无筛选全部特征集 (Model 3) ---
    X_woe_all = pd.DataFrame()
    X_woe_all['loan_paid_back'] = data_train_comb['loan_paid_back']
    for feature in all_features_for_woe:
        if feature in woe_maps:
            woe_map = woe_maps[feature]
            X_woe_all[feature] = apply_woe_transformation(data_train_comb.copy(), feature, woe_map)

    # 5. 数据集划分
    
    # === Model 1: K-1 独热编码特征模型（修正后的基准模型） ===
    data_train_orig_for_ohe = data_train_origin.copy() 
    data_train_orig_for_ohe['grade_subgrade'] = pd.to_numeric(data_train_orig_for_ohe['grade_subgrade'], errors='coerce')
    
    # 执行 K-1 独热编码
    object_cols = data_train_orig_for_ohe.select_dtypes(include=['object']).columns
    X_original_ohe = pd.get_dummies(
        data_train_orig_for_ohe.drop(['id', 'grade_subgrade'], axis=1), 
        columns=object_cols, 
        drop_first=True, # <-- K-1 编码
        dummy_na=False   
    )
    
    y_original = X_original_ohe['loan_paid_back']
    X_original = X_original_ohe.drop('loan_paid_back', axis=1)

    # 填充剩余的数值型缺失值
    X_original = X_original.apply(pd.to_numeric, errors='coerce').fillna(X_original.mean()) 
    
    X_train_orig, X_valid_orig, y_train_orig, y_valid_orig = train_test_split(
        X_original, y_original, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_original
    )
    
    # === Model 2: WOE 转换 + IV 筛选特征模型 ===
    X_comb_filter = X_woe_filter.drop('loan_paid_back', axis=1)
    y_comb_filter = X_woe_filter['loan_paid_back']
    
    X_train_filter, X_valid_filter, y_train_filter, y_valid_filter = train_test_split(
        X_comb_filter, y_comb_filter, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_comb_filter
    )

    # === Model 3: WOE 转换 + 无筛选全部特征模型 ===
    X_comb_all = X_woe_all.drop('loan_paid_back', axis=1)
    y_comb_all = X_woe_all['loan_paid_back']
    
    X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(
        X_comb_all, y_comb_all, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_comb_all
    )
    
    # 6. 训练并评估三个逻辑回归模型
    
    print("\n-------------------------------------------------")
    print("        === 模型训练与评估结果 (逻辑回归) ===")
    print("-------------------------------------------------")

    # --- Model 1 ---
    orig_result = train_evaluate_lr(
        X_train_orig, X_valid_orig, y_train_orig, y_valid_orig,
        X_original.columns.tolist()
    )
    print("\n--- 1. K-1 独热编码特征模型 (基准) ---")
    for metric, value in orig_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # --- Model 2 ---
    filter_result = train_evaluate_lr(
        X_train_filter, X_valid_filter, y_train_filter, y_valid_filter,
        X_comb_filter.columns.tolist()
    )
    print("\n--- 2. WOE转换 + IV筛选 (IV > 0.1) 模型 ---")
    for metric, value in filter_result['metrics'].items():
        print(f"{metric}: {value:.4f}")

    # --- Model 3 ---
    all_result = train_evaluate_lr(
        X_train_all, X_valid_all, y_train_all, y_valid_all,
        X_comb_all.columns.tolist()
    )
    print("\n--- 3. WOE转换 + 无筛选 (全部特征) 模型 ---")
    for metric, value in all_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
        
    # 7. 对比结果
    print("\n--------------------- 模型对比 ---------------------")
    metrics_df = pd.DataFrame({
        '指标': orig_result['metrics'].keys(),
        'K-1 OHE (LR)': orig_result['metrics'].values(),
        'WOE + IV筛选(LR)': filter_result['metrics'].values(),
        'WOE + 无筛选(LR)': all_result['metrics'].values()
    })
    
    # 确保用于计算的列是浮点数类型
    metrics_df['K-1 OHE (LR)'] = metrics_df['K-1 OHE (LR)'].astype(float)
    metrics_df['WOE + IV筛选(LR)'] = metrics_df['WOE + IV筛选(LR)'].astype(float)
    metrics_df['WOE + 无筛选(LR)'] = metrics_df['WOE + 无筛选(LR)'].astype(float)
    
    print(metrics_df.round(4).to_markdown(index=False))

    # 8. 计算提升百分比 (WOE无筛选 相对于 WOE + IV筛选)
    auc_filter = metrics_df['WOE + IV筛选(LR)'].iloc[0]
    auc_all = metrics_df['WOE + 无筛选(LR)'].iloc[0]
    
    if auc_filter > 0:
        auc_improvement = ((auc_all - auc_filter) / auc_filter) * 100
        result_df = pd.DataFrame({
            '指标': ['AUC'],
            'WOE无筛选相对提升(%)': [auc_improvement]
        })
        print("\n------------------- 相对提升分析 -------------------")
        print("WOE + 无筛选模型 AUC 相对于 WOE + IV筛选模型的提升：")
        result_df['WOE无筛选相对提升(%)'] = result_df['WOE无筛选相对提升(%)'].round(2)
        print(result_df.to_markdown(index=False))
    else:
        print("\n无法计算相对提升百分比，因为基准AUC为零或负值。")