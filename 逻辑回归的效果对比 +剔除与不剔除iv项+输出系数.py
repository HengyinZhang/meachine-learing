import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# -------------------------- 辅助函数：数据预处理 --------------------------

# 数据读取与映射
def load_and_preprocess_data(file_path):
    data_train = pd.read_csv(file_path)
    grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
    data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
    return data_train.copy()

# 组合特征分箱
def create_combination_bins(df):
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
    
    df_comb = df_comb.drop(['income_bin', 'debt_bin', 'loan_bin', 'credit_bin'], axis=1)
    return df_comb

# -------------------------- 核心函数：WOE/IV 计算与转换 --------------------------

def calculate_woe_iv(df, feature, target, bins=10, special_values=None):
    if special_values is None: special_values = []
    
    # 1. 分箱处理
    if feature in ['income_debt_comb', 'loan_credit_comb']:
        df['bin'] = df[feature].astype(str)
    elif pd.api.types.is_numeric_dtype(df[feature]) and feature not in special_values:
        df[feature] = pd.to_numeric(df[feature], errors='coerce') 
        df.dropna(subset=[feature], inplace=True)
        q_bins = min(bins, len(df[feature].unique()))
        if q_bins < 2: df['bin'] = df[feature].astype(str)
        else: df['bin'] = pd.qcut(df[feature], q=q_bins, duplicates='drop')
    else: df['bin'] = df[feature].astype(str)
    
    # 2. 计算 WOE 和 IV
    bin_stats = df.groupby('bin', observed=False)[target].agg(['count', 'sum']).rename(columns={'sum': 'bad'})
    total_bad = bin_stats['bad'].sum()
    total_good = bin_stats['count'].sum() - total_bad
    
    bin_stats['good'] = bin_stats['count'] - bin_stats['bad']
    bin_stats['bad_rate'] = bin_stats['bad'] / (total_bad + 1e-10)
    bin_stats['good_rate'] = bin_stats['good'] / (total_good + 1e-10)
    bin_stats['woe'] = np.log((bin_stats['good_rate'] + 1e-10) / (bin_stats['bad_rate'] + 1e-10))
    bin_stats['iv'] = (bin_stats['good_rate'] - bin_stats['bad_rate']) * bin_stats['woe']
    iv = bin_stats['iv'].sum()
    
    woe_dict = dict(zip(bin_stats.index, bin_stats['woe']))
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    return woe_dict, iv

def calculate_all_features_woe_iv(df, target, exclude_cols=None):
    if exclude_cols is None: exclude_cols = [target]
    else: exclude_cols = exclude_cols + [target]
    
    features = [col for col in df.columns if col not in exclude_cols]
    iv_list = []
    woe_maps = {}
    
    for feature in features:
        woe_dict, iv = calculate_woe_iv(df.copy(), feature, target)
        iv_list.append({'feature': feature, 'iv': iv})
        woe_maps[feature] = woe_dict
    
    iv_df = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    return iv_df, woe_maps

def apply_woe_transformation(df, feature, woe_map):
    """将特征值替换为其对应的 WOE 值，已修复 Categorical 类型问题"""
    
    woe_keys = list(woe_map.keys())
    
    if woe_keys and isinstance(woe_keys[0], pd.Interval):
        interval_index = pd.IntervalIndex(woe_keys) 
        df[feature] = pd.to_numeric(df[feature], errors='coerce') 
        df['bin'] = pd.cut(df[feature], bins=interval_index, duplicates='drop', include_lowest=True)
    else:
        df['bin'] = df[feature].astype(str)
        
    df_woe = df['bin'].map(woe_map)
    df_woe = df_woe.astype(float) 
    df_woe = df_woe.fillna(0) 
    
    df.drop('bin', axis=1, inplace=True, errors='ignore')
    return df_woe

# -------------------------- 评分卡构建流程 (修正为单表输出) --------------------------

def build_scorecard_system(X_train, y_train, features, woe_maps, iv_df, target_score=600, pdo=20, target_odds=50):
    
    X_train_sub = X_train[features]
    
    # 1. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    
    # 2. 训练最终逻辑回归模型
    lr_model = LogisticRegression(
        random_state=123, 
        solver='liblinear',
        C=0.1, 
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # 3. 计算 A 和 B (评分卡刻度)
    B = pdo / np.log(2) # B = PDO / ln(2)
    log_odds_target = np.log(target_odds)
    intercept_raw = lr_model.intercept_[0]
    
    A = target_score + B * log_odds_target
    
    # 4. 输出 LR 系数和刻度参数
    print("-------------------------------------------------")
    print(f"        Logistic Regression 系数 ({len(features)} 个特征)")
    print("-------------------------------------------------")
    
    coefficients = pd.DataFrame({
        'Feature': features,
        'Raw_Coefficient': lr_model.coef_[0],
    })
    print(f"原始截距 (Raw Intercept): {intercept_raw:.6f}")
    print(f"常数 A (基础项调整): {A:.4f}")
    print(f"常数 B (刻度): {B:.4f}")
    
    # 5. 计算模型基础分数（Base Score）
    base_score = A - B * intercept_raw
    print(f"模型基础分 (Base Score, i.e. Score @ Mean WOE): {base_score:.2f} 分")
    print(coefficients.to_markdown(index=False, floatfmt=".6f"))

    # 6. 计算特征分箱得分并合并到单表 (Score contribution for each bin)
    all_scorecard_data = []
    
    for feature in features:
        raw_coeff = coefficients[coefficients['Feature'] == feature]['Raw_Coefficient'].iloc[0]
        woe_map = woe_maps[feature]
        iv_val = iv_df[iv_df['feature'] == feature]['iv'].iloc[0]
        
        for bin_value, woe_value in woe_map.items():
            # Score_Contribution = -B * Coeff * WOE
            score_contribution = -B * raw_coeff * woe_value
            
            all_scorecard_data.append({
                'Feature': feature,
                'IV': iv_val,
                'Bin': str(bin_value),
                'Raw_Coeff': raw_coeff,
                'WOE': woe_value,
                'Score_Contribution': score_contribution
            })
    
    # 创建最终的评分卡单表
    final_scorecard_df = pd.DataFrame(all_scorecard_data)
    
    # 重新排序和选择列，使表格更易读
    final_scorecard_df = final_scorecard_df[[
        'Feature', 'IV', 'Bin', 'WOE', 'Raw_Coeff', 'Score_Contribution'
    ]]
    
    # 按 IV 排序，确保重要特征在前
    final_scorecard_df = final_scorecard_df.sort_values(by=['IV', 'Bin'], ascending=[False, True]).reset_index(drop=True)

    print("\n-------------------------------------------------")
    print("        *** 最终评分卡单表 ***")
    print("-------------------------------------------------")
    
    # 输出单表 (WOE 和得分贡献保留两位小数，IV保留四位小数)
    print(final_scorecard_df.round({'IV': 4, 'WOE': 2, 'Raw_Coeff': 6, 'Score_Contribution': 2}).to_markdown(index=False))

    return lr_model, coefficients, final_scorecard_df

# -------------------------- 主执行流程 --------------------------

if __name__ == "__main__":
    
    # 请确认文件路径是否正确
    FILE_PATH = "E:/Code/python/pre-payback/train.csv"
    
    # 1. 数据准备
    data_train_origin = load_and_preprocess_data(FILE_PATH)
    data_train_comb = create_combination_bins(data_train_origin)
    
    # 2. 计算所有特征IV值及 WOE 映射表
    print("--- 步骤 1/4: 计算 WOE/IV ---")
    iv_df, woe_maps = calculate_all_features_woe_iv(
        df=data_train_comb.copy(), 
        target='loan_paid_back',
        exclude_cols=['id']
    )
    
    # 3. 筛选IV>0.1的特征 (Model 2 特征集)
    useful_features = iv_df[iv_df['iv'] > 0.1]['feature'].tolist()
    print(f"IV>0.1的有用特征 ({len(useful_features)}个): {useful_features}")

    # 4. 应用 WOE 转换
    print("\n--- 步骤 2/4: 应用 WOE 转换并划分数据 ---")
    X_woe_filter = pd.DataFrame()
    X_woe_filter['loan_paid_back'] = data_train_comb['loan_paid_back']
    
    for feature in useful_features:
        if feature in woe_maps:
            woe_map = woe_maps[feature]
            X_woe_filter[feature] = apply_woe_transformation(data_train_comb.copy(), feature, woe_map)

    # 5. 数据集划分 (只取训练集用于建模)
    X_comb_filter = X_woe_filter.drop('loan_paid_back', axis=1)
    y_comb_filter = X_woe_filter['loan_paid_back']
    
    X_train_filter, _, y_train_filter, _ = train_test_split(
        X_comb_filter, y_comb_filter, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_comb_filter
    )
    
    # 6. 构建并输出评分卡体系 (单表格式)
    print("\n--- 步骤 3/4: 构建评分卡模型 ---")
    final_model, final_coeffs, scorecard_result = build_scorecard_system(
        X_train_filter, 
        y_train_filter, 
        X_train_filter.columns.tolist(), 
        woe_maps,
        iv_df,
        target_score=600, 
        pdo=20, 
        target_odds=50
    )

