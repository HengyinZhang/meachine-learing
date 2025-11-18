import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import lightgbm as lgb
# 导入日志回调和早停回调 (适配 LightGBM 4.x 版本)
from lightgbm import log_evaluation, early_stopping
from scipy.stats import ks_2samp # 导入 KS 检验所需模块

# -------------------------- 数据读取与预处理 --------------------------
# 请确保文件路径正确
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 保留原始数据（用于后续组合特征的创建）
# 注意：此时的 data_train_origin 仍包含 object 类型的列！
data_train_origin = data_train.copy()
data_test_origin = data_test.copy()

# 非数值型变量转换（排除grade_subgrade）
def convert_to_numeric(df):
    """将非数值型特征转换为数值型编码（使用 pd.Categorical.codes）"""
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        # 针对 object 或 category 类型进行编码
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            # 使用 codes 转换为整数编码，并转为 float 类型供模型使用
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

# 对原始模型使用的数据集进行转换
data_train_numeric = convert_to_numeric(data_train.copy())
data_test_numeric = convert_to_numeric(data_test.copy())

# 将 grade_subgrade 转换为数值型
data_train_numeric['grade_subgrade'] = pd.to_numeric(data_train_numeric['grade_subgrade'])
data_test_numeric['grade_subgrade'] = pd.to_numeric(data_test_numeric['grade_subgrade'])


# 数据清理 (用于原始特征模型)
test_ids = data_test_numeric['id']
data_train_clean = data_train_numeric.drop(['id', 'grade_subgrade'], axis=1)
data_test_clean = data_test_numeric.drop(['id', 'grade_subgrade'], axis=1)

# -------------------------- 组合分箱函数 --------------------------
def create_combination_bins(df, is_train=True):
    df_comb = df.copy()
    
    # 1. 收入×债务比组合
    df_comb['income_bin'] = pd.cut(
        df_comb['annual_income'],
        bins=[0, 10000, 30000, 60000, 110551.7, np.inf],
        labels=['极低收入', '低收入', '中等收入', '高收入', '极高收入']
    )
    df_comb['debt_bin'] = pd.cut(
        df_comb['debt_to_income_ratio'],
        bins=[0, 0.1, 0.2, 0.3, 0.63],
        labels=['低债务', '中低债务', '中高债务', '高债务']
    )
    # 将类别合并为字符串
    df_comb['income_debt_comb'] = df_comb['income_bin'].astype(str) + '+' + df_comb['debt_bin'].astype(str)
    
    # 2. 贷款金额×信用分组合
    df_comb['loan_bin'] = pd.cut(
        df_comb['loan_amount'],
        bins=[0, 5000, 20000, 40000, 48959.95],
        labels=['小额贷款', '中等贷款', '大额贷款', '超大额贷款']
    )
    df_comb['credit_bin'] = pd.cut(
        df_comb['credit_score'],
        bins=[395, 600, 680, 750, 849],
        labels=['低信用', '中信用', '高信用', '极高信用']
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
        # 使用 qcut 进行等频分箱以处理连续型变量
        df[feature] = pd.to_numeric(df[feature], errors='coerce') 
        df.dropna(subset=[feature], inplace=True)
        # 确保分箱数量有效
        q_bins = min(bins, len(df[feature].unique()))
        if q_bins < 2: # 如果特征值太少，无法分箱，则使用唯一的特征值作为 bin
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
    
    # 防止分母为零
    bin_stats['bad_rate'] = bin_stats['bad'] / (total_bad + 1e-10)
    bin_stats['good_rate'] = bin_stats['good'] / (total_good + 1e-10)
    # 计算WOE
    bin_stats['woe'] = np.log((bin_stats['good_rate'] + 1e-10) / (bin_stats['bad_rate'] + 1e-10))
    # 计算IV
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
        # print(f"计算特征 {feature} 的WOE和IV...")
        if feature in ['income_debt_comb', 'loan_credit_comb']:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target, bins=None)
        else:
            woe_dict, iv = calculate_woe_iv(df.copy(), feature, target)
        iv_list.append({'feature': feature, 'iv': iv})
        woe_maps[feature] = woe_dict
    
    iv_df = pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    return iv_df, woe_maps

# -------------------------- KS计算函数 --------------------------
def calculate_ks(y_true, y_pred):
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic

# -------------------------- 模型训练与评估函数（适配 LightGBM 4.x） --------------------------
def train_evaluate_lgb(X_train, X_valid, y_train, y_valid, features, params):
    """训练并评估LightGBM模型（适配4.x版本，使用回调函数进行早停）"""
    
    # 准备数据
    lgb_train = lgb.Dataset(X_train[features], label=y_train)
    lgb_valid = lgb.Dataset(X_valid[features], label=y_valid, reference=lgb_train)
    
    # 日志回调和早停回调 (核心修改点：早停必须作为回调函数)
    callbacks = [
        log_evaluation(period=100),         # 每100轮输出一次日志
        early_stopping(stopping_rounds=50)  # 早停设置
    ]
    
    # 训练模型 (移除 early_stopping_rounds 关键字参数)
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid], # 只需要在验证集上进行早停判断
        num_boost_round=params['n_estimators'],
        callbacks=callbacks
    )
    
    # 预测
    y_pred_proba = model.predict(X_valid[features], num_iteration=model.best_iteration)
    y_pred = np.round(y_pred_proba)
    
    # 计算指标
    auc = roc_auc_score(y_valid, y_pred_proba)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0) # 增加 zero_division=0 避免警告
    ks = calculate_ks(y_valid, y_pred_proba)
    
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
    
    # === FIX: 确保 data_train_comb 中的原始特征也被转换为数值型 ===
    # 否则 employment_status 将保持 object 类型，导致 LightGBM 报错
    print("开始对含组合特征的数据集进行数值编码...")
    data_train_comb = convert_to_numeric(data_train_comb)
    data_test_comb = convert_to_numeric(data_test_comb)
    
    # 确保 grade_subgrade 也是数值型
    data_train_comb['grade_subgrade'] = pd.to_numeric(data_train_comb['grade_subgrade'], errors='coerce')
    data_test_comb['grade_subgrade'] = pd.to_numeric(data_test_comb['grade_subgrade'], errors='coerce')
    # ==========================================================
    
    # 2. 计算所有特征IV值（含组合特征）
    print("开始计算特征的WOE和IV...")
    iv_df, woe_maps = calculate_all_features_woe_iv(
        df=data_train_comb.copy(), # 使用副本进行 IV 计算，避免修改原数据
        target='loan_paid_back',
        exclude_cols=['id']
    )
    
    # 3. 筛选IV>0.1的特征
    useful_features = iv_df[iv_df['iv'] > 0.1]['feature'].tolist()
    print(f"\nIV>0.1的有用特征: {useful_features}")
    
    # 4. 处理组合特征为数值型（使用 Categorical Codes）
    for col in ['income_debt_comb', 'loan_credit_comb']:
        # 统一处理训练集和测试集
        cat = pd.Categorical(data_train_comb[col])
        data_train_comb[col] = cat.codes.astype(float)
        # 对测试集使用相同的类别进行编码
        data_test_comb[col] = pd.Categorical(data_test_comb[col], categories=cat.categories).codes.astype(float)
        
    # 5. 数据集划分
    # 原始特征数据集
    X_original = data_train_clean.drop('loan_paid_back', axis=1)
    y_original = data_train_clean['loan_paid_back']
    X_train_orig, X_valid_orig, y_train_orig, y_valid_orig = train_test_split(
        X_original, y_original, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_original
    )
    
    # 带组合特征的数据集
    data_train_comb_clean = data_train_comb.drop(['id', 'grade_subgrade'], axis=1)
    X_comb = data_train_comb_clean.drop('loan_paid_back', axis=1)
    y_comb = data_train_comb_clean['loan_paid_back']
    
    X_train_comb, X_valid_comb, y_train_comb, y_valid_comb = train_test_split(
        X_comb, y_comb, train_size=0.8, test_size=0.2, 
        random_state=123, stratify=y_comb
    )
    
    # 6. 定义最佳参数
    best_params = {
        'learning_rate': 0.2,
        'max_depth': 5,
        'n_estimators': 200, # num_boost_round
        'subsample': 0.8,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 123,
        'verbose': -1 # 抑制 LightGBM 的默认输出
    }
    
    # 7. 训练并评估两个模型
    print("\n=== 原始特征模型评估 ===")
    original_features = X_original.columns.tolist()
    orig_result = train_evaluate_lgb(
        X_train_orig, X_valid_orig, y_train_orig, y_valid_orig,
        original_features, best_params
    )
    for metric, value in orig_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== IV>0.1特征（含组合特征）模型评估 ===")
    # 确保所有特征都在 X_comb 中
    valid_useful_features = [f for f in useful_features if f in X_comb.columns]
    
    comb_result = train_evaluate_lgb(
        X_train_comb, X_valid_comb, y_train_comb, y_valid_comb,
        valid_useful_features, best_params
    )
    for metric, value in comb_result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # 8. 对比结果
    print("\n=== 模型对比 ===")
    metrics_df = pd.DataFrame({
        '指标': orig_result['metrics'].keys(),
        '原始特征': orig_result['metrics'].values(),
        'IV>0.1特征(含组合)': comb_result['metrics'].values()
    })
    print(metrics_df.round(4).to_markdown(index=False))
    
    # 9. 计算提升百分比
    metrics_df['AUC提升(%)'] = (metrics_df.iloc[0, 2] - metrics_df.iloc[0, 1]) / metrics_df.iloc[0, 1] * 100
    metrics_df['Accuracy提升(%)'] = (metrics_df.iloc[1, 2] - metrics_df.iloc[1, 1]) / metrics_df.iloc[1, 1] * 100
    metrics_df['Precision提升(%)'] = (metrics_df.iloc[2, 2] - metrics_df.iloc[2, 1]) / metrics_df.iloc[2, 1] * 100
    metrics_df['KS提升(%)'] = (metrics_df.iloc[3, 2] - metrics_df.iloc[3, 1]) / metrics_df.iloc[3, 1] * 100
    print("\n=== 提升百分比 ===")
    print(metrics_df[['指标', 'AUC提升(%)', 'Accuracy提升(%)', 'Precision提升(%)', 'KS提升(%)']].round(2).to_markdown(index=False))