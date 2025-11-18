# 导入必要的Python库
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold

# 设置matplotlib支持中文字体及正确显示负号
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------
# 一、数据处理（保留原始逻辑）
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

# 8. 检查缺失值
missing = data_train.isnull().sum()
print("缺失值统计：")
print(missing)


# --------------------------
# 二、年收入特征分析（分箱与可视化）
# --------------------------
# 1. 异常值分析
q1 = data_train['annual_income'].quantile(0.25)
q3 = data_train['annual_income'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
outlier_ratio = (data_train['annual_income'] > upper_bound).mean()
print(f"\n异常值上限：{upper_bound}")
print(f"异常值占比：{outlier_ratio:.2%}")

# 2. 分位数分箱（5箱）及还款率分析
data_train['annual_income_bin'] = pd.qcut(
    data_train['annual_income'], 
    q=5, 
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)
bin_repay_rate = data_train.groupby('annual_income_bin')['loan_paid_back'].mean().reset_index()
print("\n分位数分箱还款率：")
print(bin_repay_rate)

# 3. 特征唯一值分析（筛选常数特征）
unique_counts = data_train.nunique()
print("\n各特征唯一值数量：")
print(unique_counts)
constant_features = unique_counts[unique_counts == 1].index.tolist()
print("常数特征：", constant_features)


# --------------------------
# 三、IV值计算（所有特征及分箱对比）
# --------------------------
# 1. 单个特征IV值计算函数
def calculate_iv(df, feature, target, bins=10, split_type='quantile'):
    data = df[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(data[feature]):
        if split_type == 'quantile':
            data['bin'] = pd.qcut(data[feature], q=bins, duplicates='drop')
        else:
            data['bin'] = pd.cut(data[feature], bins=bins, duplicates='drop')
    else:
        data['bin'] = data[feature]
    
    bin_stats = data.groupby('bin')[target].agg(['count', 'sum'])
    bin_stats.columns = ['total', 'bad']
    bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
    total_bad = bin_stats['bad'].sum()
    total_good = bin_stats['good'].sum()
    
    # 平滑处理避免除0
    bin_stats['bad_rate'] = (bin_stats['bad'] + 0.5) / (total_bad + 1)
    bin_stats['good_rate'] = (bin_stats['good'] + 0.5) / (total_good + 1)
    bin_stats['woe'] = np.log(bin_stats['good_rate'] / bin_stats['bad_rate'])
    bin_stats['iv'] = (bin_stats['good_rate'] - bin_stats['bad_rate']) * bin_stats['woe']
    return bin_stats['iv'].sum()

# 2. 计算所有特征的IV值
def calculate_all_iv(X, y, bins=10, split_type='quantile'):
    df = X.copy()
    df['target'] = y
    iv_results = []
    for feature in X.columns:
        iv = calculate_iv(df, feature, 'target', bins=bins, split_type=split_type)
        iv_results.append({'feature': feature, 'iv_value': iv})
    return pd.DataFrame(iv_results).sort_values(by='iv_value', ascending=False)

# 3. 输出所有特征IV值
iv_values = calculate_all_iv(X_train, y_train, bins=10, split_type='quantile')
print("\n所有特征的IV值（按降序排列）：")
print(iv_values)
print("特征数量：", iv_values.shape[0])
iv_values.to_csv('feature_iv_values.csv', index=False)  # 保存结果


# --------------------------
# 四、年收入分箱方法及可视化
# --------------------------
# 1. 等宽分箱（5箱）
def bin_annual_income(df, n_bins=5):
    if 'annual_income' in df.columns:
        df['annual_income_ew_bin5'] = pd.cut(
            df['annual_income'],
            bins=n_bins,
            duplicates='drop',
            labels=[f'bin{i+1}' for i in range(n_bins)]
        )
        df['annual_income_ew_bin5'] = pd.Categorical(df['annual_income_ew_bin5']).codes.astype(float)
    return df
data_train = bin_annual_income(data_train)
data_test = bin_annual_income(data_test)

# 2. 等宽分箱结果分析
print("\n分箱后的年收入特征示例：")
print(data_train[['annual_income', 'annual_income_ew_bin5']].head())

# 3. 等宽分箱区间与还款率分析
bin_analysis = data_train.groupby('annual_income_ew_bin5').agg(
    样本数=('loan_paid_back', 'count'),
    还款率=('loan_paid_back', 'mean')
).reset_index()

# 获取分箱区间边界
_, bins = pd.cut(data_train['annual_income'], bins=5, duplicates='drop', retbins=True)
bin_ranges = [f"[{bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins)-1)]
bin_analysis['收入区间'] = bin_ranges

print("\n等宽分箱区间对还款的影响分析：")
print(bin_analysis[['annual_income_ew_bin5', '收入区间', '样本数', '还款率']])

# 4. 等宽分箱可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=bin_analysis, x='收入区间', y='还款率', palette='Blues_d')
for i, row in bin_analysis.iterrows():
    plt.text(i, row['还款率'] + 0.01, 
             f"样本数: {row['样本数']}\n还款率: {row['还款率']:.2%}", 
             ha='center', fontsize=10)
plt.title('年收入等宽分箱区间与还款率的关系')
plt.xlabel('年收入区间')
plt.ylabel('还款率（比例）')
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()


# 5. 对数转换+等频分箱（10箱）
data_train['annual_income_log'] = np.log1p(data_train['annual_income'])  # 处理0值
data_train['annual_income_log_bin10'] = pd.qcut(
    data_train['annual_income_log'], 
    q=10, 
    labels=[f'log_bin{i+1}' for i in range(10)]
)

# 对数分箱还款率分析
log_bin_repay = data_train.groupby('annual_income_log_bin10')['loan_paid_back'].mean().reset_index()
print("\n对数分箱后的还款率：")
print(log_bin_repay)

# 6. 对数转换后分布可视化
plt.figure(figsize=(10, 6))
sns.histplot(data_train['annual_income_log'], kde=True, bins=30, color='skyblue')
plt.title('对数转换后的年收入分布')
plt.xlabel('Log(年收入 + 1)')
plt.ylabel('频数')
plt.tight_layout()
plt.show()

# 7. 对数分箱还款率可视化
plt.figure(figsize=(12, 6))
sns.barplot(x='annual_income_log_bin10', y='loan_paid_back', data=log_bin_repay, palette='Blues_d')
plt.title('对数分箱后的还款率分布')
plt.xlabel('对数收入分箱')
plt.ylabel('还款率')
plt.ylim(0.7, 0.85)
for i, row in log_bin_repay.iterrows():
    plt.text(i, row['loan_paid_back'] + 0.005, f'{row["loan_paid_back"]:.4f}', ha='center')
plt.tight_layout()
plt.show()


# --------------------------
# 五、不同分箱方法对比（IV值、标准差等）
# --------------------------
def compare_binning_methods(data, feature, target):
    results = []
    total_samples = len(data)
    
    # 方法1：等频分箱（5箱）
    data['eq_freq_bin5'] = pd.qcut(data[feature], q=5, labels=False, duplicates='drop')
    iv_eq_freq5 = calculate_iv(data, 'eq_freq_bin5', target)
    repay_eq_freq5 = data.groupby('eq_freq_bin5')[target].mean()
    results.append({
        'method': '等频分箱(5箱)',
        'iv': iv_eq_freq5,
        'repay_std': repay_eq_freq5.std(),
        'min_sample_ratio': data['eq_freq_bin5'].value_counts().min() / total_samples
    })
    
    # 方法2：等频分箱（10箱）
    data['eq_freq_bin10'] = pd.qcut(data[feature], q=10, labels=False, duplicates='drop')
    iv_eq_freq10 = calculate_iv(data, 'eq_freq_bin10', target)
    repay_eq_freq10 = data.groupby('eq_freq_bin10')[target].mean()
    results.append({
        'method': '等频分箱(10箱)',
        'iv': iv_eq_freq10,
        'repay_std': repay_eq_freq10.std(),
        'min_sample_ratio': data['eq_freq_bin10'].value_counts().min() / total_samples
    })
    
    # 方法3：等宽分箱（5箱）
    data['eq_width_bin5'] = pd.cut(data[feature], bins=5, labels=False, duplicates='drop')
    iv_eq_width5 = calculate_iv(data, 'eq_width_bin5', target)
    repay_eq_width5 = data.groupby('eq_width_bin5')[target].mean()
    results.append({
        'method': '等宽分箱(5箱)',
        'iv': iv_eq_width5,
        'repay_std': repay_eq_width5.std(),
        'min_sample_ratio': data['eq_width_bin5'].value_counts().min() / total_samples
    })
    
    # 方法4：对数+等频分箱（10箱）
    data['log_feature'] = np.log1p(data[feature])
    data['log_eq_freq_bin10'] = pd.qcut(data['log_feature'], q=10, labels=False, duplicates='drop')
    iv_log_eq_freq10 = calculate_iv(data, 'log_eq_freq_bin10', target)
    repay_log_eq_freq10 = data.groupby('log_eq_freq_bin10')[target].mean()
    results.append({
        'method': '对数+等频分箱(10箱)',
        'iv': iv_log_eq_freq10,
        'repay_std': repay_log_eq_freq10.std(),
        'min_sample_ratio': data['log_eq_freq_bin10'].value_counts().min() / total_samples
    })
    
    # 方法5：卡方分箱（需安装feature_engine）
    try:
        from feature_engine.discretisation import DecisionTreeDiscretiser
        disc = DecisionTreeDiscretiser(
            cv=3, scoring='roc_auc', variables=[feature],
            param_grid={'max_depth': [2, 3, 4]}, random_state=123, regression=False
        )
        disc.fit(data, data[target])
        data['chi2_bin'] = disc.transform(data)[feature]
        iv_chi2 = calculate_iv(data, 'chi2_bin', target)
        repay_chi2 = data.groupby('chi2_bin')[target].mean()
        results.append({
            'method': '卡方分箱',
            'iv': iv_chi2,
            'repay_std': repay_chi2.std(),
            'min_sample_ratio': data['chi2_bin'].value_counts().min() / total_samples
        })
    except ImportError:
        print("未安装feature_engine，跳过卡方分箱对比")
    
    return pd.DataFrame(results)

# 执行分箱方法对比
result_df = compare_binning_methods(data_train, 'annual_income', 'loan_paid_back')
print("\n分箱方法对比结果：")
print(result_df)

# 分箱方法可视化对比
plt.figure(figsize=(12, 6))
sns.barplot(x='method', y='iv', data=result_df, palette='Blues')
plt.title('不同分箱方法的IV值对比')
plt.xlabel('分箱方法')
plt.ylabel('IV值')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='method', y='repay_std', data=result_df, palette='Greens')
plt.title('不同分箱方法的还款率标准差对比')
plt.xlabel('分箱方法')
plt.ylabel('还款率标准差')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='method', y='min_sample_ratio', data=result_df, palette='Reds')
plt.title('不同分箱方法的最小样本占比对比')
plt.xlabel('分箱方法')
plt.ylabel('最小样本占比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()