import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ----------------------数据处理----------------------
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 处理grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 非数值变量转数值（排除grade_subgrade）
def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df
data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 转换grade_subgrade为数值型，删除id列
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])
test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# ----------------------7:2:1划分训练集、验证集、测试集（分层抽样）----------------------
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']

# 先划分训练集（90%）和测试集（10%）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=123
)

# 再从90%中划分训练集（7/9≈78%）和验证集（2/9≈22%）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=2/9, stratify=y_train_val, random_state=123
)

print(f"数据划分比例：训练集{len(X_train)}/{len(X)}≈70%，验证集{len(X_val)}/{len(X)}≈20%，测试集{len(X_test)}/{len(X)}≈10%")

# ----------------------类别不平衡分析----------------------
pos_count = sum(y == 1)
neg_count = sum(y == 0)
imbalance_ratio = neg_count / pos_count
print(f"数据不平衡情况：正样本数={pos_count}，负样本数={neg_count}，比例={imbalance_ratio:.2f}")


# ----------------------调参结果记录表格----------------------
result_df = pd.DataFrame(columns=[
    '调参步骤', '参数组合', '不平衡处理策略', '验证集F1', '验证集Recall', '测试集F1', '测试集Recall'
])


# ----------------------评估函数（适配超旧版本LightGBM）----------------------
def evaluate_params(params, X_train, y_train, X_val, y_val, X_test, y_test, step_name, imbalance_strategy):
    """
    移除所有可能不兼容的参数，只使用最基础的训练逻辑
    """
    # 训练模型（仅保留核心参数）
    model = lgb.LGBMClassifier(
        **params, 
        random_state=123, 
        n_jobs=-1
    )
    # 旧版本fit方法可能只接受X和y，不接受eval_set和verbose
    model.fit(X_train, y_train)
    
    # 预测概率
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # 搜索最优阈值
    best_thresh = 0.5
    best_f1_val = 0
    for thresh in np.arange(0.1, 0.91, 0.05):
        y_val_pred = (y_val_proba >= thresh).astype(int)
        current_f1 = f1_score(y_val, y_val_pred)
        if current_f1 > best_f1_val:
            best_f1_val = current_f1
            best_thresh = thresh
    
    # 计算指标
    y_val_pred = (y_val_proba >= best_thresh).astype(int)
    val_f1 = f1_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    test_f1 = f1_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    # 记录结果
    global result_df
    result_df.loc[len(result_df)] = {
        '调参步骤': step_name,
        '参数组合': str(params),
        '不平衡处理策略': imbalance_strategy,
        '验证集F1': round(val_f1, 4),
        '验证集Recall': round(val_recall, 4),
        '测试集F1': round(test_f1, 4),
        '测试集Recall': round(test_recall, 4)
    }
    
    print(f"【{step_name}】完成：验证集F1={val_f1:.4f}，测试集F1={test_f1:.4f}，最优阈值={best_thresh:.2f}")
    return val_f1, params


# ----------------------贪心调参过程（仅使用基础参数）----------------------
# 初始参数（只保留旧版本兼容的核心参数）
base_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,  # 固定迭代轮数
    'scale_pos_weight': imbalance_ratio  # 不平衡处理
}


# 步骤1：优化树结构参数
print("\n===== 步骤1：优化树结构参数 =====")
best_f1 = 0
best_params = base_params.copy()

# 测试num_leaves
for num_leaves in [15, 31, 63, 127]:
    params = best_params.copy()
    params['num_leaves'] = num_leaves
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"树结构：num_leaves={num_leaves}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['num_leaves'] = num_leaves

# 测试max_depth
for max_depth in [3, 5, 7, 9, -1]:
    params = best_params.copy()
    params['max_depth'] = max_depth
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"树结构：max_depth={max_depth}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['max_depth'] = max_depth

# 测试min_child_samples
for min_child_samples in [5, 10, 20, 50]:
    params = best_params.copy()
    params['min_child_samples'] = min_child_samples
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"树结构：min_child_samples={min_child_samples}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['min_child_samples'] = min_child_samples


# 步骤2：优化采样参数
print("\n===== 步骤2：优化采样参数 =====")
# 测试bagging_fraction
for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    params = best_params.copy()
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = 1
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"采样：bagging_fraction={bagging_fraction}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['bagging_fraction'] = bagging_fraction
        best_params['bagging_freq'] = 1

# 测试feature_fraction
for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    params = best_params.copy()
    params['feature_fraction'] = feature_fraction
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"采样：feature_fraction={feature_fraction}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['feature_fraction'] = feature_fraction


# 步骤3：优化正则化参数
print("\n===== 步骤3：优化正则化参数 =====")
# 测试reg_alpha
for reg_alpha in [0, 0.1, 0.5, 1.0, 5.0]:
    params = best_params.copy()
    params['reg_alpha'] = reg_alpha
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"正则：reg_alpha={reg_alpha}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['reg_alpha'] = reg_alpha

# 测试reg_lambda
for reg_lambda in [0, 0.1, 0.5, 1.0, 5.0]:
    params = best_params.copy()
    params['reg_lambda'] = reg_lambda
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"正则：reg_lambda={reg_lambda}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['reg_lambda'] = reg_lambda


# 步骤4：优化学习率
print("\n===== 步骤4：优化学习率 =====")
for learning_rate in [0.01, 0.05, 0.1, 0.2]:
    params = best_params.copy()
    params['learning_rate'] = learning_rate
    # 学习率小则增加迭代轮数
    params['n_estimators'] = 2000 if learning_rate <= 0.05 else 1000
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"迭代：learning_rate={learning_rate}, n_estimators={params['n_estimators']}",
        imbalance_strategy=f"scale_pos_weight={imbalance_ratio:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['learning_rate'] = learning_rate
        best_params['n_estimators'] = params['n_estimators']

# 微调scale_pos_weight
print("\n===== 微调不平衡参数 =====")
for sw in [imbalance_ratio*0.5, imbalance_ratio*0.8, imbalance_ratio, imbalance_ratio*1.2, imbalance_ratio*1.5]:
    params = best_params.copy()
    params['scale_pos_weight'] = sw
    current_f1, _ = evaluate_params(
        params, X_train, y_train, X_val, y_val, X_test, y_test,
        step_name=f"不平衡微调：scale_pos_weight={sw:.2f}",
        imbalance_strategy=f"scale_pos_weight={sw:.2f}"
    )
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params['scale_pos_weight'] = sw


# ----------------------保存结果----------------------
print("\n===== 调参结果汇总 =====")
print(result_df)
result_df.to_csv("lightgbm_721调参结果.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存至：lightgbm_721调参结果.csv")