import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier

# 设置中文字体和图像样式
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = (12, 7)    # 适当扩大图像尺寸
plt.rcParams['axes.labelpad'] = 10          # 标签间距
plt.rcParams['font.size'] = 10              # 基础字体大小

# 一、数据处理（保留原始逻辑并新增特征）
# --------------------------
# 1. 读取训练和测试数据
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 保存测试集ID
test_ids = data_test['id'].copy()

# 2. 预处理: 修改grade_subgrade的映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 3. 保留原始数据
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

# 6. 新增特征: credit_grade_ratio
if 'credit_score' in data_train.columns and 'credit_score' in data_test.columns:
    data_train['credit_grade_ratio'] = data_train['credit_score'] / data_train['grade_subgrade'].replace(0, 1e-6)
    data_test['credit_grade_ratio'] = data_test['credit_score'] / data_test['grade_subgrade'].replace(0, 1e-6)
    print("已成功添加特征: credit_grade_ratio")
else:
    print("警告: 数据中未找到'credit_score'列，无法生成credit_grade_ratio特征")

# 7. 处理ID列
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# 8. 划分训练集和验证集
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123
)

# ----------------------------------------------------------------
# KS 计算函数
# ----------------------------------------------------------------
def calculate_ks(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    return (data['CGR'] - data['CBR']).abs().max()

# ----------------------------------------------------------------
# 定义模型评估函数（方便对比特征选择前后的性能）
# ----------------------------------------------------------------
def evaluate_model(model, X, y_true):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'acc': accuracy_score(y_true, y_pred),
        'pre': precision_score(y_true, y_pred, zero_division=0),
        'ks': calculate_ks(y_true, y_pred_proba)
    }

# ----------------------------------------------------------------
# 1. 原始模型（全特征）训练与评估
# ----------------------------------------------------------------
best_params = {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.7}
print(f"使用最佳参数: {best_params}")

# 全特征模型
full_model = LGBMClassifier(
    objective='binary',
    random_state=123,
    verbose=-1,** best_params
)
full_model.fit(X_train, y_train)

# 评估全特征模型
full_metrics = evaluate_model(full_model, X_valid, y_valid)
print("\n--------------- 全特征模型评估 ---------------")
print(f"AUC: {full_metrics['auc']:.4f} | Accuracy: {full_metrics['acc']:.4f}")
print(f"Precision: {full_metrics['pre']:.4f} | KS: {full_metrics['ks']:.4f}")

# ----------------------------------------------------------------
# 2. 递归特征消除（RFE）
# ----------------------------------------------------------------
print("\n--------------- 开始递归特征消除（RFE） ---------------")

# 初始化用于RFE的基础模型（需能输出特征重要性）
rfe_base_model = LGBMClassifier(
    objective='binary',
    random_state=123,
    verbose=-1,
    **best_params
)

# 初始化RFE：每次移除1个最不重要的特征，保留50%的特征（可根据需要调整）
n_features = X_train.shape[1]
target_features = max(1, int(n_features * 0.5))  # 至少保留1个特征
rfe = RFE(
    estimator=rfe_base_model,
    n_features_to_select=target_features,  # 目标特征数量
    step=1  # 每次移除1个特征
)

# 执行RFE特征选择（仅在训练集上进行，避免数据泄露）
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_valid_rfe = rfe.transform(X_valid)  # 用相同的特征选择规则处理验证集
data_test_rfe = rfe.transform(data_test)  # 处理测试集

# 查看RFE结果
selected_features = X.columns[rfe.support_]  # 被选中的特征名称
print(f"\nRFE选择的特征（共{len(selected_features)}个）：")
print(selected_features.tolist())

# ----------------------------------------------------------------
# 3. 用RFE筛选后的特征重新训练模型
# ----------------------------------------------------------------
rfe_model = LGBMClassifier(
    objective='binary',
    random_state=123,
    verbose=-1,** best_params
)
rfe_model.fit(X_train_rfe, y_train)

# 评估筛选特征后的模型
rfe_metrics = evaluate_model(rfe_model, X_valid_rfe, y_valid)
print("\n--------------- RFE筛选特征后模型评估 ---------------")
print(f"AUC: {rfe_metrics['auc']:.4f} | Accuracy: {rfe_metrics['acc']:.4f}")
print(f"Precision: {rfe_metrics['pre']:.4f} | KS: {rfe_metrics['ks']:.4f}")

# ----------------------------------------------------------------
# 4. 对比全特征与RFE筛选特征的性能
# ----------------------------------------------------------------
print("\n--------------- 模型性能对比 ---------------")
print(f"特征类型 | AUC      | Accuracy | Precision | KS")
print("-" * 50)
print(f"全特征   | {full_metrics['auc']:.4f} | {full_metrics['acc']:.4f} | {full_metrics['pre']:.4f} | {full_metrics['ks']:.4f}")
print(f"RFE筛选  | {rfe_metrics['auc']:.4f} | {rfe_metrics['acc']:.4f} | {rfe_metrics['pre']:.4f} | {rfe_metrics['ks']:.4f}")

# ----------------------------------------------------------------
# 5. 用RFE筛选后的模型预测并保存结果
# ----------------------------------------------------------------
def predict_and_save(model, test_features, test_ids, output_path="predictions_rfe.csv"):
    test_pred = model.predict(test_features)
    result = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_pred
    })
    result.to_csv(output_path, index=False)
    print(f"\nRFE模型预测结果已保存至: {output_path}")
    return result

# 执行预测
rfe_test_predictions = predict_and_save(
    model=rfe_model,
    test_features=data_test_rfe,
    test_ids=test_ids,
    output_path="E:/Code/python/pre-payback/loan_predictions_rfe.csv"
)

# 显示前5条预测结果
print("\nRFE模型前5条预测结果:")
print(rfe_test_predictions.head())