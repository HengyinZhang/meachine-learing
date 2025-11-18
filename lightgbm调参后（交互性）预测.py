import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from lightgbm import LGBMClassifier

# 设置中文字体和图像样式
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = (10, 6)    # 固定图像大小
plt.rcParams['axes.labelpad'] = 10          # 标签间距
plt.rcParams['font.size'] = 10              # 基础字体大小

# 一、数据处理（保留原始逻辑并新增特征）
# --------------------------
# 1. 读取训练和测试数据
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 保存测试集ID（关键：后续用于输出结果）
test_ids = data_test['id'].copy()

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

# 6. 新增特征: credit_grade_ratio = credit_score / grade_subgrade
if 'credit_score' in data_train.columns and 'credit_score' in data_test.columns:
    data_train['credit_grade_ratio'] = data_train['credit_score'] / data_train['grade_subgrade'].replace(0, 1e-6)
    data_test['credit_grade_ratio'] = data_test['credit_score'] / data_test['grade_subgrade'].replace(0, 1e-6)
    print("已成功添加特征: credit_grade_ratio")
else:
    print("警告: 数据中未找到'credit_score'列，无法生成credit_grade_ratio特征")

# 7. 处理ID列（训练集移除，测试集已提前保存ID）
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
# 训练模型（使用最佳参数）
# ----------------------------------------------------------------
best_params = {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.7}
print(f"使用最佳参数: {best_params}")

lgb_model = LGBMClassifier(
    objective='binary',
    random_state=123,
    verbose=-1,
    **best_params
)
lgb_model.fit(X_train, y_train)

# 模型评估
y_pred_proba = lgb_model.predict_proba(X_valid)[:, 1]
y_pred = lgb_model.predict(X_valid)

auc = roc_auc_score(y_valid, y_pred_proba)
acc = accuracy_score(y_valid, y_pred)
pre = precision_score(y_valid, y_pred, zero_division=0)
ks = calculate_ks(y_valid, y_pred_proba)

print("\n--------------- 模型评估 ---------------")
print(f"AUC: {auc:.4f} | Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f} | KS: {ks:.4f}")

# ----------------------------------------------------------------
# 预测函数：对测试集进行预测并保存结果
# ----------------------------------------------------------------
def predict_and_save(model, test_features, test_ids, output_path="predictions.csv"):
    """
    对测试集进行预测并保存结果
    
    参数:
    - model: 训练好的模型
    - test_features: 预处理后的测试集特征
    - test_ids: 测试集的ID列
    - output_path: 结果保存路径
    """
    # 预测概率（可选，如需保留概率）
    # test_proba = model.predict_proba(test_features)[:, 1]
    
    # 预测类别（0/1）
    test_pred = model.predict(test_features)
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_pred  # 预测的还款状态
    })
    
    # 保存结果
    result.to_csv(output_path, index=False)
    print(f"\n预测结果已保存至: {output_path}")
    return result

# 执行预测并保存
test_predictions = predict_and_save(
    model=lgb_model,
    test_features=data_test,  # 预处理后的测试集特征
    test_ids=test_ids,        # 提前保存的测试集ID
    output_path="E:/Code/python/pre-payback/loan_predictions.csv"  # 输出路径可自定义
)

# 显示前5条预测结果
print("\n前5条预测结果:")
print(test_predictions.head())