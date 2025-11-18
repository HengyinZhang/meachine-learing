import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from linearboost import LinearBoostClassifier

# 1. 读取数据
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 2. 预处理 - grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 3. 非数值型变量转换（排除grade_subgrade）
data_train_origin = data_train.copy()
data_test_origin = data_test.copy()

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

# 4. 处理ID列
test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# 5. 划分数据集
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123
)

# 6. 定义Optuna目标函数（添加class_weight="balanced"）
def objective(trial):
    # 调优参数范围（参考官方推荐）
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
        'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
        'scaler': trial.suggest_categorical(
            'scaler', ['minmax', 'robust', 'quantile-uniform', 'quantile-normal']
        ),
        'class_weight': 'balanced'  # 固定为balanced处理类别不平衡
    }
    
    # 初始化模型
    clf = LinearBoostClassifier(**params)
    clf.fit(X_train, y_train)
    
    # 验证集预测
    y_pred = clf.predict(X_valid)
    return f1_score(y_valid, y_pred, average='weighted')  # 用加权F1作为优化目标

# 7. 运行Optuna调优（200次试验，参考官方配置）
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

# 8. 输出最佳参数
print(f"最佳F1分数: {study.best_value:.4f}")
print("最佳参数:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# 9. 用最佳参数训练最终模型
best_clf = LinearBoostClassifier(
    **study.best_params,
    class_weight='balanced'  # 确保平衡类别权重
)
best_clf.fit(X_train, y_train)

# 10. 验证集评估
y_valid_pred = best_clf.predict(X_valid)
print("\n验证集分类报告:")
print(classification_report(y_valid, y_valid_pred))

# 11. 测试集预测与提交文件生成
test_pred = best_clf.predict(data_test)
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_pred
})
submission.to_csv("linearboost_optuna_submission.csv", index=False)
print("\n提交文件已保存为 linearboost_optuna_submission.csv")