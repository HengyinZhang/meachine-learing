import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, roc_curve
import shap
import matplotlib.pyplot as plt

# -------------------------- 1. 数据加载与基础配置 --------------------------
# 加载Kaggle数据集（需提前下载到本地，或直接用Kaggle内核运行）
data_path = "train.csv"  # 数据集文件名（根据实际路径修改）
df = pd.read_csv(data_path)

# 核心配置（贴合文档逻辑）
TARGET = "loan_paid_back"  # 目标变量：1=还清，0=违约
# 特征分类（数值型+分类型）
NUMERIC_FEATURES = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
CATEGORICAL_FEATURES = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
# 业务损失权重（漏判违约损失:误拒正常损失=10:1）
WEIGHT_DEFAULT = 10.0  # 漏判（0→1）损失
WEIGHT_NORMAL = 1.0    # 误拒（1→0）损失

# -------------------------- 2. 数据预处理 --------------------------
def preprocess_data(df):
    # 2.1 处理极端值（年收入强右偏，99分位数裁剪）
    annual_income_99 = df["annual_income"].quantile(0.99)
    df["annual_income"] = df["annual_income"].apply(lambda x: min(x, annual_income_99))
    
    # 2.2 处理多重共线性（删除与credit_score强负相关的grade_subgrade）
    df = df.drop("grade_subgrade", axis=1)
    CATEGORICAL_FEATURES.remove("grade_subgrade")  # 同步更新分类特征列表
    
    # 2.3 划分训练集/测试集（按时间顺序，假设无时间列则随机划分，测试集占20%）
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify保持类别分布一致
    )
    return X_train, X_test, y_train, y_test

# 2.4 特征工程（原始特征+衍生特征）
def create_derived_features(df):
    df_new = df.copy()
    # 还款能力类衍生特征
    df_new["loan_burden_ratio"] = df_new["loan_amount"] / df_new["annual_income"]  # 贷款负担系数
    df_new["income_debt_coverage"] = df_new["annual_income"] / (
        df_new["debt_to_income_ratio"] * df_new["annual_income"] + df_new["loan_amount"] * df_new["interest_rate"] / 100
    )  # 收入债务覆盖比
    df_new["income_credit_ratio"] = df_new["annual_income"] / df_new["credit_score"]  # 收入信用匹配度
    df_new["extreme_income_flag"] = (df_new["annual_income"] > 110551.7).astype(int)  # 极端收入标识
    
    # 风险匹配类衍生特征
    df_new["credit_debt_interaction"] = df_new["credit_score"] * (1 - df_new["debt_to_income_ratio"])  # 信用-债务交互
    df_new["loan_interest_product"] = df_new["loan_amount"] * df_new["interest_rate"]  # 贷款利息总成本
    
    # 业务场景类衍生特征
    df_new["debt_consolidation_flag"] = ((df_new["loan_purpose"] == "Debt consolidation") & 
                                         (df_new["debt_to_income_ratio"] > 0.12)).astype(int)  # 债务重组倾向
    df_new["high_risk_employment_flag"] = ((df_new["employment_status"] == "Self-employed") & 
                                           (df_new["loan_amount"] > 15020.3)).astype(int)  # 高风险职业-贷款匹配
    df_new["credit_debt_bin"] = (df_new["credit_score"] // 100) * df_new["debt_to_income_ratio"]  # 信用分等级×债务比
    
    # 数值特征变换（解决右偏）
    df_new["log_annual_income"] = np.log1p(df_new["annual_income"])  # 年收入对数变换
    
    return df_new

# 2.5 预处理流水线（编码+标准化）
def build_preprocessor():
    # 数值特征：标准化
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    # 分类特征：独热编码（drop='first'避免多重共线性）
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    
    # 合并预处理逻辑
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + [
                "loan_burden_ratio", "income_debt_coverage", "income_credit_ratio", 
                "credit_debt_interaction", "loan_interest_product", "credit_debt_bin",
                "log_annual_income"
            ]),  # 原始数值特征+衍生数值特征
            ("cat", categorical_transformer, CATEGORICAL_FEATURES + [
                "extreme_income_flag", "debt_consolidation_flag", "high_risk_employment_flag"
            ])  # 原始分类特征+衍生分类特征
        ])
    return preprocessor

# -------------------------- 3. 自定义损失函数与评估指标 --------------------------
# 3.1 自定义加权交叉熵损失（LightGBM专用）
def custom_weighted_cross_entropy(y_true, y_pred):
    # y_pred是原始logit值，转换为概率
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    # 权重分配：违约样本（y_true=0）权重10，正常样本（y_true=1）权重1
    weight = WEIGHT_DEFAULT * (1 - y_true) + WEIGHT_NORMAL * y_true
    # 计算损失、梯度、二阶导
    loss = -weight * (y_true * np.log(1 - y_pred_prob + 1e-10) + (1 - y_true) * np.log(y_pred_prob + 1e-10))
    grad = -weight * ((1 - y_true) / (y_pred_prob + 1e-10) - y_true / (1 - y_pred_prob + 1e-10)) * y_pred_prob * (1 - y_pred_prob)
    hess = weight * y_pred_prob * (1 - y_pred_prob) * [
        (1 - y_true) * (1 - 2 * y_pred_prob) / (y_pred_prob + 1e-10) +
        y_true * (2 * y_pred_prob - 1) / (1 - y_pred_prob + 1e-10)
    ]
    return grad.flatten(), hess.flatten()

# 3.2 自定义评估指标（贴合业务）
def custom_eval_metric(y_true, y_pred):
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred_label = (y_pred_prob > 0.5).astype(int)
    
    # 加权F1（聚焦违约样本召回）
    tp = np.sum((y_true == 0) & (y_pred_label == 0))  # 真违约
    fp = np.sum((y_true == 1) & (y_pred_label == 0))  # 假违约
    fn = np.sum((y_true == 0) & (y_pred_label == 1))  # 假正常（漏判）
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    weighted_f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # 误判成本率
    total_loss = (fn * WEIGHT_DEFAULT) + (fp * WEIGHT_NORMAL)
    cost_rate = total_loss / len(y_true)
    
    # 传统指标（AUC/KS）
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    ks = np.max(tpr - fpr)
    
    return [
        ("weighted_f1", weighted_f1, False),
        ("cost_rate", cost_rate, True),
        ("auc", auc, False),
        ("ks", ks, False)
    ]

# -------------------------- 4. 模型训练 --------------------------
def train_model():
    # 数据加载与预处理
    df_raw = pd.read_csv(data_path)
    df_with_derived = create_derived_features(df_raw)  # 加入衍生特征
    X_train, X_test, y_train, y_test = preprocess_data(df_with_derived)
    
    # 构建预处理流水线
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 转换为LightGBM数据集格式
    lgb_train = lgb.Dataset(X_train_processed, label=y_train)
    lgb_val = lgb.Dataset(X_test_processed, label=y_test, reference=lgb_train)
    
    # LightGBM参数（文档最优参数基础上优化）
    params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.2,  # 增强正则化抑制过拟合
        "lambda_l2": 0.2,
        "verbose": 1,
        "random_state": 42
    }
    
    # 训练模型（自定义损失+评估指标）
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        valid_names=["val"],
        fobj=custom_weighted_cross_entropy,
        feval=custom_eval_metric,
        early_stopping_rounds=50,
        verbose_eval=10
    )
    
    # 模型预测
    y_pred_prob = model.predict(X_test_processed, num_iteration=model.best_iteration)
    y_pred_label = (y_pred_prob > 0.5).astype(int)
    
    # 输出测试集最终指标
    test_auc = roc_auc_score(y_test, y_pred_prob)
    test_acc = accuracy_score(y_test, y_pred_label)
    test_precision = precision_score(y_test, y_pred_label)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    test_ks = np.max(tpr - fpr)
    
    print("\n=== 测试集最终指标 ===")
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"KS: {test_ks:.4f}")
    
    return model, preprocessor, X_test_processed, y_test

# -------------------------- 5. SHAP可解释性分析 --------------------------
def shap_analysis(model, preprocessor, X_test_processed, y_test):
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)
    
    # 1. SHAP摘要图（特征重要性）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X_test_processed, feature_names=get_feature_names(preprocessor))
    plt.title("SHAP Feature Importance (Summary Plot)")
    plt.show()
    
    # 2. 单个特征影响分析（以债务收入比为例）
    # 找到债务收入比在预处理后的特征索引
    feature_names = get_feature_names(preprocessor)
    debt_ratio_idx = [i for i, name in enumerate(feature_names) if "debt_to_income_ratio" in name][0]
    
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(debt_ratio_idx, shap_values[1], X_test_processed, feature_names=feature_names)
    plt.title("SHAP Dependence Plot (debt_to_income_ratio)")
    plt.show()
    
    # 3. 单个样本解释（随机选第100个样本）
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][100],
        X_test_processed[100],
        feature_names=feature_names,
        matplotlib=True
    )
    plt.title("SHAP Force Plot (Sample 100)")
    plt.show()

# 辅助函数：获取预处理后的特征名称
def get_feature_names(preprocessor):
    # 数值特征名称
    numeric_features = NUMERIC_FEATURES + [
        "loan_burden_ratio", "income_debt_coverage", "income_credit_ratio",
        "credit_debt_interaction", "loan_interest_product", "credit_debt_bin",
        "log_annual_income"
    ]
    # 分类特征名称（独热编码后）
    categorical_features = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
        CATEGORICAL_FEATURES + ["extreme_income_flag", "debt_consolidation_flag", "high_risk_employment_flag"]
    )
    # 合并所有特征名称
    return np.concatenate([numeric_features, categorical_features])

# -------------------------- 6. 主函数（一键运行） --------------------------
if __name__ == "__main__":
    # 训练模型
    model, preprocessor, X_test_processed, y_test = train_model()
    
    # SHAP可解释性分析
    shap_analysis(model, preprocessor, X_test_processed, y_test)
    
    # 模型保存（可选）
    model.save_model("loan_default_model.txt")
    print("\n模型已保存为 loan_default_model.txt")