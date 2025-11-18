import pandas as pd
import numpy as np
import os
import ray
from ray import tune
from ray.tune import TuneConfig, Tuner, RunConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb


# ---------------------- 关键：强制Ray使用自定义定义短路径（短路径+英文目录） ----------------------
# 1. 选择一个简单的自定义目录（必须是短路径、纯英文，如D:\ray_data）
# 请手动创建该目录，或确保有写入权限
RAY_ROOT = "D:/ray_data"  # 重点：修改为你的短路径英文目录（不要用系统Temp）
os.makedirs(RAY_ROOT, exist_ok=True)

# 2. 设置所有Ray相关路径环境变量
os.environ["RAY_TEMP_DIR"] = os.path.join(RAY_ROOT, "temp")
os.environ["RAY_RESULT_DIR"] = os.path.join(RAY_ROOT, "results")
os.environ["TUNE_RESULT_DIR"] = os.path.join(RAY_ROOT, "tune_results")

# 3. 初始化Ray（禁用默认日志目录，强制使用自定义路径）
ray.init(
    ignore_reinit_error=True,
    _temp_dir=os.environ["RAY_TEMP_DIR"]  # 内部参数，强制临时文件路径
)


# ---------------------- 1. 数据处理 ----------------------
data_train = pd.read_csv("E:/Code/python/pre-payback/train.csv")
data_test = pd.read_csv("E:/Code/python/pre-payback/test.csv")

# 缩减训练数据量（20%样本）
data_train = data_train.groupby('loan_paid_back', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=123)
)
print(f"缩减后训练集样本量: {len(data_train)}")

# grade_subgrade映射
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}
data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)

# 非数值型变量转换
def convert_to_numeric(df):
    exclude_col = 'grade_subgrade'
    for col in df.columns:
        if col != exclude_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df

data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 转换grade_subgrade为数值
data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'])
data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'])

# 处理ID列
test_ids = data_test['id']
data_train = data_train.drop('id', axis=1)
data_test = data_test.drop('id', axis=1)

# 划分8:2训练集和验证集
X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123, stratify=y
)


# ---------------------- 2. 训练函数 ----------------------
def train_with_validation(config):
    # 固定参数（适配CPU）
    fixed_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 123,
        "n_jobs": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    params = {**fixed_params,** config}
    
    # 训练模型
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        num_boost_round=500,
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    # 评估
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    tune.report(
        val_auc=roc_auc_score(y_valid, y_pred),
        val_acc=accuracy_score(y_valid, (y_pred > 0.5).astype(int))
    )


# ---------------------- 3. Ray超参数搜索（随机搜索，避免自定义Searcher的复杂度） ----------------------
def main():
    # 超参数空间（与之前一致）
    param_space = {
        "learning_rate": tune.uniform(0.01, 0.1),
        "n_estimators": tune.randint(200, 800),
        "num_leaves": tune.randint(30, 150),
        "max_depth": tune.randint(5, 10),
        "min_child_samples": tune.randint(10, 50),
        "reg_alpha": tune.loguniform(1e-4, 1.0),
        "reg_lambda": tune.loguniform(1e-4, 1.0)
    }
    
    # 配置Tuner（使用Ray原生随机搜索，避免自定义搜索器的潜在问题）
    tuner = Tuner(
        train_with_validation,
        tune_config=TuneConfig(
            metric="val_auc",
            mode="max",
            num_samples=20,  # 搜索20次
            max_concurrent_trials=4  # 限制并行数，避免路径过长（关键）
        ),
        run_config=RunConfig(
            storage_path=os.path.join(RAY_ROOT, "tuner_storage"),  # 自定义存储路径
            name="lightgbm_tune"
        ),
        param_space=param_space
    )
    
    # 运行搜索
    results = tuner.fit()
    
    # 输出最优结果
    best = results.get_best_result()
    print(f"最优AUC: {best.metrics['val_auc']:.4f}")
    print(f"最优超参数: {best.config}")
    
    # 训练最终模型
    final_model = lgb.LGBMClassifier(
        **best.config,**{
            "objective": "binary",
            "n_estimators": 500,
            "random_state": 123,
            "n_jobs": -1
        }
    )
    final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    # 生成预测
    test_pred = final_model.predict_proba(data_test)[:, 1]
    pd.DataFrame({
        "id": test_ids,
        "loan_paid_back_prob": test_pred
    }).to_csv("submission.csv", index=False)
    print("预测结果已保存为 submission.csv")
    
    ray.shutdown()


if __name__ == "__main__":
    main()