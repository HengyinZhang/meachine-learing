# ================================================================
# lightboost_model_with_lime.py  使用最佳参数的LightGBM模型及LIME解释
# ================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from lightgbm import LGBMClassifier
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os
from sklearn.preprocessing import LabelEncoder  # 新增：处理非数值特征

# ----------------------------字体配置----------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------创建输出目录----------------------------
if not os.path.exists('lime_results'):
    os.makedirs('lime_results')

# ----------------------------------------------------------------
# 1. 读取数据
# ----------------------------------------------------------------
try:
    data_train = pd.read_csv(r"E:/Code/python/pre-payback/train.csv")
    data_test = pd.read_csv(r"E:/Code/python/pre-payback/test.csv")
except FileNotFoundError as e:
    print(f"文件读取错误: {e}")
    raise

# ----------------------------------------------------------------
# 2. 预处理（增强类型转换逻辑）
# ----------------------------------------------------------------
grade_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7'}

if 'grade_subgrade' in data_train.columns and 'grade_subgrade' in data_test.columns:
    data_train['grade_subgrade'] = data_train['grade_subgrade'].astype(str).replace(grade_map, regex=True)
    data_test['grade_subgrade'] = data_test['grade_subgrade'].astype(str).replace(grade_map, regex=True)
else:
    print("警告：grade_subgrade列不存在，跳过该列处理")

# 优化：明确处理object类型特征（解决ValueError）
def convert_to_numeric(df):
    # 先处理已知的分类特征
    categorical_cols = ['gender', 'marital_status', 'education_level', 
                       'employment_status', 'loan_purpose', 'grade_subgrade']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            le = LabelEncoder()
            # 合并所有可能值进行编码，避免测试集出现新类别
            all_values = pd.concat([data_train[col], data_test[col]], axis=0).drop_duplicates()
            le.fit(all_values)
            df[col] = le.transform(df[col])
    
    # 处理其他非目标列
    for col in df.columns:
        if col not in {'grade_subgrade', 'id', 'loan_paid_back'} and \
           (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            try:
                df[col] = pd.Categorical(df[col]).codes.astype(float)
            except Exception as e:
                print(f"转换列 {col} 时出错: {e}")
                df[col] = df[col].fillna(0)
    return df

data_train = convert_to_numeric(data_train)
data_test = convert_to_numeric(data_test)

# 处理grade_subgrade转换
try:
    data_train['grade_subgrade'] = pd.to_numeric(data_train['grade_subgrade'], errors='coerce')
    data_test['grade_subgrade'] = pd.to_numeric(data_test['grade_subgrade'], errors='coerce')
    data_train['grade_subgrade'].fillna(data_train['grade_subgrade'].median(), inplace=True)
    data_test['grade_subgrade'].fillna(data_test['grade_subgrade'].median(), inplace=True)
except Exception as e:
    print(f"处理grade_subgrade时出错: {e}")

# 删除id列
if 'id' in data_train.columns:
    data_train = data_train.drop('id', axis=1)
if 'id' in data_test.columns:
    data_test = data_test.drop('id', axis=1)

# ----------------------------------------------------------------
# 3. 训练/验证划分
# ----------------------------------------------------------------
if 'loan_paid_back' not in data_train.columns:
    raise ValueError("训练数据中不存在'loan_paid_back'列，请检查数据")

X = data_train.drop('loan_paid_back', axis=1)
y = data_train['loan_paid_back'].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=123, stratify=y
)

# ----------------------------------------------------------------
# 4. KS 计算函数
# ----------------------------------------------------------------
def calculate_ks(y_true, y_pred_proba):
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true和y_pred_proba长度不一致")
    if np.sum(y_true) == 0 or np.sum(1 - y_true) == 0:
        print("警告：目标变量中存在类别为0的情况，KS计算可能不准确")
        return 0.0
    
    data = pd.DataFrame({'y_true': y_true, 'y_probas': y_pred_proba})
    data = data.sort_values(by='y_probas', ascending=False)
    data['CGR'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['CBR'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    return (data['CGR'] - data['CBR']).abs().max()

# ----------------------------------------------------------------
# 5. LightGBM模型训练与评估
# ----------------------------------------------------------------
best_params = {
    'learning_rate': 0.2,
    'max_depth': 5,
    'n_estimators': 200,
    'subsample': 0.8
}

lgb_model = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    random_state=123,
    verbose=-1,** best_params
)
lgb_model.fit(X_train, y_train)

# 模型评估
y_pred_proba = lgb_model.predict_proba(X_valid)[:, 1]
y_pred = lgb_model.predict(X_valid)

try:
    auc = roc_auc_score(y_valid, y_pred_proba)
except Exception as e:
    print(f"AUC计算错误: {e}")
    auc = 0.0

try:
    acc = accuracy_score(y_valid, y_pred)
except Exception as e:
    print(f"准确率计算错误: {e}")
    acc = 0.0

try:
    pre = precision_score(y_valid, y_pred, zero_division=0)
except Exception as e:
    print(f"精确率计算错误: {e}")
    pre = 0.0

try:
    ks = calculate_ks(y_valid, y_pred_proba)
except Exception as e:
    print(f"KS计算错误: {e}")
    ks = 0.0

print("--------------- LightGBM 最佳参数模型 ---------------")
print(f"Best Parameters: {best_params}")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Kolmogorov-Smirnov (KS): {ks:.4f}")

# ----------------------------------------------------------------
# 6. LIME模型解释（优化显示版本）
# ----------------------------------------------------------------
print("\n--------------- LIME 模型解释 ---------------")

# 优化：带特征名称的预测函数（消除警告）
def predict_with_names(X_array):
    return lgb_model.predict_proba(pd.DataFrame(X_array, columns=X_train.columns))

# 创建LIME解释器
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['未还款', '已还款'],
    mode='classification',
    random_state=123,
    discretize_continuous=True,
    discretizer='quartile'
)

# 随机选择样本
np.random.seed(None)  # 每次运行随机选择不同样本
sample_size = 3
valid_indices = X_valid.index.tolist()
sample_size = min(sample_size, len(valid_indices))
sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)

# 优化：自定义HTML保存函数（解决显示不全）
def save_optimized_html(exp, path):
    html = exp.as_html()
    # 加宽表格和列宽，确保特征名完整显示
    html = html.replace(
        '<table style="border:None">',
        '<table style="border:None; width: 1400px;">'
    )
    html = html.replace(
        '<th style="text-align:left">Feature</th>',
        '<th style="text-align:left; width: 400px;">特征</th>'
    )
    html = html.replace(
        '<th style="text-align:left">Value</th>',
        '<th style="text-align:left; width: 200px;">取值</th>'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

for idx in sample_indices:
    print(f"\n解释样本索引: {idx}")
    try:
        sample = X_valid.loc[idx]
        
        # 生成解释（减少特征数量避免拥挤）
        exp = explainer.explain_instance(
            data_row=sample.values,
            predict_fn=predict_with_names,  # 使用带特征名的预测函数
            num_features=8,  # 减少显示特征数，避免拥挤
            num_samples=5000
        )
        
        # 优化展示
        try:
            # Notebook环境
            fig = exp.as_pyplot_figure()
            plt.title(f'样本 {idx} 的LIME解释')
            plt.tight_layout()
            plt.show()
            display(HTML(exp.as_html()))
        except:
            # 非Notebook环境保存优化后的HTML和图片
            html_path = f'lime_results/lime_explanation_{idx}.html'
            img_path = f'lime_results/lime_explanation_{idx}.png'
            save_optimized_html(exp, html_path)  # 保存优化后的HTML
            
            fig = exp.as_pyplot_figure()
            fig.suptitle(f'样本 {idx} 的LIME解释')
            fig.tight_layout()
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"样本 {idx} 的可视化解释已保存至 {html_path} 和 {img_path}")
        
        # 显示预测概率
        pred_proba = lgb_model.predict_proba([sample.values])[0]
        print(f"模型预测概率 - 未还款: {pred_proba[0]:.4f}, 已还款: {pred_proba[1]:.4f}")
        print("特征影响权重（正值增加已还款概率，负值增加未还款概率）:")
        for feature, weight in exp.as_list():
            print(f"  {feature}: {weight:.4f}")
            
    except Exception as e:
        print(f"处理样本 {idx} 时出错: {e}")
        continue

# 生成全局特征重要性
print("\n--------------- 全局特征重要性（LIME平均影响） ---------------")
feature_importance = {feat: 0.0 for feat in X_train.columns}

global_sample_size = 100
global_indices = np.random.choice(X_valid.index, size=min(global_sample_size, len(X_valid)), replace=False)

for idx in global_indices:
    try:
        exp = explainer.explain_instance(
            data_row=X_valid.loc[idx].values,
            predict_fn=predict_with_names,
            num_features=X_train.shape[1],
            num_samples=5000
        )
        for feature_str, weight in exp.as_list():
            original_feature = feature_str.split(' ')[0].strip()
            if original_feature in feature_importance:
                feature_importance[original_feature] += abs(weight)
    except Exception as e:
        print(f"处理样本 {idx} 时出错: {e}")
        continue

# 可视化全局特征重要性
feature_importance_df = pd.DataFrame({
    'feature': feature_importance.keys(),
    'importance': [v / len(global_indices) for v in feature_importance.values()]
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
top_n = 10
top_features = feature_importance_df.head(top_n)[::-1]
plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
plt.xlabel('平均绝对影响权重')
plt.title(f'LIME全局特征重要性（Top {top_n}）')
plt.tight_layout()
plt.savefig('lime_results/global_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(feature_importance_df.head(5))