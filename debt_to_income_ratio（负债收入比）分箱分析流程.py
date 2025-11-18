import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 定义特征和目标变量
feature_name = "debt_to_income_ratio"  # 负债收入比（数值型）
target = "loan_paid_back"

# 2. 特征基本分布分析
print(f"\n=== {feature_name} 基本分布 ===")
desc_stats = data_train[feature_name].describe()
print(desc_stats.round(2))

# 可视化分布（直方图+核密度图）
plt.figure(figsize=(10, 6))
sns.histplot(data=data_train, x=feature_name, kde=True, bins=30, color="lightcoral")
plt.axvline(x=desc_stats["50%"], color="red", linestyle="--", label=f"中位数: {desc_stats['50%']:.2f}")
plt.title(f"{feature_name} 分布 (IV=0.6646)")
plt.xlabel(f"{feature_name}（负债/收入）")
plt.ylabel("频数")
plt.legend()
plt.show()

# 3. 分箱处理（等频分箱，按业务逻辑分10箱）
n_bins = 10
# 先获取分箱结果（Categorical类型）
qcut_result = pd.qcut(
    data_train[feature_name], 
    q=n_bins, 
    duplicates="drop"  # 处理重复值导致的分箱不均
)
# 将分箱结果存入DataFrame
data_train[f"{feature_name}_bin"] = qcut_result.cat.codes  # 用整数标签表示分箱

# 4. 分箱后还款率关联分析
# 4.1 分组统计
bin_analysis = data_train.groupby(f"{feature_name}_bin")[target].agg(
    样本数="count",
    还款率="mean"
).reset_index()

# 关键修复：从Categorical结果中获取分箱区间
bin_edges = qcut_result.cat.categories  # 现在可以正确获取区间
bin_analysis["分箱区间"] = [f"{edge.left:.2f}-{edge.right:.2f}" for edge in bin_edges]

print(f"\n=== {feature_name} 分箱与还款率关联 ===")
print(bin_analysis[["分箱区间", "样本数", "还款率"]].round(4))

# 4.2 可视化还款率趋势
plt.figure(figsize=(12, 6))
sns.barplot(
    data=bin_analysis,
    x="分箱区间",
    y="还款率",
    palette="Reds_r"  # 颜色越深表示还款率越低
)
# 标注样本数
for i, row in bin_analysis.iterrows():
    plt.text(
        i, row["还款率"] + 0.01,
        f"n={row['样本数']}",
        ha="center",
        fontsize=8
    )
plt.title(f"{feature_name} 分箱与还款率的关系 (IV=0.6646)")
plt.xlabel(f"{feature_name} 分箱区间")
plt.ylabel("还款率")
plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# 5. 验证分箱合理性（计算分箱后IV值）
def calculate_iv(data, feature, target):
    """计算特征的IV值"""
    df = data[[feature, target]].copy()
    # 计算每个分组的好坏样本数（假设1=还款，0=未还款）
    iv_table = df.groupby(feature)[target].agg(
        坏样本数=lambda x: (1 - x).sum(),
        好样本数=lambda x: x.sum()
    ).reset_index()
    # 计算IV
    total_bad = iv_table["坏样本数"].sum()
    total_good = iv_table["好样本数"].sum()
    iv_table["坏占比"] = iv_table["坏样本数"] / total_bad
    iv_table["好占比"] = iv_table["好样本数"] / total_good
    iv_table["WOE"] = np.log(iv_table["坏占比"] / iv_table["好占比"].replace(0, 1e-10))  # 避免除0
    iv_table["IV"] = (iv_table["坏占比"] - iv_table["好占比"]) * iv_table["WOE"]
    return iv_table["IV"].sum()

# 计算分箱后的IV值
bin_iv = calculate_iv(data_train, f"{feature_name}_bin", target)
print(f"\n分箱后 {feature_name} 的IV值：{bin_iv:.4f}（原始IV=0.6646）")