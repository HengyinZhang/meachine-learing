import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 数据读取与预处理（适配你的Excel文件）
# 读取Excel文件（假设数据在"评分卡结果"工作表，若不同需修改sheet_name）
df = pd.read_excel("C:/Users/zhang/Desktop/逻辑回归.xlsx", sheet_name="Sheet1")  # 替换为你的实际sheet名

# 数据清洗（确保关键列存在，处理可能的空值）
required_cols = ["特征", "IV值", "Bin", "WOE", "原始系数", "评分贡献"]
df = df[required_cols].dropna(subset=["WOE", "IV值"])  # 删除WOE/IV为空的行
df["WOE"] = pd.to_numeric(df["WOE"], errors="coerce")  # 确保WOE为数值型
df["IV值"] = pd.to_numeric(df["IV值"], errors="coerce")
df["评分贡献"] = pd.to_numeric(df["评分贡献"], errors="coerce")

# -------------------------- 2. 全局绘图设置（美观性优化）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 支持英文（中文用"SimHei"）
plt.rcParams["axes.unicode_minus"] = False  # 显示负号
plt.style.use("seaborn-v0_8-whitegrid")  # 绘图风格
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]  # 配色


# -------------------------- 3. 图1：特征IV值排序图（体现区分能力）
def plot_feature_iv(df):
    # 计算每个特征的IV值（去重，取唯一IV）
    iv_summary = df.groupby("特征")["IV值"].first().sort_values(ascending=True).reset_index()
    
    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(iv_summary["特征"], iv_summary["IV值"], color=colors[:len(iv_summary)])
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", ha="left", va="center", fontsize=9)
    
    # 图表标注
    ax.set_xlabel("IV Value (Information Value)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Feature Discrimination Ability (IV Value Ranking)\n(High IV = Strong Risk Distinction)", 
                 fontsize=14, pad=20)
    ax.axvline(x=0.1, color="red", linestyle="--", alpha=0.7, label="IV=0.1 (Weak)")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.7, label="IV=0.5 (Strong)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("/mnt/feature_iv_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()

# 执行绘图
plot_feature_iv(df)


# -------------------------- 4. 图2：关键特征WOE分布柱状图（以employment_status为例）
def plot_key_feature_woe(df, target_feature="employment_status"):
    # 筛选目标特征数据
    feature_data = df[df["特征"] == target_feature].sort_values("WOE").reset_index(drop=True)
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(feature_data["Bin"], feature_data["WOE"], 
                  color=["green" if w < 0 else "red" for w in feature_data["WOE"]])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.1 if height > 0 else -0.3),
                f"{height:.2f}", ha="center", va="bottom" if height > 0 else "top", fontsize=10)
    
    # 图表标注（突出WOE正负=风险高低）
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, label="WOE=0 (Neutral Risk)")
    ax.set_xlabel(f"{target_feature} Categories", fontsize=12)
    ax.set_ylabel("WOE (Weight of Evidence)", fontsize=12)
    ax.set_title(f"WOE Distribution of {target_feature}\n(Green=WOE<0=Low Risk; Red=WOE>0=High Risk)", 
                 fontsize=14, pad=20)
    ax.legend()
    plt.xticks(rotation=45, ha="right")  # 旋转x标签，避免重叠
    
    plt.tight_layout()
    plt.savefig("/mnt/employment_status_woe.png", dpi=300, bbox_inches="tight")
    plt.close()

# 执行绘图（聚焦IV最高的employment_status）
plot_key_feature_woe(df, target_feature="employment_status")


# -------------------------- 5. 图3：评分贡献热力图（以debt_to_income_ratio为例）
def plot_score_contribution_heatmap(df, target_feature="debt_to_income_ratio"):
    # 筛选目标特征数据，整理为"分箱-WOE-评分贡献"的矩阵
    feature_data = df[df["特征"] == target_feature].sort_values("WOE").reset_index(drop=True)
    # 创建热力图数据（分箱为行，指标为列）
    heatmap_data = feature_data[["Bin", "WOE", "评分贡献"]].set_index("Bin")
    # 重命名列，方便理解
    heatmap_data.columns = ["WOE (Risk)", "Score Contribution (Deduction)"]
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".2f", cmap="RdYlGn_r", 
                ax=ax, cbar_kws={"label": "Value"})
    
    # 图表标注
    ax.set_xlabel(f"{target_feature} Bins", fontsize=12)
    ax.set_ylabel("Indicator", fontsize=12)
    ax.set_title(f"Score Contribution Heatmap of {target_feature}\n(Blue=Low Risk/ Less Deduction; Red=High Risk/ More Deduction)", 
                 fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("/mnt/debt_ratio_score_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

# 执行绘图
plot_score_contribution_heatmap(df, target_feature="debt_to_income_ratio")


# -------------------------- 6. 图4：WOE与评分贡献散点图（验证逻辑一致性）
def plot_woe_vs_score(df):
    # 筛选核心特征（IV>0.3的特征，避免数据过多）
    high_iv_features = df.groupby("特征")["IV值"].first()[df.groupby("特征")["IV值"].first() > 0.3].index.tolist()
    plot_data = df[df["特征"].isin(high_iv_features)].reset_index(drop=True)
    
    # 创建散点图（按特征着色）
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, feature in enumerate(high_iv_features):
        feature_data = plot_data[plot_data["特征"] == feature]
        ax.scatter(feature_data["WOE"], feature_data["评分贡献"], 
                   label=feature, color=colors[i], alpha=0.7, s=60)
    
    # 添加趋势线（体现整体负相关：WOE越负→评分贡献越负=扣减少）
    z = np.polyfit(plot_data["WOE"], plot_data["评分贡献"], 1)
    p = np.poly1d(z)
    ax.plot(plot_data["WOE"], p(plot_data["WOE"]), "k--", alpha=0.8, 
            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    
    # 图表标注
    ax.set_xlabel("WOE (Weight of Evidence)", fontsize=12)
    ax.set_ylabel("Score Contribution (Deduction)", fontsize=12)
    ax.set_title("WOE vs Score Contribution\n(Negative Correlation: Lower WOE = Less Score Deduction)", 
                 fontsize=14, pad=20)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # 图例放右侧，避免遮挡
    
    plt.tight_layout()
    plt.savefig("/mnt/woe_score_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

# 执行绘图
plot_woe_vs_score(df)

print("4类图表已生成，保存路径：/mnt/")
print("1. feature_iv_ranking.png（特征IV排序）")
print("2. employment_status_woe.png（就业状态WOE分布）")
print("3. debt_ratio_score_heatmap.png（债务收入比评分热力图）")
print("4. woe_score_correlation.png（WOE与评分贡献散点图）")