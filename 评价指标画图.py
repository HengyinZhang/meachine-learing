import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
data = {
    '模型名称': ['逻辑回归', '决策树（默认参数）', '决策树（网格搜索最优）', 
               'XGBoost（默认参数）', 'XGBoost（网格搜索最优）', 
               'LightGBM（默认参数）', 'LightGBM（网格搜索最优）'],
    'AUC': [0.9117, 0.7696, 0.9119, 0.921, 0.9117, 0.9197, 0.9223],
    'Accuracy': [0.9017, 0.8463, 0.9015, 0.9038, 0.9038, 0.9045, 0.9052],
    'Precision': [0.9082, 0.9086, 0.905, 0.9079, 0.9079, 0.9068, 0.9087],
    'KS': [0.6578, 0.5392, 0.6633, 0.6775, 0.6775, 0.6764, 0.6816]
}

df = pd.DataFrame(data)

# 设置图形大小
plt.figure(figsize=(12, 8))

# 定义柱状图位置和宽度
metrics = ['AUC', 'Accuracy', 'Precision', 'KS']
x = np.arange(len(metrics))  # 评价指标的位置
width = 0.1  # 每个柱子的宽度

# 为每个模型绘制柱子
for i, model in enumerate(df['模型名称']):
    plt.bar(x + i*width, df.loc[df['模型名称']==model, metrics].values[0], 
            width=width, label=model, alpha=0.8)

# 设置坐标轴
plt.xlabel('评价指标', fontsize=12)
plt.ylabel('', fontsize=12)
plt.xticks(x + width*3, metrics, fontsize=10)  # 居中显示x轴标签
plt.ylim(0.5, 0.95)  # 根据数据范围调整y轴范围，使差异更明显

# 添加图例
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()