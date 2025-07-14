import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 数据准备
ranges = [
    "[0.628, 0.760]", "[0.760, 0.850]", "[0.850, 0.883]", "[0.883, 0.909]",
    "[0.909, 0.919]", "[0.919, 0.939]", "[0.939, 0.970]", "[0.970, 0.977]",
]
accs = [0.6242, 0.7278, 0.9583, 0.9680, 0.9590, 0.9735, 0.9562, 0.9790]

# 创建专业配色方案 (蓝到红的渐变)
colors = LinearSegmentedColormap.from_list('custom', ['#1f77b4', '#ff7f0e'])(np.linspace(0, 1, len(accs)))

# 创建水平条形图
plt.figure(figsize=(10, 6), dpi=150)
bars = plt.barh(ranges, accs, color=colors, edgecolor='black', alpha=0.85)

# 添加数据标签
for i, (bar, acc) in enumerate(zip(bars, accs)):
    plt.text(acc + 0.01, i, f"{acc:.4f}", 
             va='center', fontsize=10, fontweight='bold')

# 添加趋势线
x_vals = np.arange(len(accs))
trend_line = np.poly1d(np.polyfit(x_vals, accs, 2))(x_vals)
plt.plot(trend_line, ranges, 'k--', linewidth=1.5, alpha=0.7)

# 美化图表
plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Confidence Range', fontsize=12, fontweight='bold')
plt.title('Clustering Accuracy Across Confidence Segments', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlim(0.6, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.3)

# 添加颜色条说明
sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('custom', ['#1f77b4', '#ff7f0e']))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', shrink=0.8)
cbar.set_label('Accuracy Level', rotation=270, labelpad=15, fontsize=10)

# 调整布局并保存
plt.tight_layout()
plt.savefig('acc_confidence_bins_improved.pdf', bbox_inches='tight', dpi=300)
plt.show()