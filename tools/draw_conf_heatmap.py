import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10'])
args = parser.parse_args()

data = np.load(f"confidence_heatmap_data_{args.dataset}.npz", allow_pickle=True)
heatmap = data["heatmap"]
annot_arr = data["annot_arr"]
bins = data["bins"]
class_names = data["class_names"]

# 保证 ticklabels 长度和 heatmap 维度一致
xticklabels = list(class_names)
yticklabels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(heatmap.shape[0])]

plt.figure(figsize=(12, 8))
cmap = LinearSegmentedColormap.from_list("my_cmap", ["#f7fbff", "#08306b"])

# 创建热力图，将colorbar放在下方
ax = sns.heatmap(
    heatmap,
    annot=annot_arr,
    fmt="",
    cmap=cmap,
    square=True,
    xticklabels=xticklabels,
    yticklabels=yticklabels,
    cbar_kws={"shrink": 0.8, "pad": 0.02, "location": "right"}
)

# 调整colorbar标签位置
cbar = ax.collections[0].colorbar
cbar.ax.xaxis.set_ticks_position('bottom')

plt.ylabel("Confidence Interval")
ax.set_yticklabels(yticklabels, rotation=0, ha="right", fontsize=12)
ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=12)

# 减少边距
plt.tight_layout(pad=0.5)  # 默认pad=1.08，减小此值以减少留白
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)  # 手动调整边距

# 保存（bbox_inches='tight'进一步裁剪空白）
plt.savefig(f"confidence_heatmap_{args.dataset}.pdf", bbox_inches='tight', dpi=300)
# plt.show()