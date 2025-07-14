import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

methods = ['Random Fusion', 'Fixed Fusion', 'Confidence-weighted']
accs_cifar10 = [0.690, 0.849, 0.913]
aris_cifar10 = [0.608, 0.755, 0.826]
nmis_cifar10 = [0.732, 0.818, 0.858]

accs_stl10 = [0.448, 0.641, 0.814]
aris_stl10 = [0.277, 0.481, 0.741]
nmis_stl10 = [0.514, 0.674, 0.828]

x = np.arange(len(methods))
width = 0.22
colors = ['#6baed6', '#74c476', '#fd8d3c']

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120, sharey=True)

for idx, (ax, accs, aris, nmis, title) in enumerate([
    (axes[0], accs_cifar10, aris_cifar10, nmis_cifar10, "CIFAR10"),
    (axes[1], accs_stl10, aris_stl10, nmis_stl10, "STL10")
]):
    bars1 = ax.bar(x - width, accs, width, label='ACC', color=colors[0], edgecolor='white', alpha=0.85, zorder=2)
    bars2 = ax.bar(x, aris, width, label='ARI', color=colors[1], edgecolor='white', alpha=0.85, zorder=2)
    bars3 = ax.bar(x + width, nmis, width, label='NMI', color=colors[2], edgecolor='white', alpha=0.85, zorder=2)

    # 折线+白边
    line1, = ax.plot(x - width, accs, color=colors[0], marker='o', linewidth=2, zorder=3)
    line2, = ax.plot(x, aris, color=colors[1], marker='s', linewidth=2, zorder=3)
    line3, = ax.plot(x + width, nmis, color=colors[2], marker='^', linewidth=2, zorder=3)
    for line in [line1, line2, line3]:
        line.set_path_effects([pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])

    # 数值标注
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

axes[0].set_ylabel('Score', fontsize=14)
axes[1].legend(fontsize=13, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
plt.tight_layout()
plt.savefig('confidence_bar_compare.pdf')
plt.show()