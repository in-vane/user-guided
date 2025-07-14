import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

methods = [' No LoRA', 'Standard LoRA', 'Bayesian-LoRA']

# lora
accs_cifar10 = [0.778, 0.810, 0.916]
aris_cifar10 = [0.702, 0.762, 0.831]
nmis_cifar10 = [0.813, 0.843, 0.863]
accs_cifar10_hab = [0.778, 0.821, 0.863]
aris_cifar10_hab = [0.486, 0.650, 0.670]
nmis_cifar10_hab = [0.375, 0.456, 0.500]

x = np.arange(len(methods))
width = 0.27
# colors = ['#6baed6', '#74c476', '#fd8d3c']
colors = ['#4e79a7', '#59a14f', '#e15759']

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120, sharey=True)

for idx, (ax, accs, aris, nmis, title) in enumerate([
    (axes[0], accs_cifar10, aris_cifar10, nmis_cifar10, "Standard"),
    # (axes[1], accs_stl10, aris_stl10, nmis_stl10, "STL10")
    (axes[1], accs_cifar10_hab, aris_cifar10_hab, nmis_cifar10_hab, "Criteria-Habitat")
]):
    bars1 = ax.bar(x - width, accs, width, label='ACC', color=colors[0], edgecolor='white', alpha=0.85, zorder=2)
    bars2 = ax.bar(x, aris, width, label='ARI', color=colors[1], edgecolor='white', alpha=0.85, zorder=2)
    bars3 = ax.bar(x + width, nmis, width, label='NMI', color=colors[2], edgecolor='white', alpha=0.85, zorder=2)

    # 折线+白边
    # line1, = ax.plot(x - width, accs, color=colors[0], marker='o', linewidth=1, zorder=2)
    # line2, = ax.plot(x, aris, color=colors[1], marker='s', linewidth=1, zorder=2)
    # line3, = ax.plot(x + width, nmis, color=colors[2], marker='^', linewidth=1, zorder=2)
    # for line in [line1, line2, line3]:
    #     line.set_path_effects([pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

    # 数值标注
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
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
# axes[1].legend(fontsize=13, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
axes[1].legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              frameon=True, framealpha=0.9, edgecolor='gray')
plt.tight_layout()
plt.savefig('lora_bar_compare.pdf')
# plt.show()