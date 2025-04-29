import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Non-IID Level（将第一个值改为 0.1）
non_iid_levels = [0.1, 0.2, 0.4, 0.6, 0.8]

# 每个模型对应不同 Non-IID 下的准确率
results = {
    'DynSFL':     [0.8205, 0.8054, 0.8013, 0.7564, 0.7012],
    'AdaSFL':     [0.8102, 0.8081, 0.7904, 0.6974, 0.6582],
    'FedAvg(40)': [0.7806, 0.7760, 0.7245, 0.6201, 0.5901],
    'SplitFed':   [0.8056, 0.73,   0.6975, 0.69,   0.2580]
}

# 样式配置
colors = ['red', 'green', 'blue', 'purple']
markers = ['s', 'o', 'd', '^']
linestyles = ['-', '-', '-', '-']

fig, ax = plt.subplots(figsize=(4, 3))  # 控制图像大小

for (label, accs), color, marker, ls in zip(results.items(), colors, markers, linestyles):
    ax.plot(non_iid_levels, accs, label=label,
            color=color, marker=marker, linestyle=ls, linewidth=1.5, markersize=5)

# 坐标轴标签
ax.set_xlabel('Non-IID Level', fontsize=12)
ax.set_ylabel('Test accuracy', fontsize=12)

# 图例样式
ax.legend(frameon=True, loc='lower left', fontsize=9)

# 网格和紧凑布局
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('./result_pictures/test_accuracy_vs_noniid.png', dpi=300)
plt.close()
