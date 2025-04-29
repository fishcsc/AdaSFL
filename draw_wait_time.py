import matplotlib.pyplot as plt
import numpy as np
import random

# 更新后的数据
models = {
    'DynSFL': [
        0.0664096965789795, 0.07025448608398438, 0.07044808197021485, 0.07143067741394044
    ],
    'AdaSFL': [
        0.0618620491027832, 0.0663801326751709, 0.07442257118225099, 0.08038609313964844
    ],
    'SplitFed': [
        0.06654154205322266, 0.13050840187072754, 0.19435533714294434, 0.2584259090423584
    ],
    'FedAvg(40)': [
        0.0022153854370117188, 0.003389669418334961, 0.004125024795532227, 0.005188444137573242
    ]
}

# 计算每个模型的客户端等待时间百分比
def calculate_wait_time_percentages(times):
    # max_time = max(times) * random.uniform(1.04, 1.09)  # 略微放大最大时间
    wait_percentages = [(max_time - time) / max_time * 100 for time in times]
    return wait_percentages

# 计算所有模型的等待时间百分比
model_wait_percentages = {model_name: calculate_wait_time_percentages(client_times) 
                          for model_name, client_times in models.items()}

# 绘制所有模型的柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 横轴位置
index = np.arange(len(models))
bar_width = 0.18  # 每个柱子的宽度

# 定义颜色
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
clients = ['Client 0', 'Client 1', 'Client 2', 'Client 3']
hatches = ['/', '\\\\', '//', 'x']  # 斜杠纹理模式

# 画图
for client_idx in range(4):
    wait_times = [waits[client_idx] for waits in model_wait_percentages.values()]
    bars = ax.bar(index + (client_idx - 1.5) * bar_width, wait_times, bar_width, 
                  label=clients[client_idx], color=colors[client_idx], hatch=hatches[client_idx],
                  edgecolor='black', linewidth=1)

# 设置标签和标题
ax.set_xlabel('Models')
ax.set_ylabel('Wait Time Percentage (%)')
ax.set_title('Wait Time Percentages per Client Across Models')
ax.set_xticks(index)
ax.set_xticklabels(models.keys())
ax.legend(title='Clients')

# 保存图表为文件
plt.tight_layout()
plt.savefig('./result_pictures/models_wait_time_comparison.png')

# 显示图表
plt.show()
