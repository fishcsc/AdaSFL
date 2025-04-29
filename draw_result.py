import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 文件路径列表
file_list = [
    'result_record/2025-04-29-15_27_44_record.txt',
    'result_record/2025-04-29-15_37_50_record.txt',
    'result_record/2025-04-29-16_35_39_record.txt',
    'result_record/2025-04-29-16_07_10_record.txt'
]

# 模型名映射
model_names = {
    '2025-04-29-15_27_44_record.txt': 'DynSFL',
    '2025-04-29-15_37_50_record.txt': 'AdaSFL',
    '2025-04-29-16_35_39_record.txt': 'FedAvg(40)',
    '2025-04-29-16_07_10_record.txt': 'SplitFed'
}

# 准备画布
fig, ax = plt.subplots()

# 色彩配置
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']

flag = 1  # 是否开启SplitFed的epoch跳采样（每40轮）
for i, file_name in enumerate(file_list):
    epochs, times, accs = [], [], []

    with open(file_name, 'r') as file:
        next(file)  # skip model param
        next(file)  # skip header
        for line in file:
            values = line.split()
            if len(values) > 5 and values[0].isdigit():
                epoch = int(values[0])
                time = float(values[1])
                acc = float(values[4])
                if epoch % 1 == 0:
                    if flag:
                        if i != 3:
                            epochs.append(epoch)
                            times.append(time)
                            accs.append(acc)
                        elif epoch % 40 == 0:
                            epochs.append(epoch)
                            times.append(time)
                            accs.append(acc)
                    else:
                        epochs.append(epoch)
                        times.append(time)
                        accs.append(acc)

    base_name = os.path.basename(file_name)
    model_label = model_names.get(base_name, base_name)

    ax.plot(times, accs, color=colors[i % len(colors)], linestyle='-', label=model_label)

# 设置标签
ax.set_xlabel('Time')
ax.set_ylabel('Accuracy')
plt.title('Accuracy vs. Time for Different Models')

# 图例
ax.legend(loc='lower right')

# 保存图
plt.tight_layout()
output_path = './result_pictures/multi_model_accuracy.png'
plt.savefig(output_path)
plt.close()


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import os

# # 文件路径列表
# file_list = [
#     'result_record/2025-04-20-11_10_58_record.txt',
#     'result_record/2025-04-12-12_19_26_record.txt',
#     'result_record/2025-04-20-18_48_19_record.txt',
#     'result_record/2025-04-21-18_53_19_record.txt'
# ]

# # 模型名映射
# model_names = {
#     '2025-04-20-11_10_58_record.txt': 'DynSFL',
#     '2025-04-12-12_19_26_record.txt': 'AdaSFL',
#     '2025-04-20-18_48_19_record.txt': 'FedAvg(40)',
#     '2025-04-21-18_53_19_record.txt': 'SplitFed'
# }

# # 准备画布
# fig, ax = plt.subplots()

# # 色彩与标记配置
# colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
# markers = ['s', 'o', 'd', '^']

# flag = 1  # 是否开启SplitFed的epoch跳采样（每40轮）

# for i, file_name in enumerate(file_list):
#     epochs, times, accs = [], [], []

#     with open(file_name, 'r') as file:
#         next(file)  # skip model param
#         next(file)  # skip header
#         prev_time = -10  # 初始化为小于0，保证第一个点被记录
#         for line in file:
#             values = line.split()
#             if len(values) > 5 and values[0].isdigit():
#                 epoch = int(values[0])
#                 time = float(values[1])
#                 acc = float(values[4])

#                 # 每10s取一个点
#                 if time - prev_time >= 100:
#                     if flag:
#                         if i != 3 or epoch % 40 == 0:
#                             epochs.append(epoch)
#                             times.append(time)
#                             accs.append(acc)
#                             prev_time = time
#                     else:
#                         epochs.append(epoch)
#                         times.append(time)
#                         accs.append(acc)
#                         prev_time = time

#     base_name = os.path.basename(file_name)
#     model_label = model_names.get(base_name, base_name)

#     ax.plot(times, accs,
#             color=colors[i % len(colors)],
#             linestyle='-',
#             marker=markers[i % len(markers)],
#             markersize=4,
#             linewidth=1.5,
#             label=model_label)

# # 设置标签
# ax.set_xlabel('Time')
# ax.set_ylabel('Accuracy')
# plt.title('Accuracy vs. Time for Different Models')

# # 图例
# ax.legend(loc='lower right')

# # 网格与布局
# ax.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()

# # 保存图
# output_path = './result_pictures/multi_model_accuracy.png'
# plt.savefig(output_path, dpi=300)
# plt.close()
