import matplotlib
# 设置非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 初始化存储数据的列表
epochs = []
accs = []
losses = []

# 读取文件
file_name = 'result_record/2025-04-08-15_37_27_record.txt'
with open(file_name, 'r') as file:
    # 跳过第一行的模型参数信息
    next(file)
    # 跳过第二行的表头信息
    next(file)
    for line in file:
        # 分割每行数据
        values = line.split()
        # 检查是否为有效的数据行
        if len(values) > 5 and values[0].isdigit():
            # 提取 epoch、acc 和 loss
            epoch = int(values[0])
            acc = float(values[4])
            loss = float(values[5])
            # 将数据添加到对应的列表中
            epochs.append(epoch)
            accs.append(acc)
            losses.append(loss)

# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制 acc 曲线
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(epochs, accs, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个 y 轴来绘制 loss 曲线
ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
ax2.plot(epochs, losses, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 添加标题
plt.title('Accuracy and Loss over Epochs')

# 获取文件名（不包含扩展名）
base_name = os.path.splitext(os.path.basename(file_name))[0]
# 生成图片文件名
image_name = f'./result_pictures/{base_name}_accuracy_loss.png'

# 保存图片
plt.savefig(image_name)

# 关闭图形
plt.close()
    