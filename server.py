import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.optim as optim
import torch.nn.functional as F
import datasets, models
from training_utils import test2
import torch.nn as nn
from comm_utils import *
import pdb



#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='FashionMNIST')
parser.add_argument('--model_type', type=str, default='CNN')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_pattern', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=1000) # set to 10000 for splite_sfl
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
master_listen_port_base=53710
RESULT_PATH = 'result_record'

def main():

    # init config
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.weight_decay = args.weight_decay

    #read the worker_config.json to init the worker node
    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    worker_num = len(workers_config['worker_config_list'])

    client_model,global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(client_model.parameters())
    global_model.to(device)
    client_model = client_model.to(device)

    common_config.para_nums=init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model: {}".format(common_config.model_type))
    print("客户端:")
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))    

    init_para1 = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para1.nelement() * 4 / 1024 / 1024
    print("服务器:")
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))
    # create workers
    worker_list: List[Worker] = list()
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config,custom=custom),
                    idx=worker_idx,
                    client_ip=worker_config['ip_address'],
                    user_name=worker_config['user_name'],
                    pass_wd=worker_config['pass_wd'],
                    remote_scripts_path=workers_config['scripts_path']['remote'],
                    # master_port=master_listen_port_base+worker_idx,
                    master_port=worker_config['master_port'],
                    location='local'
                    )
        )
    #到了这里，worker已经启动了

    # Create model instance
    train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern, worker_num)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        # worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(worker_list, action="init")
    # pdb.set_trace()

    recoder: SummaryWriter = SummaryWriter()
    global_model.to(device)
    _, test_dataset = datasets.load_datasets(common_config.dataset_type)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    total_time=0.0
    local_steps_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    total_resource=0.0
    total_bandwith=0.0
    #computation_resource,bandwith_resource=random_RC(10)

    path=os.getcwd()
    print (path)
    path=path+"//"+RESULT_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    path=path+"//"+now+"_record.txt"
    result_out = open(path, 'a+')

    #result_out.write(common_config.__dict__)
    print(common_config.__dict__,file=result_out)
    result_out.write('\n')
    result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    result_out.write('\n')
    local_steps=1

    epoch_lr = args.lr
    for epoch_idx in range(1, 1+common_config.epoch):

        start_time = time.time()
        communication_parallel(worker_list, action="send_local_steps", data=local_steps)
        print("epoch {} send local steps ({}) to workers done".format(epoch_idx, local_steps))
        warmup_epochs = 10
        base_lr = common_config.lr


        if epoch_idx <= warmup_epochs:
            epoch_lr = base_lr * epoch_idx / warmup_epochs
        else:
            decay_epoch = epoch_idx - warmup_epochs
            epoch_lr = max(base_lr * (args.decay_rate ** decay_epoch), args.min_lr)
        optimizer = optim.SGD(global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        total_resource,total_bandwith,total_time=train2(global_model, device, worker_list,epoch_lr, local_steps,total_resource,total_bandwith,total_time)

        print("get begin")
        communication_parallel(worker_list, action="get_para")
        communication_parallel(worker_list, action="get_time")
        print("get end")
        
        
        para_delta = aggregate_model_para2(client_model, worker_list, device)
        # global_para = aggregate_compressed_model(global_model,worker_list)
        
        print("send begin")
        communication_parallel(worker_list, action="send_model",data=para_delta)
        print("send end")
        
        test_loss, acc = test2(global_model,client_model, test_loader, device)
        recoder.add_scalar('Accuracy/average', acc, epoch_idx)
        recoder.add_scalar('Test_loss/average', test_loss, epoch_idx)
        print("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))

        local_steps,sum_time=update_E(worker_list)
        # total_time+=sum_time 
        end_time = time.time()
        total_time += end_time-start_time
        # print("total_time:", total_time)
        # print("sum_time:",sum_time)
        total_resource=total_resource+Sum(computation_resource,local_steps_list)
        total_bandwith=total_bandwith+Sum(bandwith_resource,compre_ratio_list)
        print("total_time: {}, total_resource: {}, total_bandwith: {}\n".format(total_time,total_resource,total_bandwith))
        recoder.add_scalar('Accuracy/average_time', acc, total_time)
        recoder.add_scalar('Test_loss/average_time', test_loss, total_time)
        recoder.add_scalar('resource_time', total_resource, total_time)
        recoder.add_scalar('bandwith_time', total_bandwith, total_time)
        recoder.add_scalar('resource_epoch', total_resource, epoch_idx)
        recoder.add_scalar('bandwith_epoch', total_bandwith, epoch_idx)
        result_out.write('{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx,total_time,total_bandwith,total_resource,acc,test_loss))
        result_out.write('\n')

        print(local_steps_list)
        print(compre_ratio_list)
        
    # close socket
    result_out.close()
    for worker in worker_list:
        worker.socket.shutdown(2)
        worker.client.close()
    
def Sum(list1,list2):
    sum=0.0
    for idx in range(0, len(list1)):
        sum=sum+float(list1[idx])*float(list2[idx])
    return sum

def random_RC(num):
    computation_resource=np.random.randint(1,num,num)
    bandwith_resourc=np.random.randint(1,num,num)
    return computation_resource,bandwith_resourc

def update_E(worker_list):
    '''重点是更新local_steps
    
    '''
    # local_steps = 40
    local_steps = 1 # splite_sfl
    compre_ratio = local_steps / 200.0
    train_time_list = [0.8, 0.7, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6]
    send_time_list = [0.8, 0.7, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6]
    min_train_time = float('inf')
    min_train_time_idx = -1
    min_send_time = float('inf')
    min_send_time_idx = -1
    sum_local_steps = 0

    # Ensure no division by zero
    epsilon = 1e-6

    for worker in worker_list:
        train_time = max(worker.config.train_time, epsilon)
        send_time = max(worker.config.send_time, epsilon)
        train_time_list[worker.idx] = train_time
        send_time_list[worker.idx] = send_time

        if train_time < min_train_time:
            min_train_time = train_time
            min_train_time_idx = worker.idx
        if send_time < min_send_time:
            min_send_time = send_time
            min_send_time_idx = worker.idx

    for worker in worker_list:
        # worker.config.batch_size=int((train_time_list[min_train_time_idx]/train_time_list[worker.idx])*local_steps)
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.idx])*compre_ratio
        # worker.config.batch_size=5
        #worker.config.batch_size=int(local_steps/2)+3
        # worker.config.compre_ratio=1.0
        # local_steps_list[worker.idx]=worker.config.batch_size
        # compre_ratio_list[worker.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.batch_size
    for worker in worker_list:
        worker.config.average_weight = (1.0 * worker.config.batch_size) / sum_local_steps
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    total_time=max_train_time
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return local_steps,total_time


def update_B(worker_list, batch_size_list, compre_ratio_list):
    """ 重点是更新批次大小
    更新各worker的批次大小和压缩比，并计算总训练时间
    
    根据各worker的训练时间和传输时间动态调整配置参数，实现负载均衡
    
    Args:
        worker_list: Worker对象列表，每个对象需包含config配置属性和idx索引属性
        batch_size_list: list类型，用于记录各worker的批次大小配置
        compre_ratio_list: list类型，用于记录各worker的压缩比配置
        
    Returns:
        tuple: 包含两个元素的元组
        - batch_size_list: 更新后的批次大小配置列表
        - total_time: 计算得出的预估总训练时间
    """
    batch = 64
    compre_ratio = batch / 200.0
    train_time_list = [0.8, 0.7, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6]
    send_time_list = [0.8, 0.7, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6]
    min_train_time = float('inf')
    min_train_time_idx = -1
    min_send_time = float('inf')
    min_send_time_idx = -1
    sum_local_steps = 0

    # Ensure no division by zero
    epsilon = 1e-6

    tmp = ""
    for idx, worker in enumerate(worker_list):
        train_time = max(worker.config.train_time, epsilon)
        send_time = max(worker.config.send_time, epsilon)
        train_time_list[worker.idx] = train_time
        send_time_list[worker.idx] = send_time
        tmp += " " + str(idx) + ": " + str(train_time) 

        if train_time < min_train_time:
            min_train_time = train_time
            min_train_time_idx = worker.idx
        if send_time < min_send_time:
            min_send_time = send_time
            min_send_time_idx = worker.idx
    print(tmp)
    for worker in worker_list:
        #动态更新批次大小
        worker.config.batch_size=int((train_time_list[min_train_time_idx]/train_time_list[worker.idx])*batch)
        #固定批次大小
        # worker.config.batch_size=batch
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.idx])*compre_ratio
        # worker.config.batch_size=5
        #worker.config.batch_size=int(local_steps/2)+3
        # worker.config.compre_ratio=1.0
        batch_size_list[worker.idx]=worker.config.batch_size
        compre_ratio_list[worker.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.batch_size
    for worker in worker_list:
        worker.config.average_weight = (1.0 * worker.config.batch_size) / sum_local_steps
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    # total_time=max_train_time*4
    total_time=max_train_time
    
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return batch_size_list, total_time

def aggregate_model_para(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para)
        for worker in worker_list:
            model_delta = (worker.config.neighbor_paras - global_para)
            para_delta += worker.config.average_weight * model_delta
            #print(para_delta)
        global_para += para_delta
    torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para

def aggregate_model_dict(global_model,worker_list):
    with torch.no_grad():
        local_model_para = []
        for worker in worker_list:
            local_model_para.append(worker.config.neighbor_paras)
        para_delta = copy.deepcopy(local_model_para[0])
        for para in para_delta.keys():
            para_delta[para] = para_delta[para]*0.0
            for p in local_model_para:
                para_delta[para] = para_delta[para]*1.0 + p[para]*1.0
            para_delta[para] = para_delta[para] / (len(local_model_para))
    global_model.load_state_dict(para_delta)
    return para_delta

def aggregate_compressed_model(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para)
        for worker in worker_list:
            indice = worker.config.neighbor_indices
            selected_indicator = torch.zeros_like(global_para)
            selected_indicator[indice] = 1.0
            # model_delta = (worker.config.neighbor_paras - global_para) * selected_indicator
            #gradient
            model_delta = worker.config.neighbor_paras
            para_delta += worker.config.average_weight * model_delta
        global_para += para_delta
    torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para

# def communication_parallel(worker_list, action, data=None):
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
#         tasks = []
#         for worker in worker_list:
#             if action == "init":
#                 tasks.append(loop.run_in_executor(executor, worker.send_init_config))
#             elif action == "get_para":
#                 tasks.append(loop.run_in_executor(executor, get_model,worker))
#             elif action == "get_time":
#                 tasks.append(loop.run_in_executor(executor, get_time,worker))
#             elif action == "get_data_feature":
#                 tasks.append(loop.run_in_executor(executor, get_data_feature,worker))
#             elif action == "send_model":
#                 tasks.append(loop.run_in_executor(executor, worker.send_data, data))
#             elif action == "send_para":
#                 data=worker.config.batch_size
#                 tasks.append(loop.run_in_executor(executor, worker.send_data,data))
#         loop.run_until_complete(asyncio.wait(tasks))
#         loop.close()
#     except:
#         sys.exit(0)

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                # 发送初始配置和打招呼
                tasks.append(loop.run_in_executor(executor, send_init_config_with_greeting, worker))
            elif action == "get_para":
                tasks.append(loop.run_in_executor(executor, get_model,worker))
            elif action == "get_time":
                tasks.append(loop.run_in_executor(executor, get_time,worker))
            elif action == "get_data_feature":
                tasks.append(loop.run_in_executor(executor, get_data_feature,worker))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, send_data, worker, data))
            elif action == "send_batch_size":
                data=worker.config.batch_size
                # print("发送batch_size: ", data)
                tasks.append(loop.run_in_executor(executor, send_data, worker, data))
            elif action == "send_local_steps":
                # data=worker.config.local_steps
                tasks.append(loop.run_in_executor(executor, send_data, worker, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)


def send_data(worker, data):
    worker.client.send(data)

def send_init_config_with_greeting(worker):
    try:
        # 服务器主动连接到worker
        worker.socket = socket.create_connection(
            (worker.client_ip, worker.master_port), timeout=5
        )
        reuse = 1
        worker.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuse)

        print(f"✅ {worker.user_name} connected to {worker.client_ip}:{worker.master_port}")
        # print(worker.socket.fileno())
        worker.client = ConnectionHandler(worker.socket, is_worker=False)
        # print(worker.socket.fileno())
        # pdb.set_trace()
        # 发送打招呼消息
        greeting = {"message": f"Hello from server to worker {worker.idx}"}
        worker.client.send(greeting)
        # send_data_socket(greeting, worker.socket)
        
        # 接收worker的响应
        # response = get_data_socket(worker.socket)
        response = worker.client.recv()
        if response:
            print(f"收到 worker {worker.idx} 的响应: {response}")
        
        # 发送实际配置
        # send_data_socket(worker.config, worker.socket)
        worker.client.send(worker.config)
        print(f"已发送配置到 worker {worker.idx}")
        
    except Exception as e:
        print(f"❌ {worker.user_name} connection failed: {str(e)}")


def get_time(worker):
    try:
        result = worker.client.recv()
        if result is None:
            print(f"Worker {worker.idx} 未收到时间数据")
            worker.config.train_time = 1.0  # 设置默认值
            worker.config.send_time = 1.0   # 设置默认值
            return
            
        if not isinstance(result, tuple) or len(result) != 2:
            print(f"Worker {worker.idx} 收到的时间数据格式错误")
            worker.config.train_time = 1.0
            worker.config.send_time = 1.0
            return
            
        train_time, send_time = result
        worker.config.train_time = max(float(train_time), 1e-6)  # 确保非零
        worker.config.send_time = max(float(send_time), 1e-6)    # 确保非零
        # print(f"Worker {worker.idx} train time: {train_time}, send time: {send_time}")
    except Exception as e:
        print(f"获取 Worker {worker.idx} 时间数据时出错: {str(e)}")
        worker.config.train_time = 1.0
        worker.config.send_time = 1.0

def get_compressed_model_top(worker):
    nelement=worker.config.common_config.para_nums
    received_para, indices = worker.client.recv()
    received_para.to(device)

    restored_model = torch.zeros(nelement).to(device)
    
    restored_model[indices] = received_para

    worker.config.neighbor_paras = restored_model.data
    worker.config.neighbor_indices = indices

def get_model(worker):
    try:
        # received_para = get_data_socket(worker.socket)
        received_para = worker.client.recv()
        if received_para is None:
            print(f"Worker {worker.idx} 未收到参数")
            worker.config.neighbor_paras = None
            return
            
        if not isinstance(received_para, torch.Tensor):
            print(f"Worker {worker.idx} 收到的参数类型错误: {type(received_para)}")
            worker.config.neighbor_paras = None
            return
            
        worker.config.neighbor_paras = received_para.to(device)
    except Exception as e:
        print(f"获取 Worker {worker.idx} 参数时出错: {str(e)}")
        worker.config.neighbor_paras = None

def non_iid_partition111(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def non_iid_partition_multi(ratio, train_class_num, worker_num, prefer_worker_num=3):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num - prefer_worker_num))
    
    for i in range(train_class_num):
        # 选择 prefer_worker_num 个客户端对该类别有偏好
        preferred_workers = [(i + j) % worker_num for j in range(prefer_worker_num)]
        for w in preferred_workers:
            partition_sizes[i][w] = ratio / prefer_worker_num

    return partition_sizes


# def partition_data(dataset_type, data_pattern, worker_num=10):
#     train_dataset, _ = datasets.load_datasets(dataset_type)

#     if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST" or dataset_type == "MNIST":
#         train_class_num=10
#         if data_pattern == 0:
#             partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
#         elif data_pattern == 1:
#             non_iid_ratio = 0.2
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 2:
#             non_iid_ratio = 0.4
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 3:
#             non_iid_ratio = 0.6
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 4:
#             non_iid_ratio = 0.8
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#     elif dataset_type == "EMNIST":
#         train_class_num=62
#         if data_pattern == 0:
#             partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
#         elif data_pattern == 1:
#             non_iid_ratio = 0.2
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 2:
#             non_iid_ratio = 0.4
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 3:
#             non_iid_ratio = 0.6
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 4:
#             non_iid_ratio = 0.8
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#     if dataset_type == "CIFAR100" or dataset_type == "image100":
#         train_class_num=100
#         if data_pattern == 0:
#             partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
#         elif data_pattern == 1:
#             non_iid_ratio = 0.2
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 2:
#             non_iid_ratio = 0.4
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 3:
#             non_iid_ratio = 0.6
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#         elif data_pattern == 4:
#             non_iid_ratio = 0.8
#             partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
#     train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
#     return train_data_partition

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type)

    if dataset_type in ["CIFAR10", "FashionMNIST", "MNIST"]:
        train_class_num = 10
    elif dataset_type == "EMNIST":
        train_class_num = 62
    elif dataset_type in ["CIFAR100", "image100"]:
        train_class_num = 100
    elif dataset_type == "tinyImageNet":
        train_class_num = 200
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    else:
        raise ValueError(f"Unsupported data_pattern: {data_pattern}")

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    return train_data_partition

def partition_data11(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100" or dataset_type == "image100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
                
    elif dataset_type == 'tinyImageNet':
        test_partition_sizes = np.ones((200, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((200, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*20
            for _ in range(int(data_pattern/10)):
                partition_sizes[tmp_idx:tmp_idx+10, worker_idx] = 0
                tmp_idx = (tmp_idx + 10) % 200
        axis = np.sum(partition_sizes, axis=1)
        for i in range(200):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "EMNIST":
        test_partition_sizes = np.ones((62, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((62, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*6
            for _ in range(int(data_pattern/2)):
                partition_sizes[tmp_idx:tmp_idx+2, worker_idx] = 0
                tmp_idx = (tmp_idx + 2) % 62
        axis = np.sum(partition_sizes, axis=1)
        for i in range(62):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "CIFAR10" or dataset_type == "FashionMNIST" or dataset_type == "MNIST":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148,0.111],
                                [0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1472,0.112],
                                [0.1482,  0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0  , 0.111],
                                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
                                ]
        elif data_pattern == 2:
            partition_sizes = [
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428,  0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0],
                                [0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0],
                                [0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                                [0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                                [0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428],
                                ]
        elif data_pattern == 4:
            partition_sizes = [[0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                                [0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                                [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125],
                                ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    # test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition

def get_data_feature(worker):
    try:
        # received_para = get_data_socket(worker.socket)
        received_para = worker.client.recv()
        worker.config.neighbor_paras = received_para  # 确保赋值
    except Exception as e:
        print(f"❌ Failed to receive data for {worker.user_name}: {str(e)}")
        worker.config.neighbor_paras = None  # 显式赋值 None

#ada1, 每个worker都有一个model2,最后再聚合（平均）TODO: 按照论文修改聚合方式，不是简单的平均
def train2(model, device, worker_list, epoch_lr, local_steps, total_resource, total_bandwith, total_time):
    # model1,model2 = models.create_model_instance()

    batch_size_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    train_loss = 0.0
    samples_num = 0
    loss_func = nn.CrossEntropyLoss() 
    for iter_idx in range(local_steps):
        communication_parallel(worker_list, action="send_batch_size")  # 发送batch_size给workers
        origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()      # 保存原来的参数
        communication_parallel(worker_list, action="get_data_feature")
        paras = []
        for worker in worker_list:
            if worker.config.neighbor_paras is None:
                print(f"⚠️ Worker {worker.idx} 接收数据为空，跳过处理")
                continue
                
            # 检查2: 数据类型错误的情况
            if not isinstance(worker.config.neighbor_paras, tuple):
                print(f"⚠️ Worker {worker.idx} 数据类型错误，期望tuple，实际收到{type(worker.config.neighbor_paras)}")
                continue
                
            # 检查3: 数据长度不符的情况
            if len(worker.config.neighbor_paras) != 2:
                print(f"⚠️ Worker {worker.idx} 数据长度不符，期望2个元素，实际收到{len(worker.config.neighbor_paras)}")
                continue
                
            model1, model2 = models.create_model_instance("a", "b")
            model2.to(device)
            torch.nn.utils.vector_to_parameters(origin_para, model2.parameters())
            optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            
            try:
                data_feature, target = worker.config.neighbor_paras[0].to(device), worker.config.neighbor_paras[1].to(device)
                if data_feature.dim() == 0:
                    print(f"Worker {worker.idx} 特征维度错误，跳过")
                    continue
                # print("模型参数验证：", data_feature.sum().item())

                input = data_feature.detach().requires_grad_()
                optimizer.zero_grad()
                output1 = model2(input)
                loss = loss_func(output1, target)
                loss.backward()
                train_loss += loss.item()
                
                grad_in = input.grad
                optimizer.step()
                worker.client.send(grad_in.cpu())  # 发送前转到CPU
                paras.append(copy.deepcopy(model2))
            except Exception as e:
                print(f"处理 Worker {worker.idx} 数据时出错: {str(e)}")
                continue

        if not paras:  # 如果没有成功处理任何worker的数据
            continue
        communication_parallel(worker_list, action="get_time")  # 接受训练时间
        
        batch_size_list, sum_time = update_B(worker_list, batch_size_list, compre_ratio_list)
        # total_time += sum_time
        total_resource += Sum(computation_resource, batch_size_list)
        total_bandwith += Sum(bandwith_resource, batch_size_list)

        vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        vector = vector * 0.0
        for para in paras:
            new_para = torch.nn.utils.parameters_to_vector(para.parameters()).detach()
            vector += new_para 
        vector /= len(paras)
        torch.nn.utils.vector_to_parameters(vector, model.parameters())

    print("forward and back propagation")
    if samples_num != 0:

        train_loss /= samples_num
    train_loss /= local_steps
    print(f"一轮的训练损失(train loss): {train_loss}")
    
    return total_resource, total_bandwith, total_time


#ada2：在一轮中worker共享一个model2,model2一直伴随着不同的worker在更新 TODO: 如果模型是non-iid,model2更新会不会东一下西一下的
# def train2(model, device, worker_list, epoch_lr, local_steps, total_resource, total_bandwith, total_time):
#     batch_size_list = [50] * 10
#     compre_ratio_list = [1] * 10
#     computation_resource = [3, 1, 6, 7, 7, 5, 5, 2, 6, 2]
#     bandwith_resource = [5, 6, 8, 1, 5, 8, 2, 4, 4, 2]

#     loss_func = nn.CrossEntropyLoss()
#     train_loss = 0.0
#     samples_num = 0

#     for iter_idx in range(local_steps):
#         communication_parallel(worker_list, action="send_batch_size")  # 通知 batch size
#         origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()  # 保存模型参数
#         communication_parallel(worker_list, action="get_data_feature")  # 获取 features

#         grad_dict = {}

#         model1, model2 = models.create_model_instance("a", "b")
#         model2.to(device)
#         torch.nn.utils.vector_to_parameters(origin_para, model2.parameters())
#         optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)

#         for worker in worker_list:
#             # 跳过无效 worker
#             if worker.config.neighbor_paras is None:
#                 print(f"⚠️ Worker {worker.idx} 接收数据为空，跳过处理")
#                 continue
#             if not isinstance(worker.config.neighbor_paras, tuple):
#                 print(f"⚠️ Worker {worker.idx} 数据类型错误，期望 tuple，实际收到 {type(worker.config.neighbor_paras)}")
#                 continue
#             if len(worker.config.neighbor_paras) != 2:
#                 print(f"⚠️ Worker {worker.idx} 数据长度不符，期望2个元素，实际收到 {len(worker.config.neighbor_paras)}")
#                 continue

#             data_feature, target = worker.config.neighbor_paras
#             data_feature = data_feature.detach().requires_grad_().to(device)
#             target = target.to(device)

#             optimizer.zero_grad()
#             output = model2(data_feature)
#             loss = loss_func(output, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             samples_num += target.size(0)

#             grad_dict[worker.idx] = data_feature.grad.cpu()

#         # 发送梯度回客户端
#         for worker in worker_list:
#             if worker.idx in grad_dict:
#                 worker.client.send(grad_dict[worker.idx])

#         communication_parallel(worker_list, action="get_time")  # 获取时间
#         batch_size_list, sum_time = update_B(worker_list, batch_size_list, compre_ratio_list)
#         total_resource += Sum(computation_resource, batch_size_list)
#         total_bandwith += Sum(bandwith_resource, batch_size_list)

#         # 参数更新
#         new_para = torch.nn.utils.parameters_to_vector(model2.parameters()).detach()
#         torch.nn.utils.vector_to_parameters(new_para, model.parameters())

#     print("forward and back propagation")
#     if samples_num != 0:
#         train_loss /= samples_num
#     train_loss /= local_steps
#     print(f"一轮的训练损失(train loss): {train_loss:.4f}")

#     return total_resource, total_bandwith, total_time


#merge
# def train2(model, device, worker_list, epoch_lr, local_steps, total_resource, total_bandwith, total_time):
#     # model1,model2 = models.create_model_instance()

#     batch_size_list=[50,50,50,50,50,50,50,50,50,50]
#     compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
#     computation_resource=[3,1,6,7,7,5,5,2,6,2]
#     bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
#     train_loss = 0.0
#     samples_num = 0
#     loss_func = nn.CrossEntropyLoss() 
#     for iter_idx in range(local_steps):
#         communication_parallel(worker_list, action="send_batch_size")  # 发送batch_size给workers
#         origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()      # 保存原来的参数
#         communication_parallel(worker_list, action="get_data_feature")
#         paras = []
#         all_features = []
#         all_targets = []
#         sum_batch = 0
#         for worker in worker_list:
#             if worker.config.neighbor_paras is None:
#                 print(f"⚠️ Worker {worker.idx} 接收数据为空，跳过处理")
#                 continue
                
#             # 检查2: 数据类型错误的情况
#             if not isinstance(worker.config.neighbor_paras, tuple):
#                 print(f"⚠️ Worker {worker.idx} 数据类型错误，期望tuple，实际收到{type(worker.config.neighbor_paras)}")
#                 continue
                
#             # 检查3: 数据长度不符的情况
#             if len(worker.config.neighbor_paras) != 2:
#                 print(f"⚠️ Worker {worker.idx} 数据长度不符，期望2个元素，实际收到{len(worker.config.neighbor_paras)}")
#                 continue
#             data_feature, target = worker.config.neighbor_paras
#             all_features.append(data_feature)
#             all_targets.append(target)
#             sum_batch += worker.config.batch_size
            
#         model1, model2 = models.create_model_instance("a", "b")
#         model2.to(device)
#         torch.nn.utils.vector_to_parameters(origin_para, model2.parameters())
#         optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        
#         fused_features = torch.cat(all_features, dim=0).to(device)
#         fused_targets = torch.cat(all_targets, dim=0).to(device)
        
#         input = fused_features.detach().requires_grad_()
#         optimizer.zero_grad()
#         output1 = model2(input)
#         loss = loss_func(output1, fused_targets)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
            
#         grad_dict = {}
#         with torch.no_grad():
#             fused_grad = input.grad 
#             ptr = 0  # 梯度分割指针
#             # tmp = ""
#             for i, worker in enumerate(worker_list):
#                 original_size = all_features[i].size(0)
#                 # tmp += str(original_size) + " "
#                 worker_grad = fused_grad[ptr:ptr + original_size]
#                 grad_dict[worker.idx] = worker_grad.cpu()  # 保存到字典
#                 ptr += original_size
#         # print(tmp)
#         for worker in worker_list:
#             if worker.idx in grad_dict:
#                 worker.client.send(grad_dict[worker.idx])
#         # if not paras:  # 如果没有成功处理任何worker的数据
#         #     continue
#         communication_parallel(worker_list, action="get_time")  # 接受训练时间
        
#         batch_size_list, sum_time = update_B(worker_list, batch_size_list, compre_ratio_list)
#         total_resource += Sum(computation_resource, batch_size_list)
#         total_bandwith += Sum(bandwith_resource, batch_size_list)

#         new_para = torch.nn.utils.parameters_to_vector(model2.parameters()).detach()
#         torch.nn.utils.vector_to_parameters(new_para, model.parameters())


#     print("forward and back propagation")
#     if samples_num != 0:

#         train_loss /= samples_num
#     train_loss /= local_steps
#     print(f"一轮的训练损失(train loss): {train_loss}")
    
#     return total_resource, total_bandwith, total_time


# merge2: 切分成小batch
# def train2(model, device, worker_list, epoch_lr, local_steps, total_resource, total_bandwith, total_time):
#     batch_size_list = [50] * 10
#     compre_ratio_list = [1] * 10
#     computation_resource = [3,1,6,7,7,5,5,2,6,2]
#     bandwith_resource = [5,6,8,1,5,8,2,4,4,2]
#     train_loss = 0.0
#     loss_func = nn.CrossEntropyLoss()

#     for iter_idx in range(local_steps):
#         communication_parallel(worker_list, action="send_batch_size")
#         origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
#         communication_parallel(worker_list, action="get_data_feature")

#         all_features = []
#         all_targets = []
#         sum_batch = 0

#         for worker in worker_list:
#             if worker.config.neighbor_paras is None:
#                 print(f"⚠️ Worker {worker.idx} 接收数据为空，跳过处理")
#                 continue
#             if not isinstance(worker.config.neighbor_paras, tuple):
#                 print(f"⚠️ Worker {worker.idx} 数据类型错误，期望tuple，实际收到{type(worker.config.neighbor_paras)}")
#                 continue
#             if len(worker.config.neighbor_paras) != 2:
#                 print(f"⚠️ Worker {worker.idx} 数据长度不符，期望2个元素，实际收到{len(worker.config.neighbor_paras)}")
#                 continue

#             data_feature, target = worker.config.neighbor_paras
#             all_features.append(data_feature)
#             all_targets.append(target)
#             sum_batch += worker.config.batch_size

#         if not all_features:
#             print("⚠️ 所有 worker 数据为空，本轮跳过")
#             continue

#         model1, model2 = models.create_model_instance("a", "b")
#         model2.to(device)
#         torch.nn.utils.vector_to_parameters(origin_para, model2.parameters())
#         optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)

#         fused_features = torch.cat(all_features, dim=0).to(device)
#         fused_targets = torch.cat(all_targets, dim=0).to(device)

#         input = fused_features.detach().requires_grad_()
#         input_chunks = torch.chunk(input, 4, dim=0)
#         target_chunks = torch.chunk(fused_targets, 4, dim=0)

#         optimizer.zero_grad()
#         loss_total = 0.0
#         grads_list = []

#         for x_chunk, y_chunk in zip(input_chunks, target_chunks):
#             x_chunk = x_chunk.detach().requires_grad_()
#             output = model2(x_chunk)
#             loss = loss_func(output, y_chunk)
#             loss.backward()
#             loss_total += loss.item()
#             grads_list.append(x_chunk.grad.detach())

#         optimizer.step()
#         train_loss += loss_total / len(input_chunks)

#         with torch.no_grad():
#             fused_grad = torch.cat(grads_list, dim=0)
#             grad_dict = {}
#             ptr = 0
#             for i, worker in enumerate(worker_list):
#                 original_size = all_features[i].size(0)
#                 worker_grad = fused_grad[ptr:ptr + original_size]
#                 grad_dict[worker.idx] = worker_grad.cpu()
#                 ptr += original_size

#         for worker in worker_list:
#             if worker.idx in grad_dict:
#                 worker.client.send(grad_dict[worker.idx])

#         communication_parallel(worker_list, action="get_time")
#         batch_size_list, sum_time = update_B(worker_list, batch_size_list, compre_ratio_list)
#         total_resource += Sum(computation_resource, batch_size_list)
#         total_bandwith += Sum(bandwith_resource, batch_size_list)

#         new_para = torch.nn.utils.parameters_to_vector(model2.parameters()).detach()
#         torch.nn.utils.vector_to_parameters(new_para, model.parameters())

#     train_loss /= local_steps
#     print(f"一轮的训练损失(train loss): {train_loss}")
#     return total_resource, total_bandwith, total_time

# merge3: 顺序打乱
# def train2(model, device, worker_list, epoch_lr, local_steps, total_resource, total_bandwith, total_time):
#     batch_size_list = [50] * 10
#     compre_ratio_list = [1] * 10
#     computation_resource = [3,1,6,7,7,5,5,2,6,2]
#     bandwith_resource = [5,6,8,1,5,8,2,4,4,2]
#     train_loss = 0.0
#     loss_func = nn.CrossEntropyLoss()

#     for iter_idx in range(local_steps):
#         communication_parallel(worker_list, action="send_batch_size")
#         origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
#         communication_parallel(worker_list, action="get_data_feature")

#         all_features = []
#         all_targets = []
#         sum_batch = 0
#         worker_info = []

#         # 收集所有worker的数据并记录它们的原始索引
#         for worker in worker_list:
#             if worker.config.neighbor_paras is None:
#                 print(f"⚠️ Worker {worker.idx} 接收数据为空，跳过处理")
#                 continue
#             if not isinstance(worker.config.neighbor_paras, tuple):
#                 print(f"⚠️ Worker {worker.idx} 数据类型错误，期望tuple，实际收到{type(worker.config.neighbor_paras)}")
#                 continue
#             if len(worker.config.neighbor_paras) != 2:
#                 print(f"⚠️ Worker {worker.idx} 数据长度不符，期望2个元素，实际收到{len(worker.config.neighbor_paras)}")
#                 continue

#             data_feature, target = worker.config.neighbor_paras
#             all_features.append(data_feature)
#             all_targets.append(target)
#             worker_info.append((worker.idx, len(data_feature)))  # 保存每个worker的数据大小
#             sum_batch += worker.config.batch_size

#         if not all_features:
#             print("⚠️ 所有 worker 数据为空，本轮跳过")
#             continue

#         model1, model2 = models.create_model_instance("a", "b")
#         model2.to(device)
#         torch.nn.utils.vector_to_parameters(origin_para, model2.parameters())
#         optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)

#         # 打乱数据：首先拼接所有特征和目标
#         fused_features = torch.cat(all_features, dim=0).to(device)
#         fused_targets = torch.cat(all_targets, dim=0).to(device)

#         # 随机打乱数据
#         perm = torch.randperm(fused_features.size(0))
#         fused_features = fused_features[perm]
#         fused_targets = fused_targets[perm]

#         # 对每个worker的数据进行分块：需要记录每个worker的数据在打乱后的位置
#         input = fused_features.detach().requires_grad_()
#         input_chunks = torch.chunk(input, 4, dim=0)
#         target_chunks = torch.chunk(fused_targets, 4, dim=0)

#         optimizer.zero_grad()
#         loss_total = 0.0
#         grads_list = []

#         for x_chunk, y_chunk in zip(input_chunks, target_chunks):
#             x_chunk = x_chunk.detach().requires_grad_()
#             output = model2(x_chunk)
#             loss = loss_func(output, y_chunk)
#             loss.backward()
#             loss_total += loss.item()
#             grads_list.append(x_chunk.grad.detach())

#         optimizer.step()
#         train_loss += loss_total / len(input_chunks)

#         with torch.no_grad():
#             fused_grad = torch.cat(grads_list, dim=0)

#             # 恢复梯度的原始顺序
#             grad_unshuffled = torch.zeros_like(fused_grad)  # 创建空 tensor 来还原梯度顺序
#             grad_unshuffled[perm] = fused_grad  # 恢复打乱前的梯度顺序

#             grad_dict = {}
#             ptr = 0

#             # 将每个worker的梯度根据原始数据顺序重新分配
#             for idx, size in worker_info:
#                 worker_grad = grad_unshuffled[ptr:ptr + size]
#                 grad_dict[idx] = worker_grad.cpu()
#                 ptr += size

#         # 发送回每个worker的梯度
#         for worker in worker_list:
#             if worker.idx in grad_dict:
#                 worker.client.send(grad_dict[worker.idx])

#         communication_parallel(worker_list, action="get_time")
#         batch_size_list, sum_time = update_B(worker_list, batch_size_list, compre_ratio_list)
#         total_resource += Sum(computation_resource, batch_size_list)
#         total_bandwith += Sum(bandwith_resource, batch_size_list)

#         # 更新全局模型参数
#         new_para = torch.nn.utils.parameters_to_vector(model2.parameters()).detach()
#         torch.nn.utils.vector_to_parameters(new_para, model.parameters())

#     train_loss /= local_steps
#     print(f"一轮的训练损失sssssss(train loss): {train_loss}")
#     return total_resource, total_bandwith, total_time


def aggregate_model_para2(client_model, worker_list, device):
    global_para = torch.nn.utils.parameters_to_vector(client_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para).to(device)
        valid_workers = 0
        for worker in worker_list:
            if worker.config.neighbor_paras is not None and isinstance(worker.config.neighbor_paras, torch.Tensor):
                try:
                    worker_para = worker.config.neighbor_paras.to(device)
                    para_delta += worker_para
                    valid_workers += 1
                except Exception as e:
                    print(f"处理 Worker {worker.idx} 参数时出错: {str(e)}")
                    continue
        
        if valid_workers > 0:
            para_delta = para_delta / valid_workers
            
    torch.nn.utils.vector_to_parameters(para_delta, client_model.parameters())
    return para_delta


if __name__ == "__main__":

    # global_model = models.create_model_instance('CIFAR10', 'AlexNet')
    # print(global_model.state_dict())
    main()
