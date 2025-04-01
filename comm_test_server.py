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
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_pattern', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
master_listen_port_base=53710
RESULT_PATH = 'result_record'

def main():
    
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

    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)
    
    worker_num = len(workers_config['worker_config_list'])

    client_model,global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(client_model.parameters())
    global_model.to(device)

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
    
    #create workers
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
    # time.sleep(3)
    train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)

    for worker in worker_list:
        send_init_config_with_greeting(worker)
    
    # while True:
    #     for worker in worker_list:
    #         if not worker.client.running:
    #             continue
            
    #         # 正确接收方式（持续处理队列）
    #         while True:
    #             msg = worker.client.recv()
    #             if msg is None:
    #                 break
    #             print(f"[{time.ctime()}] 收到来自 {worker.idx} 的消息: {msg}")
        
    #     time.sleep(1)  # 降低CPU占用
    local_steps=20
    for epoch_idx in range(1, 1+common_config.epoch):
        for worker in worker_list:
            print("sending local steps to workers:", local_steps)
            worker.client.send(local_steps)
            print("sending local steps to workers done")


    
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
        worker.client = ConnectionHandler(worker.socket, is_worker=True)
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
def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num=10
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    elif dataset_type == "EMNIST":
        train_class_num=62
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    if dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num=100
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
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

    elif dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
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



if __name__ == "__main__":
    main()