import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from pulp import *
import random
from config import ClientConfig, CommonConfig
from client_comm_utils import *
from training_utils import train2, test
import datasets, models
import pdb

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx)) % 2 + 0)
    # if args.idx == '8':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # if args.idx == '9':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
# if int(args.idx) == 0:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        common_config=CommonConfig()
    )
    # recorder = SummaryWriter("log_"+str(args.idx))
    # receive config
    master_socket, addr = connect_get_socket(args.master_ip, args.master_port)
    reuse = 1
    master_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuse)

    print(f"接收到来自 {addr} 的连接")
    client = ConnectionHandler(master_socket, is_worker=True)
    
    try:
        # 接收服务器的打招呼
        greeting = client.recv()
        if greeting and "message" in greeting:
            print(f"收到服务器消息: {greeting['message']}")
            
            # 发送响应
            response = {"message": f"Hello from worker {args.idx}"}
            client.send(response)
            print("已发送响应到服务器")
        
        # 接收实际配置
        print("等待接收配置...")
        config_received = client.recv()
        if config_received:
            print("成功接收配置")
            for k, v in config_received.__dict__.items():
                setattr(client_config, k, v)
    except Exception as e:
        print(f"❌ 通信错误: {str(e)}")
        client.close()
        return
    
    # pdb.set_trace()

    computation = client_config.custom["computation"]
    dynamics=client_config.custom["dynamics"]

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    # common_config.batch_size = 1
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    

    # init config
    print(common_config.__dict__)

    local_model,_ = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    init_para = torch.nn.utils.parameters_to_vector(local_model.parameters())           # 计算参数
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("para num: {}".format(init_para.nelement()))
    print("Model Size: {} MB".format(model_size))

    # create dataset
    print(len(client_config.custom["train_data_idxes"]))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    train_dataset= torch.utils.data.Subset(train_dataset, client_config.custom["train_data_idxes"])
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size)
    train_data, train_label = load_dataset(train_dataset)
    # train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)

    epoch_lr = common_config.lr
    local_steps=30
    start_idx=0
    for epoch in range(1, 1+common_config.epoch):
        local_steps=client.recv()
        print("recieved local_steps from server: ", local_steps)

        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))
        # print("local steps: ", local_steps)
        # print("Compression Ratio: ", compre_ratio)

        #print("***")
        start_time = time.time()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
        train2(local_model, train_data, train_label, optimizer, local_steps, device, client,start_idx,train_loader)
        train_loss = 0.0

        train_time = time.time() - start_time
        train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        while train_time>10 or train_time<1:
             train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        train_time=train_time/10
        print("train time: ", train_time)

        acc,test_loss=0,0
        
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("send para")

        send_model_para(local_model,client)
        
        send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        while send_time>10 or send_time<1:
             send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        
        print("send time: ",send_time)
        client.send((train_time,send_time))
        print("get begin")
        get_model_para(local_model,client)
        print("get end")
    master_socket.shutdown(2)
    client.close()

def send_model_dict(local_model,object):
    model_dict = dict()
    for para in local_model.state_dict().keys():
        model_dict[para] = copy.deepcopy(local_model.state_dict()[para])
    
    start_time = time.time()
    object.send(model_dict)
    send_time=time.time()-start_time
    pass

def get_model_dict(local_model,object):
    local_para = object.recv()
    local_model.load_state_dict(local_para)
    local_model.to(device)

def send_model_para(local_model,object):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    object.send(local_paras)
    

def send_compressed_model(local_model,object,ratio):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()
    compress_paras=compress_model_top(local_paras, ratio)
    object.send(compress_paras)
    # send_data_socket(compress_paras, master_socket)

def send_compressed_gradient(local_model,object,ratio,old_para,memory_para):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    memory_para,compress_paras=compress_gradient_top(local_paras, old_para, memory_para,ratio)
    object.send(compress_paras)
    # send_data_socket(compress_paras, master_socket)
    return memory_para

def get_model_para(local_model, object):
    try:
        local_para = object.recv()
        if not isinstance(local_para, torch.Tensor):
            print(f"收到的数据类型错误: {type(local_para)}")
            return
            
        local_para = local_para.to(device)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
        return local_para
    except Exception as e:
        print(f"获取模型参数时出错: {str(e)}")
        return None

def compress_model_rand(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        select_n = int(send_para.nelement() * ratio)
        rd_seed = np.random.randint(0, np.iinfo(np.uint32).max)
        rng = np.random.RandomState(rd_seed)
        indices = rng.choice(send_para.nelement(), size=select_n, replace=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    return (select_para, select_n, rd_seed)

def compress_model_top(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, indices)

def compress_gradient_top(local_para, old_para, memory_para,ratio):
    start_time = time.time()
    with torch.no_grad():
        # print("local_para:",local_para)
        # print("old_para:",old_para)
        old_para=local_para-old_para+memory_para
        # print("local_para:",local_para)
        send_para = old_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(old_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    restored_model = torch.zeros(send_para.nelement()).to(device)
    restored_model[indices] = select_para
    memory_para=old_para - restored_model
    model_size = select_para.nelement() * 4 / 1024 / 1024
    print("model_size:",model_size)
    # print("memory_para:",memory_para)
    return (memory_para,(select_para, indices))

def load_dataset(dataset):
    num_samples = len(dataset)
    indices = [i for i in range(num_samples)]
    random.shuffle(indices)
    tx2_data, tx2_label = zip(*([dataset[i] for i in range(num_samples)]))
    return torch.cat(tx2_data, 0), torch.tensor(tx2_label)

if __name__ == '__main__':
    main()
