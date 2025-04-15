import sys
import time
import math
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from client_comm_utils import *
from models import *



def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    #print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        data, target = next(data_loader)

        if model_type == 'LR':
            data = data.squeeze(1).view(-1, 28 * 28)
        
        # target=target%5

        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        loss_func = nn.CrossEntropyLoss() 
        loss =loss_func(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num
    
    return train_loss

def test(model, data_loader, device=torch.device("cpu"), model_type=None):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)


            # target=target%5

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
            output = model(data)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    # TODO: Record

    return test_loss, test_accuracy

def train2(model, global_model, train_data, train_label, optimizer, optimizer2, local_iters, device, client, start_idx,train_loader, sleep_time):
    model.train()
    global_model.train()
    train_loss = 0.0
    samples_num = len(train_label)
    
    for iter_idx in range(local_iters):
        try:
            t_start = time.time()
            batch_size = client.recv()
            
            # batch_size = get_data_socket(master_socket)
            print("recieved batch_size from server: ", batch_size)
            if not isinstance(batch_size, int):
                print(f"收到的batch_size类型错误: {type(batch_size)}")
                continue
                
            data = torch.reshape(train_data[start_idx:start_idx+batch_size, :, :], [-1, 32, 32]).to(device)
            target = (train_label[start_idx:start_idx+batch_size]).to(device)
            start_idx = start_idx + batch_size
            if start_idx >= samples_num:
                start_idx = 0
            
            data, target = next(train_loader)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            time_1 = time.time()
            data_feature = model(data)
            x_data = data_feature.detach()
            time_forward = time.time() - time_1
            # target = target.cpu()
            # print("模型参数验证：", x_data.sum().item())
            
            print("feature num: ", x_data.numel())
            print("feature size: ", x_data.numel() * 4 / 1024 / 1024)
            print("send feature and target to server")
            
            client.send((x_data, target))
            
            #本地计算lossloss
            input = x_data.detach().requires_grad_()
            optimizer2.zero_grad()
            output = global_model(input)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(output, target)
            train_loss += loss.item()
            loss.backward()  
            grad_in1 = input.grad 
            optimizer2.step()
           
            
                
            grad_in = client.recv() 
            if grad_in is None:
                print("❌ 梯度为空，跳过反向传播")
                continue
            time_2 = time.time()
            grad_in = torch.as_tensor(grad_in, device=device)
            data_feature.backward(grad_in)
            optimizer.step()
            time_backward = time.time() - time_2

            time.sleep(sleep_time)
            compute_time = time_forward + time_backward + sleep_time
           
            print("compute_time: ", compute_time)
            one_step_time = time.time() - t_start
            print("one_step_time: ", one_step_time)
            client.send((compute_time, 0))
            
        except Exception as e:
            print(f"训练过程出错: {str(e)}")
            continue
    
    train_loss /= local_iters
    print("train_loss: ", train_loss)
    return

# test_loss, acc = test2(global_model,client_model, test_loader, device)

