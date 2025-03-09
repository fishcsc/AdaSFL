import os
import sys
import struct
import socket
import pickle
from time import sleep
import time

def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)

def connect_send_socket(dst_ip, dst_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.settimeout(120)

    while s.connect_ex((dst_ip, dst_port)) != 0:
        sleep(0.5)

    return s

def connect_get_socket(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)
    
    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print(e)
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            # killport(listen_port)
            print(listen_port)
            if time.time() - start_time > 30:
                sys.exit(0)
    s.listen(1)

    conn, _ = s.accept()
    conn.settimeout(6000)

    return conn

def send_data_socket(data, s):
    """发送数据，确保完整性"""
    data_serialized = pickle.dumps(data)
    data_len = len(data_serialized)
    # 发送数据长度（4字节） + 数据内容
    header = struct.pack('>I', data_len)
    s.sendall(header)
    s.sendall(data_serialized)

# def get_data_socket(conn):
#     data_len = struct.unpack(">I", conn.recv(4))[0]
#     #print(data_len)
#     '''
#     try:
#         data_len = struct.unpack(">I", conn.recv(4))[0]
#     except:
#         return None
#     '''
#     data = conn.recv(data_len, socket.MSG_WAITALL)
#     recv_data = pickle.loads(data)

#     return recv_data

def get_data_socket(s, timeout=120):
    """接收数据，处理超时和截断"""
    s.settimeout(timeout)  # 设置超时时间
    try:
        # 读取数据长度（4字节）
        data_len_bytes = b''
        while len(data_len_bytes) < 4:
            packet = s.recv(4 - len(data_len_bytes))
            if not packet:
                break
            data_len_bytes += packet
        if len(data_len_bytes) != 4:
            raise ValueError("Invalid data length header")
        data_len = struct.unpack('>I', data_len_bytes)[0]

        # 读取数据内容
        data_bytes = b''
        while len(data_bytes) < data_len:
            packet = s.recv(data_len - len(data_bytes))
            if not packet:
                break
            data_bytes += packet
        if len(data_bytes) != data_len:
            raise ValueError(f"Received incomplete data: {len(data_bytes)} of {data_len} bytes")
        
        return pickle.loads(data_bytes)
    except Exception as e:
        print(f"❌ 数据接收失败: {str(e)}")
        return None