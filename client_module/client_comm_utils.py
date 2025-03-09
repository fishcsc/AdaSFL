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
    # s.settimeout(120)

    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print(e)
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            killport(listen_port)
            if time.time() - start_time > 30:
                sys.exit(0)
    s.listen(1)

    conn, _ = s.accept()
    # conn.settimeout(120)

    return conn

def send_data_socket(data, s):
    try:
        data = pickle.dumps(data)
        s.sendall(struct.pack(">I", len(data)))
        s.sendall(data)
    except Exception as e:
        print(f"发送数据失败: {str(e)}")

def get_data_socket(conn):
    try:
        header = conn.recv(4)
        if not header or len(header) != 4:
            print("❌ 接收头部数据失败")
            return None
            
        data_len = struct.unpack(">I", header)[0]
        data = conn.recv(data_len, socket.MSG_WAITALL)
        
        if not data or len(data) != data_len:
            print("❌ 接收数据不完整")
            return None
            
        return pickle.loads(data)
    except socket.timeout:
        print("❌ 数据接收超时")
        return None
    except Exception as e:
        print(f"❌ 数据接收失败: {str(e)}")
        return None
