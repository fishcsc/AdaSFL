import os
import sys
import struct
import socket
import pickle
from time import sleep
import time
import queue    
import threading

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
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

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
    s.listen(5)

    conn, addr = s.accept()
    # conn.settimeout(120)

    return conn, addr

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

class ConnectionHandler:
    def __init__(self, conn, is_worker=False):
        self.conn = conn
        self.message_queue = queue.Queue()
        self.last_heartbeat = time.time()
        self.running = True
        self.heartbeat_interval = 5  # 心跳间隔5秒
        self.timeout_threshold = 10  # 超时时间10秒
        
        # 启动接收线程和心跳线程
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.receive_thread.daemon = True
        self.heartbeat_thread.daemon = True
        self.receive_thread.start()
        if is_worker:
            self.heartbeat_thread.start()

    def _receive_loop(self):
        """接收消息的循环"""
        while self.running:
            try:
                data = self._raw_recv()
                if data:
                    # 处理心跳包
                    if data.get("message") == "heartbeat":
                        self.last_heartbeat = time.time()
                    else:
                        self.message_queue.put(data["message"])
                else:
                    time.sleep(0.1)  
            except socket.timeout:
            # 超时不断开，继续尝试
                continue
            except Exception as e:
                print(f"\n❌ 接收消息错误: {str(e)}")
                break
        self.close()

    def _heartbeat_loop(self):
        """心跳发送循环"""
        while self.running and self.conn.fileno() != -1:
            try:
                # 发送心跳包
                if time.time() - self.last_heartbeat > self.heartbeat_interval:
                    # print("fileno:" ,self.conn.fileno())
                    self.send("heartbeat")
                    self.last_heartbeat = time.time()
                    
                # 检查超时
                if time.time() - self.last_heartbeat > self.timeout_threshold:
                    print("❌ 心跳超时，连接断开")
                    self.close()
                    break
                    
                time.sleep(1)
            except Exception as e:
                print(f"心跳错误: {str(e)}")
                self.close()

    def _raw_recv(self):
        """原始接收方法"""
        try:
            data_length = int.from_bytes(self.conn.recv(4), byteorder='big')
            data = b""
            while len(data) < data_length:
                packet = self.conn.recv(data_length - len(data))
                if not packet:
                    return None
                data += packet
            return pickle.loads(data)
        except Exception as e:
            return None

    # def recv(self):
    #     """从队列获取消息（非阻塞）"""
    #     try:
    #         return self.message_queue.get_nowait()
    #     except queue.Empty:
    #         return None

    def recv(self, timeout=20):
        """从队列获取消息（阻塞模式，可选超时）"""
        try:
            return self.message_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None  # 

    def send(self, message, max_retries=3):
        """改进后的发送方法（含重试机制）"""
        retries = 0
        while retries < max_retries:
            if self.conn.fileno() == -1:
                print("⚠️ 连接已关闭，无法发送消息")
                return
            try:
                data = pickle.dumps({"message": message})
                self.conn.sendall(len(data).to_bytes(4, byteorder='big'))
                self.conn.sendall(data)
                return  # 发送成功立即退出
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"🚨 连接异常: {str(e)}，正在重试 ({retries+1}/{max_retries})")
                retries += 1
                time.sleep(1)
            except OSError as e:
                if e.errno == 9:
                    print("🔌 连接已主动关闭")
                    return
                else:
                    raise
        print(f"❌ 发送失败，已达最大重试次数 {max_retries}次")
        
    def close(self):
        """关闭连接"""
        self.running = False  # 停止线程循环
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()