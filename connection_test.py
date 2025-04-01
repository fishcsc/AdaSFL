import os
import sys
import socket
import json
import pickle
import time
import threading

def send_data_socket(data, socket_connection):
    # 将数据序列化为二进制
    data = pickle.dumps(data)
    # 先发送数据长度（4字节）
    socket_connection.sendall(len(data).to_bytes(4, byteorder='big'))
    # 再发送实际数据
    socket_connection.sendall(data)

def get_data_socket(socket_connection):
    try:
        # 接收数据长度
        data_length = int.from_bytes(socket_connection.recv(4), byteorder='big')
        # 接收数据
        data = b''
        while len(data) < data_length:
            packet = socket_connection.recv(data_length - len(data))
            if not packet:
                return None
            data += packet
        return pickle.loads(data)
    except Exception as e:
        error_type = str(type(e).__name__)
        error_message = str(e).lower()
        
        # 检查是否为连接错误
        if error_type == "ConnectionResetError" or error_type == "BrokenPipeError" or "connection reset" in error_message or "broken pipe" in error_message:
            print("❌ 发送数据时连接已断开")
        if not "timed out" in error_message:
            print(f"❌ 数据接收失败: {str(e)}")
        return None

def receive_messages(socket_connection):
    """持续接收消息的线程函数"""
    socket_connection.settimeout(10)  # 设置更长的超时时间
    last_heartbeat = time.time()
    
    while True:
        try:
            # 发送心跳包
            if time.time() - last_heartbeat > 5:  # 每10秒发送一次心跳
                send_data_socket({"message": "heartbeat"}, socket_connection)
                last_heartbeat = time.time()
                
            data = get_data_socket(socket_connection)
            if data:
                if data.get("message") == "heartbeat":
                    last_heartbeat = time.time()
                    print("心跳响应")
                    continue
                    
                print(f"\n收到消息: {data['message']}")
                print("请输入要发送的消息: ", end='', flush=True)
            else:
                # print("\n❌ 连接已断开")
                # break
                time.sleep(0.1)
        except socket.timeout:
            # 超时不断开，继续尝试
            continue
        except Exception as e:
            print(f"\n❌ 接收消息错误: {str(e)}")
            break
    socket_connection.close()

class TestServer:
    def __init__(self, port=57008):
        self.port = port
        self.socket = None
        self.running = True
        
    def start(self):
        # 先尝试杀死占用端口的进程
        kill_port_process(self.port)
        
        # 创建服务器socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 启用 TCP keepalive
        
        # TCP keepalive 参数设置
        if hasattr(socket, 'TCP_KEEPIDLE'):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        if hasattr(socket, 'TCP_KEEPINTVL'):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        if hasattr(socket, 'TCP_KEEPCNT'):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        
        # 添加重试机制
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.socket.bind(('127.0.0.1', self.port))
                break
            except OSError as e:
                print(f"绑定端口失败，尝试重试 {retry_count + 1}/{max_retries}")
                retry_count += 1
                kill_port_process(self.port)
                time.sleep(1)
                
        if retry_count == max_retries:
            print("无法绑定端口，请检查端口是否被占用")
            return
            
        self.socket.listen(5)
        print(f"服务器启动，监听端口 {self.port}")
        
        # 等待客户端连接
        client_socket, address = self.socket.accept()
        print(f"接受到来自 {address} 的连接")
        
        # 创建接收消息的线程
        receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
        receive_thread.daemon = True
        receive_thread.start()
        
        # 主线程负责发送消息
        try:
            while self.running:
                message = input("请输入要发送的消息: ")
                if message.lower() == 'quit':
                    break
                send_data_socket({"message": message}, client_socket)
        except Exception as e:
            print(f"发送消息错误: {str(e)}")
        finally:
            self.running = False
            client_socket.close()
            self.socket.close()

class TestWorker:
    def __init__(self, server_port=57008):
        self.server_port = server_port
        self.running = True
        
    def connect(self):
        # 添加重试机制
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                socket_connection = socket.create_connection(
                    ('127.0.0.1', self.server_port), timeout=30  # 增加超时时间
                )
                # 设置 TCP keepalive
                socket_connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    socket_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    socket_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    socket_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
                
                print(f"成功连接到服务器 127.0.0.1:{self.server_port}")
                
                # 创建接收消息的线程
                receive_thread = threading.Thread(target=receive_messages, args=(socket_connection,))
                receive_thread.daemon = True
                receive_thread.start()
                
                # 主线程负责发送消息
                try:
                    while self.running:
                        message = input("请输入要发送的消息: ")
                        if message.lower() == 'quit':
                            break
                        send_data_socket({"message": message}, socket_connection)
                except Exception as e:
                    print(f"发送消息错误: {str(e)}")
                finally:
                    self.running = False
                    socket_connection.close()
                break  # 连接成功，跳出重试循环
                    
            except Exception as e:
                print(f"连接错误: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"等待重试 {retry_count}/{max_retries}")
                    time.sleep(2)
                else:
                    print("连接失败，已达到最大重试次数")

def start_server():
    server = TestServer()
    server.start()

def start_worker():
    time.sleep(2)  # 等待服务器启动
    worker = TestWorker()
    worker.connect()

# 添加一个检查和杀死占用端口进程的函数
def kill_port_process(port):
    try:
        # 对于Linux/Unix系统
        os.system(f'lsof -t -i:{port} | xargs kill -9')
        # 对于Windows系统
        os.system(f'netstat -ano | findstr :{port} | xargs taskkill /F /PID')
        time.sleep(1)  # 等待进程完全释放端口
    except:
        pass

def cleanup():
    """清理函数，确保socket正确关闭"""
    try:
        # 对于服务器
        if hasattr(server, 'socket') and server.socket:
            server.socket.close()
        # 对于客户端
        if hasattr(worker, 'socket_connection') and worker.socket_connection:
            worker.socket_connection.close()
    except:
        pass

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("请指定运行模式: server 或 worker")
            sys.exit(1)
            
        mode = sys.argv[1]
        if mode == "server":
            start_server()
        elif mode == "worker":
            start_worker()
        else:
            print("无效的运行模式，请使用 server 或 worker")
    finally:
        cleanup() 