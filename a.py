import os
import sys
import socket
import pickle
import threading
import queue
import time

# class ConnectionHandler:
#     def __init__(self, conn):
#         self.conn = conn
#         self.message_queue = queue.Queue()
#         self.last_heartbeat = time.time()
#         self.running = True
#         self.heartbeat_interval = 5  # 心跳间隔5秒
#         self.timeout_threshold = 15  # 超时时间15秒
        
#         # 启动接收线程和心跳线程
#         self.receive_thread = threading.Thread(target=self._receive_loop)
#         self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
#         self.receive_thread.daemon = True
#         self.heartbeat_thread.daemon = True
#         self.receive_thread.start()
#         self.heartbeat_thread.start()

#     def _receive_loop(self):
#         """接收消息的循环"""
#         while self.running:
#             try:
#                 data = self._raw_recv()
#                 if not data:
#                     continue
                    
#                 # 处理心跳包
#                 if data.get("message") == "heartbeat":
#                     self.last_heartbeat = time.time()
#                 else:
#                     self.message_queue.put(data["message"])
                    
#             except Exception as e:
#                 print(f"接收错误: {str(e)}")
#                 self.close()

#     def _heartbeat_loop(self):
#         """心跳发送循环"""
#         while self.running:
#             try:
#                 # 发送心跳包
#                 if time.time() - self.last_heartbeat > self.heartbeat_interval:
#                     self.send("heartbeat")
#                     self.last_heartbeat = time.time()
                    
#                 # 检查超时
#                 if time.time() - self.last_heartbeat > self.timeout_threshold:
#                     print("❌ 心跳超时，连接断开")
#                     self.close()
#                     break
                    
#                 time.sleep(1)
#             except Exception as e:
#                 print(f"心跳错误: {str(e)}")
#                 self.close()

#     def _raw_recv(self):
#         """原始接收方法"""
#         try:
#             data_length = int.from_bytes(self.conn.recv(4), byteorder='big')
#             data = b""
#             while len(data) < data_length:
#                 packet = self.conn.recv(data_length - len(data))
#                 if not packet:
#                     return None
#                 data += packet
#             return pickle.loads(data)
#         except Exception as e:
#             return None

#     def recv(self):
#         """从队列获取消息（非阻塞）"""
#         try:
#             return self.message_queue.get_nowait()
#         except queue.Empty:
#             return None


#     def send(self, message):
#         """发送消息"""
#         data = pickle.dumps({"message": message})
#         self.conn.sendall(len(data).to_bytes(4, byteorder='big'))
#         self.conn.sendall(data)
#     def close(self):
#         """关闭连接"""
#         self.conn.close()

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
                if not data:
                    continue
                    
                # 处理心跳包
                if data.get("message") == "heartbeat":
                    self.last_heartbeat = time.time()
                else:
                    self.message_queue.put(data["message"])
                    
            except Exception as e:
                print(f"接收错误: {str(e)}")
                self.close()

    def _heartbeat_loop(self):
        """心跳发送循环"""
        while self.running and self.conn.fileno() != -1:
            try:
                # 发送心跳包
                if time.time() - self.last_heartbeat > self.heartbeat_interval:
                    print("fileno:" ,self.conn.fileno())
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

    def recv(self):
        """从队列获取消息（非阻塞）"""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None


    def send(self, message):
        """改进后的发送方法"""
        if self.conn.fileno() == -1:  # 新增文件描述符检查
            print("⚠️ 连接已关闭，无法发送消息")
            return
            
        try:
            data = pickle.dumps({"message": message})
            self.conn.sendall(len(data).to_bytes(4, byteorder='big'))
            self.conn.sendall(data)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"🚨 连接异常: {str(e)}")
            self.close()
        except OSError as e:
            if e.errno == 9:  # 处理Bad file descriptor
                print("🔌 连接已主动关闭")
            else:
                raise
    def close(self):
        """关闭连接"""
        self.running = False  # 停止线程循环
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()

class TestServer:
    def __init__(self, port=57001):
        self.port = port
        self.socket = None
        self.client = None

    def start(self):
        """启动服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", self.port))
        self.socket.listen(1)

        print(f"🚀 服务器已启动，监听端口 {self.port}...")
        conn, addr = self.socket.accept()
        print(f"✅ 客户端 {addr} 已连接")

        self.client = ConnectionHandler(conn)
        self.console_loop()

    def console_loop(self):
        """服务器交互命令行"""
        while True:
            cmd = input("Server> ").strip()
            if cmd.lower() == "quit":
                break
            elif cmd.startswith("send "):
                self.client.send(cmd[5:])
            elif cmd == "recv":
                print(f"📩 收到消息: {self.client.recv()}")
            else:
                print("❓ 命令无效，使用：send <message> 或 recv")
                

        self.client.close()
        self.socket.close()
        print("🚪 服务器已关闭")


class TestWorker:
    def __init__(self, server_port=57001):
        self.server_port = server_port
        self.conn = None

    def connect(self):
        """连接服务器"""
        self.conn = socket.create_connection(("127.0.0.1", self.server_port))
        print(f"✅ 已连接到服务器 127.0.0.1:{self.server_port}")

        self.client = ConnectionHandler(self.conn)
        self.console_loop()

    def console_loop(self):
        """客户端交互命令行"""
        while True:
            cmd = input("Worker> ").strip()
            if cmd.lower() == "quit":
                break
            elif cmd.startswith("send "):
                self.client.send(cmd[5:])
            elif cmd == "recv":
                print(f"📩 收到消息: {self.client.recv()}")
            else:
                print("❓ 命令无效，使用：send <message> 或 recv")

        self.client.close()
        print("🚪 客户端已断开连接")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请指定运行模式: server 或 worker")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "server":
        TestServer().start()
    elif mode == "worker":
        TestWorker().connect()
    else:
        print("无效的运行模式，请使用 server 或 worker")
