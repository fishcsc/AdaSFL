import torchvision.datasets as datasets

# 本地存储路径，根据你的实际需求修改
data_path = "/data0/scchen/data"

# 下载 EMNIST 的 train 和 test（split='byclass'）
print("Downloading MNIST (byclass)...")
dataset=datasets.FashionMNIST(root=data_path, train=True, download=True)
datasets.FashionMNIST(root=data_path, train=False, download=True)
print("Download complete.") 

import os
import requests
import zipfile

def download_tiny_imagenet(destination=data_path):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    filepath = os.path.join(destination, filename)
    extracted_path = os.path.join(destination, "tiny-imagenet-200")

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not os.path.exists(filepath):
        print("Downloading TinyImageNet...")
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

    if not os.path.exists(extracted_path):
        print("Extracting...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print("Extraction complete.")

    return extracted_path

# 用法
data_path = download_tiny_imagenet()
print("TinyImageNet 数据保存在:", data_path)
