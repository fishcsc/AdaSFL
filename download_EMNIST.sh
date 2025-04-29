#!/bin/bash

# 设置数据保存路径
DATA_DIR="/data0/scchen/data/EMNIST/raw"

# 1. 创建目录
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

# 定义下载函数
download_emnist() {
    local url=$1
    local output=$2
    echo "🚀 正在尝试下载 EMNIST 数据集：$url"
    wget --no-check-certificate "$url" -O "$output"
}

# 2. 下载 EMNIST gzip.zip (尝试NIST官网)
download_emnist "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip" "gzip.zip"

# 3. 检查文件大小
FILE_SIZE=$(stat -c%s "gzip.zip")
if [ "$FILE_SIZE" -lt 500000 ]; then
    echo "⚠️ NIST 官网下载失败或文件异常 (只有 $(($FILE_SIZE/1024)) KB)，尝试从 Kaggle 镜像下载..."

    # 删掉错误的小文件
    rm -f gzip.zip

    # 重新从 Kaggle 镜像下载
    download_emnist "https://www.kaggleusercontent.com/datasets/crawford/emnist/download?datasetVersionNumber=1" "gzip.zip"

    # 再次检查
    FILE_SIZE=$(stat -c%s "gzip.zip")
    if [ "$FILE_SIZE" -lt 500000 ]; then
        echo "❌ Kaggle 镜像下载也失败了，请手动下载 gzip.zip！"
        exit 1
    fi
fi

# 4. 解压
echo "📦 正在解压..."
unzip -o gzip.zip

# 5. 检查是否解压成功
if ls *.gz 1> /dev/null 2>&1; then
    echo "✅ EMNIST 数据集解压成功！位于 $DATA_DIR"
else
    echo "❌ 解压失败，请检查 gzip.zip 是否完整。"
fi
