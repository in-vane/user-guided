import os
import numpy as np
from PIL import Image
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_cifar_to_images(batch_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载批次数据
    batch = unpickle(batch_path)
    
    # 获取图片数据和标签
    images = batch[b'data']
    labels = batch[b'labels']
    
    # 定义标签名称
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 转换每张图片
    for i in range(len(images)):
        # 重塑图片格式 (3072 -> 32x32x3)
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        pil_img = Image.fromarray(img)
        
        # 创建标签子目录
        label_dir = os.path.join(output_dir, label_names[labels[i]])
        os.makedirs(label_dir, exist_ok=True)
        
        # 保存图片
        img_path = os.path.join(label_dir, f'image_{i}.png')
        pil_img.save(img_path)

# 使用示例
convert_cifar_to_images('data/cifar10/data_batch_1', 'cifar10_images_batch1')