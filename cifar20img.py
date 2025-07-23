import os
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

# ---------- 配置参数 ----------
SAVE_DIR = "./cifar100_coarse_upscaled"  # 保存路径
UPSCALE_FACTOR = 4                      # 放大倍数 (32x32 → 128x128)
RESAMPLE_METHOD = Image.BICUBIC         # 使用双三次插值

# ---------- CIFAR-100 粗粒度映射 ----------
CIFAR100_COARSE_MAPPING = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

coarse_class_names = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium-sized_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

# 加载数据集
testset = datasets.CIFAR100(root="./data", train=False, download=True)
testset.targets = [CIFAR100_COARSE_MAPPING[label] for label in testset.targets]

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)
for class_name in coarse_class_names:
    os.makedirs(os.path.join(SAVE_DIR, class_name), exist_ok=True)

# 保存放大后的图片
for idx in tqdm(range(len(testset)), desc="Upscaling Images"):
    img, coarse_label = testset[idx]
    class_name = coarse_class_names[coarse_label]
    
    # 放大图片
    original_size = img.size  # CIFAR图片是32x32
    upscaled_img = img.resize(
        (original_size[0] * UPSCALE_FACTOR, 
         original_size[1] * UPSCALE_FACTOR),
        resample=RESAMPLE_METHOD
    )
    
    # 保存文件
    filename = f"upscaled_{class_name}_{idx}.png"
    upscaled_img.save(os.path.join(SAVE_DIR, class_name, filename))

print(f"所有图片已放大{UPSCALE_FACTOR}倍并保存到 {SAVE_DIR}")