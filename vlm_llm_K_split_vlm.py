# 优化后的单文件 VQA-LLM 图像分类系统（VQA部分）

import os
import json
import torch
import argparse
import logging
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm

# ---------- 日志配置 ----------
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description='VQA-LLM图像分类系统 - VQA部分')
parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'ppmi', 'oxford_pet'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--criteria', type=str, default='default', choices=['flying', 'living', 'situation', 'instrument', 'cifar20', 'default'])
args = parser.parse_args()

# ---------- 数据集类 ----------
class CustomDataset:
    def __init__(self, dataset_name, transform=None, split='test'):
        self.dataset_name = dataset_name
        self.transform = transform
        self.split = split
        self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == 'mnist':
            self.dataset = datasets.MNIST(root="./data", train=(self.split == 'train'), download=True)
            self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10(root="./data", train=(self.split == 'train'), download=True)
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_name == 'cifar100':
            self.dataset = datasets.CIFAR100(root="./data", train=(self.split == 'train'), download=True)
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
            # 将细粒度标签转换为粗粒度标签
            original_targets = self.dataset.targets
            self.dataset.targets = [CIFAR100_COARSE_MAPPING[t] for t in original_targets]
            self.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                            'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                            'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                            'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 
                            'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
        elif self.dataset_name == 'stl10':
            self.dataset = datasets.STL10(root="./data", split=self.split, download=True)
            self.classes = ['airplane', 'bird', 'car', 'cat', 'deer',
                            'dog', 'horse', 'monkey', 'ship', 'truck']
        elif self.dataset_name == 'ppmi':
            full_dataset = ImageFolder(root="./data/ppmi/play_instrument")
            test_samples = [(p, l) for p, l in full_dataset.samples if "/test/" in p.replace("\\", "/")]
            self.dataset = torch.utils.data.Subset(full_dataset, indices=[i for i, (p, _) in enumerate(full_dataset.samples) if "/test/" in p.replace("\\", "/")])
            self.classes = ['bassoon', 'flute', 'frenchhorn', 'guitar', 'saxophone', 'violin']
        elif self.dataset_name == 'oxford_pet':
            self.dataset = datasets.OxfordIIITPet(root="./data", split=self.split, download=True)
            self.classes = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 
                            'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 
                            'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 
                            'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
                            'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 
                            'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(np.transpose(img, (1, 2, 0)))
        return self.transform(pil_img) if self.transform else pil_img, target, pil_img

def custom_collate_fn(batch):
    tensors, labels, pil_imgs = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(pil_imgs)

# ---------- 模型加载 ----------
def load_vqa_model():
    logger.info("加载VQA模型…")
    vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return vqa_model.eval(), processor

# ---------- VQA生成 ----------
def generate_vqa_answers(pil_imgs, processor, vqa_model, question):
    inputs = processor(images=pil_imgs, text=[question]*len(pil_imgs), return_tensors="pt", padding=True).to(vqa_model.device)
    with torch.no_grad():
        vqa_out = vqa_model.generate(**inputs, use_cache=False)
    answers = processor.batch_decode(vqa_out, skip_special_tokens=True)
    return [ans.split("Answer:")[-1].strip() for ans in answers]

# ---------- 主处理流程 ----------
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    vqa_model, processor = load_vqa_model()
    
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"加载 {args.dataset.upper()} 数据集...")
    test_dataset = CustomDataset(dataset_name=args.dataset, transform=transform_tensor, split='test')
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_vqa_intermediate.jsonl")

    # question = f"Question: What is the main object in the picture? You can only choose one answer from {test_dataset.classes}. Answer:"
    question = f"Question: What is the main object in the picture? Answer:"
    print(f"开始生成VQA答案，结果将保存到: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(data_loader) as pbar:
            for images_tensor, labels, images_pil in pbar:
                vqa_answers = generate_vqa_answers(images_pil, processor, vqa_model, question)
                for label, vqa_ans in zip(labels, vqa_answers):
                    result = {
                        "label_id": int(label),
                        "class_name": test_dataset.classes[label],
                        "vqa_answer": vqa_ans
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                torch.cuda.empty_cache()

    print(f"✅ VQA结果已保存到: {output_path}")
    print("VQA处理完成!")

if __name__ == "__main__":
    main()