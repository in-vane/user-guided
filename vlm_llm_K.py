# 优化后的单文件 VQA-LLM 图像分类系统（统一 Prompt 版）

import os
import json
import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm
import Levenshtein

# ---------- 日志配置 ----------
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description='VQA-LLM图像分类系统')
parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'ppmi', 'oxford_pet'])
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--sample_size', type=int, default=200)
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
def load_models():
    logger.info("加载VQA模型和LLM模型…")
    vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", padding_side="left")
    llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    return vqa_model.eval(), processor, tokenizer, llm_model.eval()

# ---------- Prompt构建 ----------
def create_prompt(desc, criteria, candidates):
    if criteria == 'flying':
        return f"Analyze whether the object can fly:\nDescription: {desc}\nOptions: [flying, non-flying]\nAnswer:"
    elif criteria == 'situation':
        return f"You will be given a list of objects. Decide which of the following each object is more related to: 'Sky', 'Land', or 'Water'.\nDescription: {desc}\nOptions: [sky, land, water]\nAnswer:"
    elif criteria == 'living':
        return f"You will be given a list of object descriptions. Determine for each object whether it is 'existing in nature or 'man-made'. For 'existing in nature', answer 'organism' and for 'man-made', answer 'manufactured'.\nDescription: {desc}\nOptions: [organism, manufactured]\nAnswer:"
    elif criteria == 'instrument':
        return f"You will be given a list of objects. Decide which of the following each object is more related to: 'string', 'brass'.\nDescription: {desc}\nOptions: [string, brass]\nAnswer:"
    elif criteria == 'cifar20':
        return f"You will be given a list of object descriptions. Decide which of the following each object is more related to: 'organism', 'plant', 'man-made', 'landscape'.\nDescription: {desc}\nOptions: ['organism', 'plant', 'man-made', 'landscape']\nAnswer:"
    else:
        return f"Classify the object:\nDescription: {desc}\nOptions: {', '.join(candidates)}\nAnswer:"

# ---------- 置信度计算 ----------
def compute_constrained_confidence(prompt, word, candidate_set, tokenizer, model):
    try:
        # 编码 prompt
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**input_ids).logits[:, -1, :]  # 取最后一个 token 的 logits
            probs = F.softmax(logits[0], dim=-1)

        # 尝试编码预测单词
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            logger.warning(f"[Tokenizer Warning] Word [{word}] could not be tokenized.")
            return 0.5
        token_id = token_ids[0]

        # 计算候选集的总概率
        total = 0
        candidate_probs = []
        for c in candidate_set:
            c_token_ids = tokenizer.encode(c, add_special_tokens=False)
            if not c_token_ids:
                logger.warning(f"[Tokenizer Warning] Candidate [{c}] could not be tokenized, skipping.")
                continue
            c_prob = probs[c_token_ids[0]].item()
            candidate_probs.append((c, c_prob))
            total += c_prob

        if total == 0:
            logger.warning(f"[Probability Warning] Sum of candidate probabilities is zero. word={word}")
            return 0.5

        # 归一化目标概率
        prob = probs[token_id].item()
        norm_prob = prob / total

        # 直接返回归一化概率（不再 sigmoid 缩放，可根据需要调节）
        confidence = round(min(max(norm_prob, 0.0), 1.0), 4)

        # logger.info(f"[Confidence] Word: {word}, Prob: {prob:.4f}, Sum: {total:.4f}, Norm: {confidence:.4f}")
        # logger.info(f"[Candidates Prob] " + ", ".join([f"{c}:{p:.4f}" for c, p in candidate_probs]))

        return confidence

    except Exception as e:
        logger.warning(f"[Exception] Confidence calculation failed for word [{word}]: {e}")
        return 0.5

# ---------- 标签生成 ----------
def generate_constrained_labels(pil_imgs, labels, fout, candidate_set, processor, vqa_model, tokenizer, llm_model, class_names, question, criteria):
    inputs = processor(images=pil_imgs, text=[question]*len(pil_imgs), return_tensors="pt", padding=True).to(vqa_model.device)
    with torch.no_grad():
        vqa_out = vqa_model.generate(**inputs, use_cache=False)
    answers = processor.batch_decode(vqa_out, skip_special_tokens=True)
    answers = [ans.split("Answer:")[-1].strip() for ans in answers]
    
    prompts = [create_prompt(desc, criteria, candidate_set) for desc in answers]
    
    tok_in = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(llm_model.device)
    with torch.no_grad():
        out = llm_model.generate(**tok_in, max_new_tokens=5)
    decs = tokenizer.batch_decode(out, skip_special_tokens=True)
    
    for label, vqa_ans, prompt, dec in zip(labels, answers, prompts, decs):
        word = dec[len(prompt):].strip().split()[0].strip('.,!?').lower()
        if word not in candidate_set:
            word = min(candidate_set, key=lambda x: Levenshtein.distance(x, word))
        conf = compute_constrained_confidence(prompt, word, candidate_set, tokenizer, llm_model)
        result = {
            "label_id": int(label),
            "class_name": class_names[label],
            "vqa_answer": vqa_ans,
            "llm_answer": word,
            "confidence": conf
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

# ---------- 主处理流程 ----------
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    # question = f"Question: What is the object in the picture? You can only choose one answer from [{test_dataset.classes}]. Answer:"
    # question = "Question: What instrument is in this picture? Answer:"
    # question = "Question: What is the specific species of animal in the picture? Answer:"
    # question = "Question: What color is the animal in the picture? Answer:"
    # question = "Question: What is the number in the picture? Answer:"
    vqa_model, processor, tokenizer, llm_model = load_models()
    
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"加载 {args.dataset.upper()} 数据集...")
    test_dataset = CustomDataset(dataset_name=args.dataset, transform=transform_tensor, split='test')
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    question = f"Question: What is the object in the picture? You can only choose one answer from {test_dataset.classes}. Answer:"
    print(f"使用问题: {question}")

    # 设置候选集
    if args.criteria == "flying":
        candidate_set = ["flying", "non-flying"]
    elif args.criteria == "habitat":
        candidate_set = ["sky", "land", "water"]
    elif args.criteria == "living":
        candidate_set = ["organism", "manufactured"]
    elif args.criteria == "instrument":
        candidate_set = ["string", "brass"]
    elif args.criteria == "cifar20":
        candidate_set = ["organism", "plant", "man-made", "landscape"]
    else:
        candidate_set = test_dataset.classes

    print(f"候选集内容 ({len(candidate_set)} 类): {candidate_set}")
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_results_k{args.k}_{args.criteria}.jsonl")
    print(f"开始生成标签，结果将保存到: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(data_loader) as pbar:
            for images_tensor, labels, images_pil in pbar:
                generate_constrained_labels(
                    images_pil,
                    labels,
                    f,
                    candidate_set,
                    processor,
                    vqa_model,
                    tokenizer,
                    llm_model,
                    test_dataset.classes,
                    question,
                    criteria=args.criteria
                )
                torch.cuda.empty_cache()

    print(f"✅ 结果已保存到: {output_path}")
    print("处理完成!")

if __name__ == "__main__":
    main()
