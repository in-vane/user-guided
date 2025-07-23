# 优化后的单文件 VQA-LLM 图像分类系统（LLM部分）

import os
import json
import torch
import argparse
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import Levenshtein

# ---------- 日志配置 ----------
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description='VQA-LLM图像分类系统 - LLM部分')
parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'ppmi', 'oxford_pet'])
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for LLM processing')
parser.add_argument('--input_dir', type=str, default='results')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--criteria', type=str, default='default', choices=['range', 'flying', 'living', 'habitat', 'instrument', 'species', 'number', 'default'])
args = parser.parse_args()

# ---------- 数据集类 ----------
class CustomDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.classes = self.load_classes()

    def load_classes(self):
        if self.dataset_name == 'mnist':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_name == 'cifar100':
            return ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 
                    'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
        elif self.dataset_name == 'stl10':
            return ['airplane', 'bird', 'car', 'cat', 'deer',
                    'dog', 'horse', 'monkey', 'ship', 'truck']
        elif self.dataset_name == 'ppmi':
            return ['bassoon', 'flute', 'frenchhorn', 'guitar', 'saxophone', 'violin']
        elif self.dataset_name == 'oxford_pet':
            return ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 
                    'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 
                    'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 
                    'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
                    'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 
                    'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
        return []

# ---------- 模型加载 ----------
def load_llm_model():
    logger.info("加载LLM模型…")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", padding_side="left")
    llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, llm_model.eval()

# ---------- Prompt构建 ----------
def create_prompt(desc, criteria, candidates):
    if criteria == 'range':
        return f"You will get a list of descriptions of images from the MNIST dataset. Determine whether the number in each description falls into the range of small (0, 1, 2, 3), medium (4, 5, 6), or large (7, 8, 9).\nDescription: {desc}\nOptions: [small, medium, large]\nAnswer:"
    elif criteria == 'number':
        return f"You will be given a list of objects. Decide which of the following each object is more related to: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'.\nDescription: {desc}\nOptions: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']\nAnswer:"
    elif criteria == 'flying':
        return f"Analyze whether the object can fly:\nDescription: {desc}\nOptions: [flying, non-flying]\nAnswer:"
    elif criteria == 'habitat':
        return f"You will be given a list of objects. Decide which of the following each object is more related to: 'Sky', 'Land', or 'Water'.\nDescription: {desc}\nOptions: [sky, land, water]\nAnswer:"
    elif criteria == 'living':
        return f"You will be given a list of object descriptions. Determine for each object whether it is 'existing in nature or 'man-made'. For 'existing in nature', answer 'organism' and for 'man-made', answer 'manufactured'.\nDescription: {desc}\nOptions: [organism, manufactured]\nAnswer:"
    elif criteria == 'instrument':
        return f"You will be given a list of objects. Decide which of the following each object is more related to: 'string', 'brass'.\nDescription: {desc}\nOptions: [string, brass]\nAnswer:"
    elif criteria == 'species':
        return f"You will be given a list of object descriptions. Please determine whether each object is 'animal', 'plant', 'human', or 'non-living'.\nDescription: {desc}\nOptions: ['animal', 'plant', 'human', 'non-living']\nAnswer:"
    else:
        return f"Classify the object:\nDescription: {desc}\nOptions: {', '.join(candidates)}\nAnswer:"

# ---------- 置信度计算 ----------
def compute_constrained_confidence(prompt, word, candidate_set, tokenizer, model):
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**input_ids).logits[:, -1, :]
            probs = F.softmax(logits[0], dim=-1)

        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            logger.warning(f"[Tokenizer Warning] Word [{word}] could not be tokenized.")
            return 0.5
        token_id = token_ids[0]

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

        prob = probs[token_id].item()
        norm_prob = prob / total
        confidence = round(min(max(norm_prob, 0.0), 1.0), 4)

        logger.info(f"[Confidence] Word: {word}, Prob: {prob:.4f}, Sum: {total:.4f}, Norm: {confidence:.4f}")
        logger.info(f"[Candidates Prob] " + ", ".join([f"{c}:{p:.4f}" for c, p in candidate_probs]))

        return confidence

    except Exception as e:
        logger.warning(f"[Exception] Confidence calculation failed for word [{word}]: {e}")
        return 0.5

# ---------- LLM标签生成 ----------
def process_vqa_results(vqa_results, fout, candidate_set, tokenizer, llm_model, class_names, criteria, batch_size):
    for i in range(0, len(vqa_results), batch_size):
        batch_results = vqa_results[i:i + batch_size]
        prompts = [create_prompt(res["vqa_answer"], criteria, candidate_set) for res in batch_results]
        
        tok_in = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(llm_model.device)
        with torch.no_grad():
            out = llm_model.generate(**tok_in, max_new_tokens=5)
        decs = tokenizer.batch_decode(out, skip_special_tokens=True)
        
        for res, prompt, dec in zip(batch_results, prompts, decs):
            word = dec[len(prompt):].strip().split()[0].strip('.,!?').lower()
            if word not in candidate_set:
                word = min(candidate_set, key=lambda x: Levenshtein.distance(x, word))
            conf = compute_constrained_confidence(prompt, word, candidate_set, tokenizer, llm_model)
            result = {
                "label_id": res["label_id"],
                "class_name": res["class_name"],
                "vqa_answer": res["vqa_answer"],
                "llm_answer": word,
                "confidence": conf
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()

# ---------- 主处理流程 ----------
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, llm_model = load_llm_model()
    
    dataset = CustomDataset(dataset_name=args.dataset)
    
    # 设置候选集
    if args.criteria == "range":
        candidate_set = ["small", "medium", "large"]
    elif args.criteria == "flying":
        candidate_set = ["flying", "non-flying"]
    elif args.criteria == "situation":
        candidate_set = ["sky", "land", "water"]
    elif args.criteria == "living":
        candidate_set = ["organism", "manufactured"]
    elif args.criteria == "instrument":
        candidate_set = ["string", "brass"]
    elif args.criteria == "species":
        candidate_set = ['animal', 'plant', 'human', 'non-living']
    elif args.criteria == "number":
        candidate_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        candidate_set = dataset.classes

    print(f"候选集内容 ({len(candidate_set)} 类): {candidate_set}")
    
    input_path = os.path.join(args.input_dir, f"{args.dataset}_vqa_intermediate.jsonl")
    output_path = os.path.join(args.output_dir, f"{args.dataset}_results_k{args.k}_{args.criteria}.jsonl")
    print(f"从 {input_path} 加载VQA结果...")
    print(f"开始生成LLM标签，结果将保存到: {output_path}")
    
    vqa_results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            vqa_results.append(json.loads(line.strip()))
    
    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(total=len(vqa_results)) as pbar:
            process_vqa_results(
                vqa_results,
                f,
                candidate_set,
                tokenizer,
                llm_model,
                dataset.classes,
                criteria=args.criteria,
                batch_size=args.batch_size
            )
            pbar.update(len(vqa_results))
            torch.cuda.empty_cache()

    print(f"✅ 结果已保存到: {output_path}")
    print("LLM处理完成!")

if __name__ == "__main__":
    main()