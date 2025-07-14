import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载模型 ----------
vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
vqa_model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
llm_model.eval()

# ---------- transform & dataset ----------
transform_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        pil_img = Image.fromarray(img).convert("RGB")
        tensor_img = self.transform(pil_img) if self.transform else pil_img
        return tensor_img, target, pil_img
# class STL10(datasets.STL10):
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         pil_img = Image.fromarray(img).convert("RGB")
#         tensor_img = self.transform(pil_img) if self.transform else pil_img
#         return tensor_img, target, pil_img

def custom_collate_fn(batch):
    tensors, labels, pil_images = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(pil_images)

dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_tensor)
# dataset = STL10(root="./data", split='train', download=True, transform=transform_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
class_names = dataset.classes

# question = "Question: What is the object in the picture? Answer:"
question = "Question: Can the object in the picture fly? Answer:"
# system_prompt = "You will be given a list of objects. Determine whether each can fly. Answer only 'yes' or 'no'."
# system_prompt = "You will be given a list of objects. For each object, determine whether it is a vehicle. Answer only 'yes' or 'no'."
system_prompt = "You will be given a list of objects. Decide which of the following each object is more related to: 'Sky', 'Land', or 'Water'. Simply answer 'Sky' or 'Land' or 'Water'."

# ---------- 置信度计算 ----------
def compute_confidence(prompt, answer, tokenizer, model):
    full_input = prompt + answer
    input_ids = tokenizer(full_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    probs = []
    for i in range(len(answer_ids)):
        logit_index = -len(answer_ids) + i - 1
        logit = logits[0, logit_index, :]
        prob = F.softmax(logit, dim=-1)[answer_ids[i]].item()
        probs.append(prob)

    joint_confidence = torch.tensor(probs).prod().item()
    return joint_confidence

# ---------- VQA + LLM 推理 ----------
def generate_text_labels_with_vqa(images_pil, vqa_model, processor, labels, file_writer):
    inputs = processor(images=images_pil, text=[question] * len(images_pil), return_tensors="pt", padding=True).to(device)
    outputs = vqa_model.generate(**inputs)
    answers = processor.batch_decode(outputs, skip_special_tokens=True)
    answers = [ans.split("Answer:")[-1].strip() for ans in answers]

    prompts = [f"{system_prompt}\nItem: {str(item)}\nAnswer:" for item in answers]
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **input_ids,
            max_new_tokens=5,
            do_sample=False
        )
    full_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for idx, (label, item, full_text) in enumerate(zip(labels, answers, full_texts)):
        prompt = prompts[idx]
        generated = full_text[len(prompt):].strip().lower()

        if "yes" in generated:
            answer = " yes"
        elif "no" in generated:
            answer = " no"
        else:
            answer = generated
            
        # if "sky" in generated.lower():
        #     answer = "Sky"
        # elif "land" in generated.lower():
        #     answer = "Land"
        # elif "water" in generated.lower():
        #     answer = "Water"
        # else:
        #     # 未明确分类时使用置信度最高的选项
        #     answer = generated  # 默认值

        confidence = compute_confidence(prompt, answer, tokenizer, llm_model)
        class_name = class_names[int(label)]
        file_writer.write(json.dumps({
            "label_id": int(label),
            "class_name": class_name,
            "vqa_answer": str(item),
            "llm_answer": answer.strip(),
            "confidence": confidence
        }, ensure_ascii=False) + "\n")

# ---------- 主处理流程 ----------
output_path = "results/cifar10_results_k3_hlk.jsonl"
pbar = tqdm(data_loader)

with open(output_path, "w", encoding="utf-8") as f:
    for images_tensor, labels, images_pil in pbar:
        images_tensor = images_tensor.to(device)
        generate_text_labels_with_vqa(images_pil, vqa_model, processor, labels, f)

print(f"✅ 已保存到：{output_path}")
