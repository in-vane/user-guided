import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载 VQA 模型 ----------
vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
vqa_model.eval()

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

def custom_collate_fn(batch):
    tensors, labels, pil_images = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(pil_images)

dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
class_names = dataset.classes

# ---------- VQA 问题 ----------
question = "Question: Can the object in the picture fly? Answer:"

# ---------- 仅 VQA 推理 ----------
def generate_text_labels_with_vqa(images_pil, vqa_model, processor, labels, file_writer):
    inputs = processor(images=images_pil, text=[question] * len(images_pil), return_tensors="pt", padding=True).to(device)
    outputs = vqa_model.generate(**inputs)
    answers = processor.batch_decode(outputs, skip_special_tokens=True)
    answers = [ans.split("Answer:")[-1].strip() for ans in answers]

    for label, vqa_answer in zip(labels, answers):
        class_name = class_names[int(label)]
        file_writer.write(json.dumps({
            "label_id": int(label),
            "class_name": class_name,
            "vqa_answer": vqa_answer
        }, ensure_ascii=False) + "\n")

# ---------- 主处理流程 ----------
output_path = "results/cifar10_results_vqa_only.jsonl"
os.makedirs("results", exist_ok=True)
pbar = tqdm(data_loader)

with open(output_path, "w", encoding="utf-8") as f:
    for images_tensor, labels, images_pil in pbar:
        images_tensor = images_tensor.to(device)
        generate_text_labels_with_vqa(images_pil, vqa_model, processor, labels, f)

print(f"✅ 已保存到：{output_path}")
