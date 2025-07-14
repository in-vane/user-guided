import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from transformers import Blip2Processor, Blip2VisionModelWithProjection, Blip2TextModelWithProjection
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载模型 ----------
model_vision = Blip2VisionModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
model_text = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
model_vision.eval()
model_text.eval()

# ---------- 复制原始投影权重 ----------
with torch.no_grad():
    W0_v = model_vision.vision_projection.weight.data.clone().detach()
    W0_t = model_text.text_projection.weight.data.clone().detach()

# ---------- 定义 LoRA 层 ----------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return (self.B @ (self.A @ x.T)).T * self.scaling

lora_v = LoRALayer(256, 256, rank=4).to(device)
lora_t = LoRALayer(256, 256, rank=4).to(device)

# ---------- 自定义数据集 ----------
class BLIP2CompatibleDataset(Dataset):
    def __init__(self, dataset_name='mnist', split='train', root='./data',
                 transform=None, return_text=False):
        assert dataset_name in ['mnist', 'cifar10']
        assert split in ['train', 'test']
        self.return_text = return_text
        self.transform = transform

        is_train = (split == 'train')
        if dataset_name == 'mnist':
            self.base_dataset = datasets.MNIST(root=root, train=is_train, download=True)
            self.class_names = [str(i) for i in range(10)]
        elif dataset_name == 'cifar10':
            self.base_dataset = datasets.CIFAR10(root=root, train=is_train, download=True)
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_text:
            return image, label, self.class_names[label]
        return image, label

# ---------- 自定义 collate 函数 ----------
def collate_pil_with_text(batch):
    """
    处理 (PIL.Image, label, text_label) 的 batch。
    返回:
        - images: List[PIL.Image]
        - labels: Tensor
        - text_labels: List[str]
    """
    images, labels, text_labels = zip(*batch)
    return list(images), torch.tensor(labels), list(text_labels)

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'], help='选择数据集: mnist 或 cifar10')
args = parser.parse_args()
checkpoint_path = f"lora_weights_{args.dataset}-1.pth"

# ---------- 加载数据集 ----------
train_dataset = BLIP2CompatibleDataset(dataset_name=args.dataset, split='train', transform=None, return_text=True)
test_dataset = BLIP2CompatibleDataset(dataset_name=args.dataset, split='test', transform=None, return_text=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_pil_with_text)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_pil_with_text)
class_names = train_dataset.class_names

# ---------- 加载或训练 ----------
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lora_v.load_state_dict(checkpoint["lora_v_state_dict"])
    lora_t.load_state_dict(checkpoint["lora_t_state_dict"])
    print(f"✅ 已加载 LoRA 权重：{checkpoint_path}")
else:
    print("⚠️ 未检测到已保存的 LoRA 权重，将执行训练...")
    optimizer = torch.optim.Adam(list(lora_v.parameters()) + list(lora_t.parameters()), lr=1e-4)
    tau = 0.07

    for epoch in range(1):
        total_loss = 0.
        pbar = tqdm(train_loader)

        for images, labels, text_labels in pbar:
            visual_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                visual_embeds = model_vision(**visual_inputs).image_embeds
                text_embeds = model_text(**text_inputs).text_embeds

                visual_embeds_flat = visual_embeds.view(-1, 256)
                text_embeds_flat = text_embeds.view(-1, 256)

            lora_v_out = lora_v(visual_embeds_flat).reshape_as(visual_embeds)
            lora_t_out = lora_t(text_embeds_flat).reshape_as(text_embeds)
            y_v = visual_embeds + lora_v_out
            y_t = text_embeds + lora_t_out

            y_v_pooled = y_v.mean(dim=1)
            y_t_pooled = y_t.mean(dim=1)
            y_v_norm = F.normalize(y_v_pooled, dim=-1)
            y_t_norm = F.normalize(y_t_pooled, dim=-1)

            logits = y_v_norm @ y_t_norm.T / tau
            labels_pos = torch.arange(len(images)).to(device)
            loss = F.cross_entropy(logits, labels_pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        print(f"Epoch {epoch} avg loss: {total_loss / len(train_loader):.4f}")

    torch.save({
        "lora_v_state_dict": lora_v.state_dict(),
        "lora_t_state_dict": lora_t.state_dict(),
    }, checkpoint_path)
    print(f"✅ LoRA 权重已保存到 {checkpoint_path}")

# ---------- 验证 ----------
images, labels, text_labels = next(iter(test_loader))

visual_inputs = processor(images=images, return_tensors="pt").to(device)
text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    visual_embeds = model_vision(**visual_inputs).image_embeds  # [B, 1, 256]
    text_embeds = model_text(**text_inputs).text_embeds         # [B, seq_len, 256]

    # 统一处理：取均值池化
    visual_embeds_flat = visual_embeds.mean(dim=1)  # [B, 256]
    text_embeds_flat = text_embeds.mean(dim=1)      # [B, 256]

    y_v0 = F.normalize(visual_embeds_flat, dim=-1)
    y_t0 = F.normalize(text_embeds_flat, dim=-1)
    cos_sim0 = (y_v0 * y_t0).sum(dim=-1).cpu()
    print(f"Cosine similarity before LoRA: {cos_sim0.mean().item():.4f}")

    y_v = F.normalize(visual_embeds_flat + lora_v(visual_embeds_flat), dim=-1)
    y_t = F.normalize(text_embeds_flat + lora_t(text_embeds_flat), dim=-1)
    cos_sim = (y_v * y_t).sum(dim=-1).cpu()
    print(f"Cosine similarity after LoRA: {cos_sim.mean().item():.4f}")

# ---------- 绘图 ----------
plt.hist(cos_sim0.numpy(), bins=50, alpha=0.6, label="Before LoRA")
plt.hist(cos_sim.numpy(), bins=50, alpha=0.6, label="After LoRA")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Cosine Similarity Distribution (Modality Gap)")
plt.savefig(f"cosine_similarity_{args.dataset}.png")
