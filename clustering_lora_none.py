import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from transformers import Blip2Processor, Blip2TextModelWithProjection
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
import argparse
import json
from open_clip import create_model_and_transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10'], help='选择数据集')
parser.add_argument('--text_key', type=str, default='llm_answer', help='jsonl文件中用于聚类的文本字段名')
parser.add_argument('--k', type=int, default=10, help='类别数')
parser.add_argument('--c', type=str, default="", help='类别映射文件路径（可选）')
args = parser.parse_args()
jsonl_path = f"results/{args.dataset}_results_k{args.k}{args.c}.jsonl"
print(f"Using dataset: {args.dataset}, text key: {args.text_key}, jsonl_path: {jsonl_path}")

# ---------- 加载 OpenCLIP 模型（用于 vision 替换） ----------
model_vision, preprocess, _ = create_model_and_transforms("ViT-B-32-quickgelu", pretrained="metaclip_fullcc")
model_vision.to(device).eval()

# ---------- text 投影模块（将 256 -> 512） ----------
class TextProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)

text_proj = TextProjector().to(device)

# ---------- 加载 BLIP2 的 text 模型 ----------
model_text = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
model_text.eval()

# ---------- transform & dataset ----------
transform_tensor = preprocess

class DatasetWithPIL(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, transform):
        assert dataset_name in ['cifar10', 'stl10']
        self.transform = transform
        if dataset_name == 'cifar10':
            self.base = datasets.CIFAR10(root="./data", train=(split == 'train'), download=True)
        elif dataset_name == 'stl10':
            self.base = datasets.STL10(root="./data", split=split, download=True)
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.dataset_name == 'cifar10':
            img, target = self.base.data[idx], int(self.base.targets[idx])
            pil_img = Image.fromarray(img).convert("RGB")
        elif self.dataset_name == 'stl10':
            img, target = self.base.data[idx], int(self.base.labels[idx])
            pil_img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert("RGB")
        tensor_img = self.transform(pil_img) if self.transform else pil_img
        return tensor_img, target, pil_img

def custom_collate_fn(batch):
    tensors, labels, pil_images = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(pil_images)

# ---------- 根据参数加载数据集 ----------
if args.dataset == 'cifar10':
    split = 'test'
elif args.dataset == 'stl10':
    split = 'test'
else:
    raise ValueError("不支持的数据集")

dataset = DatasetWithPIL(dataset_name=args.dataset, split=split, transform=transform_tensor)
data_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)

# ---------- 加载 pseudo text labels & conf ----------
print(f"Loading JSONL data from {jsonl_path}...")
text_str = args.text_key
print(f"Using {text_str}...")
assert os.path.exists(jsonl_path), "缺少 jsonl 文件"
with open(jsonl_path, "r", encoding="utf-8") as f:
    json_data = [json.loads(line) for line in f.readlines()]

text_labels_all = [entry[text_str] for entry in json_data]
conf_all = [entry["confidence"] for entry in json_data]

# ---------- 特征提取 ----------
all_v_feats, all_t_feats, all_fused_feats, all_labels = [], [], [], []

start_idx = 0
for images_tensor, labels, images_pil in tqdm(data_loader):
    batch_size = len(images_pil)
    images_tensor = images_tensor.to(device)

    # 从 jsonl 中提取当前 batch 的 text_labels 和 confidence
    text_labels = text_labels_all[start_idx: start_idx + batch_size]
    conf = conf_all[start_idx: start_idx + batch_size]
    conf = torch.tensor(conf, dtype=torch.float32, device=device).unsqueeze(1)  # (B, 1)
    start_idx += batch_size

    with torch.no_grad():
        # visual_inputs = processor(images=images_tensor, return_tensors="pt").to(device)
        # visual_embeds = model_vision(**visual_inputs).image_embeds  # (B, 1, D)
        # v_out = visual_embeds[:, 0, :]
        visual_feats = model_vision.encode_image(images_tensor)
        visual_feats = F.normalize(visual_feats, dim=-1)

        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = model_text(**text_inputs).text_embeds
        text_out = text_embeds.mean(dim=1)
        
        text_out_proj = text_proj(text_out)
        fused = (1 - conf) * visual_feats + conf * text_out_proj
        # fused = 0.5 * visual_feats + 0.5 * text_out_proj
        # fused = (1 - conf) * F.normalize(v_out, dim=-1) + conf * F.normalize(t_out, dim=-1)

    all_v_feats.append(F.normalize(visual_feats, dim=-1).cpu())
    all_t_feats.append(F.normalize(text_out_proj, dim=-1).cpu())
    all_fused_feats.append(fused.cpu())
    all_labels.append(labels)

v_feats = torch.cat(all_v_feats, dim=0).numpy()
t_feats = torch.cat(all_t_feats, dim=0).numpy()
fused_feats = torch.cat(all_fused_feats, dim=0).numpy()
true_labels = torch.cat(all_labels, dim=0).numpy()

# ---------- 聚类评估 ----------
def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

def evaluate_kmeans(features, labels_true, n_clusters):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if features.ndim == 3:
        features = features.mean(axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    preds = kmeans.fit_predict(features)
    ari = adjusted_rand_score(labels_true, preds)
    nmi = normalized_mutual_info_score(labels_true, preds)
    acc = cluster_accuracy(labels_true, preds)
    return acc, ari, nmi, preds


# ---------- 输出评估结果 ----------
def map_to_class(y):
    # y: numpy array of original labels
    y_new = np.zeros_like(y)
    y_new[np.isin(y, [0, 2])] = 0
    y_new[np.isin(y, [8])] = 1
    y_new[np.isin(y, [1, 3, 4, 5, 6, 7, 9])] = 2
    return y_new

mapped_labels = map_to_class(true_labels)
num_classes = args.k
acc_v, ari_v, nmi_v, preds_v = evaluate_kmeans(v_feats, mapped_labels, n_clusters=num_classes)
acc_t, ari_t, nmi_t, preds_t = evaluate_kmeans(t_feats, mapped_labels, n_clusters=num_classes)
acc_f, ari_f, nmi_f, preds_f = evaluate_kmeans(fused_feats, mapped_labels, n_clusters=num_classes)


print("\nClustering Results (KMeans + Hungarian Matching):")
print("{:<25} {:<10} {:<10} {:<10}".format("Method", "ACC", "ARI", "NMI"))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Vision Only", acc_v, ari_v, nmi_v))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Text Only", acc_t, ari_t, nmi_t))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Fused (V+T)", acc_f, ari_f, nmi_f))

# # ---------- 结果可视化 ----------
# def plot_tsne(features, labels, preds, title, save_path="t-SNE.pdf"):
#     # 使用 t-SNE 降维到2D
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     features_2d = tsne.fit_transform(features)
    
#     # 创建画布
#     plt.figure(figsize=(12, 6))
    
#     # 子图1: 真实标签可视化
#     plt.subplot(1, 2, 1)
#     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', s=5)
#     plt.title(f'True Labels ({title})')
#     plt.colorbar(scatter, boundaries=range(num_classes+1)).set_ticks(range(num_classes))
    
#     # 子图2: 预测聚类可视化
#     plt.subplot(1, 2, 2)
#     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=preds, cmap='tab20', s=5)
#     plt.title(f'Predicted Clusters ({title})')
#     plt.colorbar(scatter, boundaries=range(num_classes+1)).set_ticks(range(num_classes))
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# plot_tsne(v_feats, true_labels, preds_v, 'Vision Features', 'tsne_vision.pdf')
# plot_tsne(t_feats, true_labels, preds_t, 'Text Features', 'tsne_text.pdf')
# plot_tsne(fused_feats, true_labels, preds_f, 'Fused Features', 'tsne_fused.pdf')