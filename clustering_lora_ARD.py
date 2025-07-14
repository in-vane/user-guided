import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
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
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'stl10', 'ppmi', 'oxford_pet'], help='选择数据集')
parser.add_argument('--text_key', type=str, default='llm_answer', help='jsonl文件中用于聚类的文本字段名')
parser.add_argument('--k', type=int, default=10, help='类别数')
parser.add_argument('--c', type=str, default="", help='类别映射文件路径（可选）')
args = parser.parse_args()
checkpoint_path = f"checkpoints/lora_openclip_ard_{args.dataset}.pth"
jsonl_path = f"results/{args.dataset}_results_k{args.k}{args.c}.jsonl"
print(f"Using dataset: {args.dataset}, text key: {args.text_key}, jsonl_path: {jsonl_path}")

# ---------- ARD-LoRA 定义 ----------
class ARDLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.log_lambda = nn.Parameter(torch.zeros(out_dim))  # 初始 lambda = 1，避免过度稀疏

    def forward(self, x):
        delta_W = (self.B @ self.A) * (self.alpha / self.rank)
        lambda_scale = torch.exp(-0.5 * self.log_lambda)
        out = F.linear(x, delta_W)
        return out * lambda_scale.unsqueeze(0)

    def kl_regularization(self):
        lamb = self.log_lambda.exp()
        kl = 0.5 * torch.sum(lamb - self.log_lambda - 1.0)
        l2_reg = (self.A ** 2).sum() + (self.B ** 2).sum()
        return kl + 1e-3 * l2_reg

# ---------- 加载 OpenCLIP 模型 ----------
model_vision, preprocess, _ = create_model_and_transforms("ViT-B-32-quickgelu", pretrained="metaclip_fullcc")
model_vision.to(device).eval()

# ---------- 加载 ARD-LoRA 权重 ----------
lora = ARDLoRALayer(512, 512, rank=4).to(device)

if os.path.exists(checkpoint_path):
    lora.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ 已加载 LoRA 权重：{checkpoint_path}")
else:
    raise FileNotFoundError("❌ 未找到 LoRA 权重，请先进行训练。")

# ---------- text 投影模块 ----------
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

# ---------- 数据与加载器 ----------
transform_tensor = preprocess

class DatasetWithPIL(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, transform):
        assert dataset_name in ['mnist', 'cifar10', 'stl10', 'ppmi', 'oxford_pet']
        self.transform = transform
        if dataset_name == 'mnist':
            self.base = datasets.MNIST(root="./data", train=(split == 'train'), download=True)
        elif dataset_name == 'cifar10':
            self.base = datasets.CIFAR10(root="./data", train=(split == 'train'), download=True)
        elif dataset_name == 'stl10':
            self.base = datasets.STL10(root="./data", split=split, download=True)
        elif dataset_name == 'ppmi':
            full_dataset = ImageFolder(root="./data/ppmi/play_instrument")
            samples = [(p, l) for p, l in full_dataset.samples if f"/{split}/" in p.replace("\\", "/")]
            self.samples = samples
            self.classes = full_dataset.classes
            self.class_to_idx = full_dataset.class_to_idx
        elif dataset_name == 'oxford_pet':
            self.base = datasets.OxfordIIITPet(root="./data", split=split, download=True)
        self.dataset_name = dataset_name

    def __len__(self):
        if self.dataset_name == 'ppmi':
            return len(self.samples)
        else:
            return len(self.base)

    def __getitem__(self, idx):
        if self.dataset_name == 'mnist':
            img, target = self.base.data[idx], int(self.base.targets[idx])
            img = img.numpy()
            pil_img = Image.fromarray(img).convert("RGB")
        elif self.dataset_name == 'cifar10':
            img, target = self.base.data[idx], int(self.base.targets[idx])
            pil_img = Image.fromarray(img).convert("RGB")
        elif self.dataset_name == 'stl10':
            img, target = self.base.data[idx], int(self.base.labels[idx])
            pil_img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert("RGB")
        elif self.dataset_name == 'ppmi':
            path, target = self.samples[idx]
            pil_img = Image.open(path).convert("RGB")
        elif self.dataset_name == 'oxford_pet':
            img, target = self.base[idx]
            pil_img = img.convert("RGB")
            target = int(target)
        tensor_img = self.transform(pil_img) if self.transform else pil_img
        return tensor_img, target, pil_img
    
def custom_collate_fn(batch):
    tensors, labels, pil_images = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(pil_images)

# ---------- 根据参数加载数据集 ----------
if args.dataset == 'mnist':
    split = 'test'
elif args.dataset == 'cifar10':
    split = 'test'
elif args.dataset == 'stl10':
    split = 'test'
elif args.dataset == 'ppmi':
    split = 'test'
elif args.dataset == 'oxford_pet':
    split = 'test'
else:
    raise ValueError("不支持的数据集")

dataset = DatasetWithPIL(dataset_name=args.dataset, split=split, transform=transform_tensor)
data_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)

# ---------- 读取文本伪标签 ----------
print(f"Loading JSONL data from {jsonl_path}...")
assert os.path.exists(jsonl_path), "缺少 jsonl 文件"
text_str = args.text_key
print(f"Using {text_str}...")
with open(jsonl_path, "r", encoding="utf-8") as f:
    json_data = [json.loads(line) for line in f.readlines()]

text_labels_all = [entry[text_str] for entry in json_data]
assert len(text_labels_all) == len(dataset), f"JSONL 样本数量 {len(text_labels_all)} 与数据集数量 {len(dataset)} 不一致"
conf_all = [entry["confidence"] for entry in json_data]

# ---------- 特征提取 ----------
all_v_feats, all_t_feats, all_fused_feats, all_labels = [], [], [], []
start_idx = 0

for images_tensor, labels, images_pil in tqdm(data_loader):
    batch_size = len(images_pil)
    images_tensor = images_tensor.to(device)
    text_labels = text_labels_all[start_idx:start_idx + batch_size]
    conf = torch.tensor(conf_all[start_idx:start_idx + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
    start_idx += batch_size

    with torch.no_grad():
        visual_feats = model_vision.encode_image(images_tensor)
        visual_feats = F.normalize(visual_feats, dim=-1)
        visual_feats = F.normalize(visual_feats + lora(visual_feats), dim=-1)

        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = model_text(**text_inputs).text_embeds
        text_raw = text_embeds.mean(dim=1)
        text_out = F.normalize(text_raw, dim=-1)

        text_out_proj = text_proj(text_out)
        fused = (1 - conf) * visual_feats + conf * text_out_proj

    all_v_feats.append(visual_feats.cpu())
    all_t_feats.append(text_out.cpu())
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    preds = kmeans.fit_predict(features)
    ari = adjusted_rand_score(labels_true, preds)
    nmi = normalized_mutual_info_score(labels_true, preds)
    acc = cluster_accuracy(labels_true, preds)
    return acc, ari, nmi, preds

def map_to_class(y):
    # y: numpy array of original labels
    y_new = np.zeros_like(y)
    # # vehicle
    # y_new[np.isin(y, [0, 1, 8, 9])] = 0
    # y_new[np.isin(y, [2, 3, 4, 5, 6, 7])] = 1
    # # habitat: water、land、sky
    # y_new[np.isin(y, [0, 2])] = 0
    # y_new[np.isin(y, [8])] = 1
    # y_new[np.isin(y, [1, 3, 4, 5, 6, 7, 9])] = 2
    # # instrument: string brass
    # y_new[np.isin(y, [0, 1, 2, 4])] = 0
    # y_new[np.isin(y, [3, 5])] = 1
    # # instrument: black white gray brown
    # y_new[np.isin(y, [7, 17, 18, 22, 30, ])] = 0   # black
    # y_new[np.isin(y, [1, 6, 10, 13, 15, 16, 23, 26, 29, 32, 33, 35])] = 1   # white
    # y_new[np.isin(y, [9, 11, 20, 27, ])] = 2   # gray
    # y_new[np.isin(y, [0, 2, 3, 4, 5, 8, 12, 14, 19, 21, 24, 25, 28, 31, 34, 36])] = 3 # brown
    # # range: small medium large
    # y_new[np.isin(y, [0, 1, 2, 3])] = 0   # small
    # y_new[np.isin(y, [4, 5, 6])] = 1   # medium
    # y_new[np.isin(y, [7, 8, 9])] = 2   # large
    # living: living non-living
    y_new[np.isin(y, [1, 3, 4, 5, 6, 7])] = 0   # living
    y_new[np.isin(y, [0, 2, 8, 9])] = 1   # non-living
    return y_new

mapped_labels = map_to_class(true_labels)
num_classes = args.k

acc_v, ari_v, nmi_v, preds_v = evaluate_kmeans(v_feats, mapped_labels, n_clusters=num_classes)
acc_t, ari_t, nmi_t, preds_t = evaluate_kmeans(t_feats, mapped_labels, n_clusters=num_classes)
acc_f, ari_f, nmi_f, preds_f = evaluate_kmeans(fused_feats, mapped_labels, n_clusters=num_classes)

print("\nClustering Results with ARD-LoRA-enhanced Features:")
print("{:<25} {:<10} {:<10} {:<10}".format("Method", "ACC", "ARI", "NMI"))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Vision + ARD-LoRA", acc_v, ari_v, nmi_v))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Text", acc_t, ari_t, nmi_t))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Fused (V+T)", acc_f, ari_f, nmi_f))

# ---------- 结果可视化 ----------
# def plot_tsne_multi(features_list, preds_list, titles, num_classes):
#     sns.set_theme(style="whitegrid", font_scale=1.2)
#     plt.rcParams['figure.facecolor'] = 'white'
#     fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=120)

#     for i, (features, preds, title) in enumerate(zip(features_list, preds_list, titles)):
#         tsne = TSNE(n_components=2, random_state=42, perplexity=150, n_iter=1000)
#         features_2d = tsne.fit_transform(features)
#         palette = sns.color_palette("husl", n_colors=num_classes)
#         ax = axes[i]
#         for cluster_id in range(num_classes):
#             mask = preds == cluster_id
#             ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
#                        color=palette[cluster_id],
#                        label=f'Cluster {cluster_id}',
#                        s=15, alpha=0.3)
#         ax.set_title(title, fontsize=14, pad=20)
#         ax.set_xlabel('t-SNE 1', fontsize=12)
#         ax.set_ylabel('t-SNE 2', fontsize=12)
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#         ax.grid(True, alpha=0.3)

#     plt.suptitle('t-SNE Visualization of Predicted Clusters', y=1.02, fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'tsne_predicted_clusters_ard{args.c}.pdf', bbox_inches='tight')
#     plt.close()

# # 调用新函数
# plot_tsne_multi(
#     [v_feats, t_feats, fused_feats],
#     [preds_v, preds_t, preds_f],
#     ['Vision Features', 'Text Features', 'Fused Features'],
#     num_classes
# )