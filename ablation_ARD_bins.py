# clusteringARD + 置信度分段评估版

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
from matplotlib.colors import LinearSegmentedColormap

# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10'])
parser.add_argument('--text_key', type=str, default='llm_answer')
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--c', type=str, default="")
parser.add_argument('--conf_bins', type=int, default=10, help='置信度分段数量')
args = parser.parse_args()

checkpoint_path = f"checkpoints/lora_openclip_ard_{args.dataset}.pth"
jsonl_path = f"results/{args.dataset}_results_k{args.k}{args.c}.jsonl"

# ---------- ARD-LoRA ----------
class ARDLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.log_lambda = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        delta_W = (self.B @ self.A) * (self.alpha / self.rank)
        lambda_scale = torch.exp(-0.5 * self.log_lambda)
        return F.linear(x, delta_W) * lambda_scale.unsqueeze(0)

# ---------- 模型 ----------
model_vision, preprocess, _ = create_model_and_transforms("ViT-B-32-quickgelu", pretrained="metaclip_fullcc")
model_vision.to(device).eval()

lora = ARDLoRALayer(512, 512, rank=4).to(device)
lora.load_state_dict(torch.load(checkpoint_path, map_location=device))

model_text = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
model_text.eval()

class TextProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)

text_proj = TextProjector().to(device)

# ---------- 数据 ----------
class DatasetWithPIL(datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.STL10):
    def __init__(self, root="./data", split='test', transform=None):
        super().__init__(root=root, train=(split == 'train'), download=True) if args.dataset == 'cifar10' else super().__init__(root=root, split=split, download=True)
        self.transform = transform
        


    def __getitem__(self, idx):
        img, target = (self.data[idx], int(self.targets[idx])) if args.dataset == 'cifar10' else (self.data[idx], int(self.labels[idx]))
        pil_img = Image.fromarray(img if args.dataset == 'cifar10' else np.transpose(img, (1, 2, 0))).convert("RGB")
        return self.transform(pil_img), target, pil_img

data_loader = DataLoader(DatasetWithPIL(transform=preprocess), batch_size=256, shuffle=False,
                         collate_fn=lambda batch: (torch.stack([x[0] for x in batch]), torch.tensor([x[1] for x in batch]), [x[2] for x in batch]))

# ---------- 读取伪标签 ----------
with open(jsonl_path, "r", encoding="utf-8") as f:
    json_data = [json.loads(line) for line in f.readlines()]

text_labels_all = [entry[args.text_key] for entry in json_data]
conf_all = [entry["confidence"] for entry in json_data]

# ---------- 特征提取 ----------
all_v, all_t, all_fused, all_labels, all_conf = [], [], [], [], []
start_idx = 0

for images_tensor, labels, images_pil in tqdm(data_loader):
    bs = len(images_pil)
    images_tensor = images_tensor.to(device)
    text_labels = text_labels_all[start_idx: start_idx + bs]
    conf = torch.tensor(conf_all[start_idx: start_idx + bs], dtype=torch.float32, device=device).unsqueeze(1)
    start_idx += bs

    with torch.no_grad():
        v_feat = model_vision.encode_image(images_tensor)
        v_feat = F.normalize(v_feat + lora(F.normalize(v_feat, dim=-1)), dim=-1)

        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(device)
        t_raw = model_text(**text_inputs).text_embeds.mean(dim=1)
        t_out = F.normalize(t_raw, dim=-1)
        t_out_proj = text_proj(t_out)

        fused = (1 - conf) * v_feat + conf * t_out_proj

    all_v.append(v_feat.cpu())
    all_t.append(t_out.cpu())
    all_fused.append(fused.cpu())
    all_labels.append(labels)
    all_conf.append(conf.cpu())

v_feats = torch.cat(all_v, dim=0).numpy()
t_feats = torch.cat(all_t, dim=0).numpy()
fused_feats = torch.cat(all_fused, dim=0).numpy()
true_labels = torch.cat(all_labels, dim=0).numpy()
conf_array = torch.cat(all_conf, dim=0).numpy().flatten()

# ---------- 聚类 & 映射 ----------
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

def best_mapping(y_pred, y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_aligned = np.array([mapping[p] for p in y_pred])
    return y_aligned


def map_to_class(y):
    y_new = np.zeros_like(y)
    y_new[np.isin(y, [0, 2])] = 0
    y_new[np.isin(y, [8])] = 1
    y_new[np.isin(y, [1, 3, 4, 5, 6, 7, 9])] = 2
    return y_new

# mapped_labels = map_to_class(true_labels)
num_classes = args.k

acc_v, ari_v, nmi_v, preds_v = evaluate_kmeans(v_feats, true_labels, n_clusters=num_classes)
acc_t, ari_t, nmi_t, preds_t = evaluate_kmeans(t_feats, true_labels, n_clusters=num_classes)
acc_f, ari_f, nmi_f, preds_f = evaluate_kmeans(fused_feats, true_labels, n_clusters=num_classes)

aligned_preds_f = best_mapping(preds_t, true_labels)

print("\nClustering Results with ARD-LoRA-enhanced Features:")
print("{:<25} {:<10} {:<10} {:<10}".format("Method", "ACC", "ARI", "NMI"))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Vision + ARD-LoRA", acc_v, ari_v, nmi_v))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Text", acc_t, ari_t, nmi_t))
print("{:<25} {:.4f}     {:.4f}     {:.4f}".format("Fused (V+T)", acc_f, ari_f, nmi_f))

# # ---------- 分段评估 ----------
# conf_bins = np.percentile(conf_array, np.linspace(0, 100, args.conf_bins + 1))
# bin_metrics = []
# for i in range(args.conf_bins):
#     bin_mask = (conf_array >= conf_bins[i]) & (conf_array <= conf_bins[i + 1])
#     if bin_mask.sum() < 10:
#         continue
#     fused_bin = fused_feats[bin_mask]
#     labels_bin = true_labels[bin_mask]
#     acc, ari, nmi = evaluate_kmeans(fused_bin, labels_bin, n_clusters=num_classes)
#     bin_metrics.append((f"[{conf_bins[i]:.3f}, {conf_bins[i+1]:.3f}]", acc, ari, nmi))

# # ---------- 输出 ----------
# print("\nConfidence Range            ACC      ARI      NMI")
# for conf_range, acc, ari, nmi in bin_metrics:
#     print(f"{conf_range:<25} {acc:.4f}   {ari:.4f}   {nmi:.4f}")

# # 去头去尾
# bin_metrics_plot = bin_metrics[1:-1]

# ranges, accs = zip(*[(r, a) for (r, a, _, _) in bin_metrics])
# plt.plot(range(len(accs)), accs, marker='o')
# plt.xticks(range(len(accs)), ranges, rotation=45)
# plt.xlabel('Confidence Range')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Confidence Segments')
# plt.tight_layout()
# plt.savefig(f'acc_confidence_bins.pdf', bbox_inches='tight')

# ---------- 置信度均匀区间 ----------
bins = np.linspace(0, 1, args.conf_bins + 1)
heatmap = np.zeros((args.conf_bins, num_classes))
print("\n[Uniform Bins] Confidence × Class Accuracy:")
for bin_idx in range(args.conf_bins):
    print(f"Confidence [{bins[bin_idx]:.4f}, {bins[bin_idx + 1]:.4f}]")
    for cls in range(num_classes):
        conf_mask = (conf_array >= bins[bin_idx]) & (conf_array <= bins[bin_idx + 1])
        class_mask = (true_labels == cls)
        mask = conf_mask & class_mask
        total = mask.sum()
        correct = (aligned_preds_f[mask] == true_labels[mask]).sum()
        acc = correct / total if total > 0 else 0.0
        heatmap[bin_idx, cls] = acc
        print(f"  Class {cls:<2}: count = {total:<4}, acc = {acc:.4f}")
    print("-" * 50)

# 构造 annot 数组
annot_arr = np.empty_like(heatmap, dtype=object)
for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        val = heatmap[i, j]
        if val == 0:
            annot_arr[i, j] = "0"
        else:
            annot_arr[i, j] = f"{val:.1f}"

if args.dataset == 'cifar10':
    class_names = datasets.CIFAR10('./data').classes
elif args.dataset == 'stl10':
    class_names = datasets.STL10('./data').classes
else:
    class_names = datasets.CIFAR10('./data').classes
    
np.savez(
    f"confidence_heatmap_data_{args.dataset}.npz",
    heatmap=heatmap,
    annot_arr=annot_arr,
    bins=bins,
    class_names=np.array(class_names)
)

plt.figure(figsize=(12, 6))
cmap = LinearSegmentedColormap.from_list("my_cmap", ["#f7fbff", "#08306b"])
sns.heatmap(
    heatmap,
    annot=annot_arr,
    fmt="",
    cmap=cmap,
    square=True,
    xticklabels=class_names,
    yticklabels=[f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(args.conf_bins)],
    cbar_kws={"shrink": 0.8, "pad": 0.02}
)
plt.ylabel("Confidence Interval")
plt.title("Class-wise Accuracy across Confidence Intervals")
plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.95)  # 适当调整左右边距
plt.savefig("confidence_heatmap_uniform.pdf")
plt.close()