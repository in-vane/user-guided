import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from open_clip import create_model_and_transforms
import argparse


# ---------- 配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10'], help='选择数据集')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
args = parser.parse_args()

dataset = datasets.LSUN(root="./data", split='train', download=True)

# ---------- 加载 OpenCLIP 模型 ----------
model, preprocess, _ = create_model_and_transforms("ViT-B-32-quickgelu", pretrained="metaclip_fullcc")
model.to(device).eval()

# ---------- LoRA 层 ----------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return (self.B @ (self.A @ x.T)).T * self.scaling

# ---------- 数据集 ----------
transform = preprocess
if args.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
elif args.dataset == 'stl10':
    dataset = datasets.STL10(root="./data", split='train', download=True, transform=transform)
    testset = datasets.STL10(root="./data", split='test', download=True, transform=transform)
else:
    raise ValueError("不支持的数据集")

data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(testset, batch_size=256, shuffle=False)

# ---------- LoRA 模块与优化器 ----------
in_dim = model.visual.output_dim  # 512
lora = LoRALayer(in_dim, in_dim).to(device)
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-4)

# ---------- 聚类评估函数 ----------
def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

def evaluate_clustering(features, labels):
    kmeans = KMeans(n_clusters=10, random_state=42).fit(features)
    preds = kmeans.labels_
    acc = cluster_accuracy(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds)
    return acc, ari, nmi

# ---------- 提取特征并评估聚类 ----------
def run_evaluation(loader=data_loader):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            feats = F.normalize(feats + lora(feats), dim=-1)
            all_feats.append(feats.cpu())
            all_labels.extend(labels.numpy())
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = np.array(all_labels)
    acc, ari, nmi = evaluate_clustering(all_feats, all_labels)
    print(f"\n📊 Clustering Result | ACC: {acc:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")

# ---------- 检查是否已存在权重 ----------
# checkpoint_path = "checkpoints/lora_openclip.pth"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"lora_openclip_{args.dataset}.pth")

if os.path.exists(checkpoint_path):
    lora.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✅ 已加载 LoRA 权重，进行聚类评估...")
    run_evaluation(test_loader)
else:
    # ---------- 训练 ----------
    run_evaluation(test_loader)
    for epoch in range(args.epochs):
        model.eval()
        total_loss = 0.
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                feats = model.encode_image(images)
                feats = F.normalize(feats, dim=-1)

            feats_lora = feats + lora(feats)
            feats_lora = F.normalize(feats_lora, dim=-1)

            sim_matrix = feats_lora @ feats_lora.T
            labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
            labels_matrix = labels_matrix.float().to(device)
            logits = sim_matrix / 0.07
            loss = F.cross_entropy(logits, labels_matrix.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # 每轮训练后提取特征进行聚类测试
        run_evaluation(test_loader)

    # ---------- 保存 ----------
    torch.save(lora.state_dict(), checkpoint_path)
    print(f"✅ LoRA weights saved to {checkpoint_path}")
