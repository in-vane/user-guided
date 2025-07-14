import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
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
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10', 'ppmi', 'mnist', 'oxford_pet'], help='选择数据集')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
args = parser.parse_args()

# ---------- 加载 OpenCLIP 模型 ----------
model, preprocess, _ = create_model_and_transforms("ViT-B-32-quickgelu", pretrained="metaclip_fullcc")
model.to(device).eval()

# ---------- 优化版 ARD-LoRA 定义 ----------
class ARDLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        # self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, rank))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        self.log_lambda = nn.Parameter(torch.zeros(out_dim))

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

# ---------- 数据集 ----------
transform = preprocess

if args.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    num_classes = 10
elif args.dataset == 'stl10':
    dataset = datasets.STL10(root="./data", split='train', download=True, transform=transform)
    testset = datasets.STL10(root="./data", split='test', download=True, transform=transform)
    num_classes = 10
elif args.dataset == 'mnist':
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    num_classes = 10
elif args.dataset == 'oxford_pet':
    dataset = datasets.OxfordIIITPet(root="./data", split='trainval', download=True, transform=transform)
    testset = datasets.OxfordIIITPet(root="./data", split='test', download=True, transform=transform)
    num_classes = 37
elif args.dataset == 'ppmi':
    full_dataset = ImageFolder(root="./data/ppmi/play_instrument", transform=transform)
    train_samples = [(p, l) for p, l in full_dataset.samples if "/train/" in p.replace("\\", "/")]
    test_samples = [(p, l) for p, l in full_dataset.samples if "/test/" in p.replace("\\", "/")]
    dataset = torch.utils.data.Subset(full_dataset, indices=[i for i, (p, _) in enumerate(full_dataset.samples) if "/train/" in p.replace("\\", "/")])
    testset = torch.utils.data.Subset(full_dataset, indices=[i for i, (p, _) in enumerate(full_dataset.samples) if "/test/" in p.replace("\\", "/")])
    dataset.classes = full_dataset.classes
    testset.classes = full_dataset.classes
    print(f"PPMI classes: {full_dataset.classes}")
    num_classes = len(full_dataset.classes)
    print(f"PPMI dataset loaded with {num_classes} classes.")
else:
    raise ValueError("不支持的数据集")

data_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

# ---------- LoRA 初始化与优化器 ----------
in_dim = model.visual.output_dim  # 512
lora = ARDLoRALayer(in_dim, in_dim, rank=4).to(device)
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-4)

# ---------- 聚类评估函数 ----------
def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

def evaluate_clustering(features, labels, num_classes):
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(features)
    preds = kmeans.labels_
    acc = cluster_accuracy(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds)
    return acc, ari, nmi

# ---------- 提取特征并评估聚类 ----------
def run_evaluation(loader=test_loader, num_classes=10):
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
    acc, ari, nmi = evaluate_clustering(all_feats, all_labels, num_classes=num_classes)
    print(f"\n\U0001F4CA Clustering Result | ACC: {acc:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")

# ---------- 检查是否已存在权重 ----------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"lora_openclip_ard_{args.dataset}.pth")

if os.path.exists(checkpoint_path):
    lora.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✅ 已加载 LoRA 权重，进行聚类评估...")
    run_evaluation(test_loader, num_classes)
else:
    run_evaluation(test_loader, num_classes)

    # ---------- 训练 ----------
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

            kl_coeff = min(1e-4 * (epoch + 1) / args.epochs, 1e-4)
            loss += kl_coeff * lora.kl_regularization()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "kl_coeff": kl_coeff})

        run_evaluation(test_loader, num_classes)

    # ---------- 保存 ----------
    torch.save(lora.state_dict(), checkpoint_path)
    print(f"✅ LoRA weights saved to {checkpoint_path}")

# ---------- 可视化 ARD 权重 ----------
# def plot_ard_importance(lora_layer, title="ARD Importance", save_path="ard_importance_v2.pdf"):
#     log_lambda = lora_layer.log_lambda.detach().cpu()
#     importance = torch.exp(-log_lambda)
#     plt.figure(figsize=(8, 4))
#     x = torch.arange(len(importance))
#     plt.bar(x, importance.numpy(), color="teal")
#     plt.xlabel("Output Channel Index")
#     plt.ylabel("Importance ($\\lambda_j^{-1}$)")
#     plt.title(title)
#     plt.grid(True, linestyle="--", alpha=0.4)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# plot_ard_importance(lora, title="ARD Importance (Optimized)", save_path="ard_importance_v2.pdf")
