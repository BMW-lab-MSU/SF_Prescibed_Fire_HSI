import numpy as np
import os
from spectral import open_image
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Load VNIR Cube and Mask ===
hdr_path = "VNIR.hdr"
vnir = open_image(hdr_path).load()  # (H, W, B)
mask = np.array(Image.open("segmentation_mask.png"))


# === 3D Patch Extraction ===
def extract_patches(vnir_cube, mask, patch_size=50, stride=25, min_class_ratio=0.6, max_per_class=500):
    H, W, B = vnir_cube.shape
    patches, labels = [], []
    class_count = {1: 0, 2: 0}
    for i in range(0, H - patch_size, stride):
        for j in range(0, W - patch_size, stride):
            patch = vnir_cube[i:i + patch_size, j:j + patch_size, :]
            patch_mask = mask[i:i + patch_size, j:j + patch_size]
            unique, counts = np.unique(patch_mask, return_counts=True)
            dominant = unique[np.argmax(counts)]
            if dominant == 0:  # skip background/soil
                continue
            ratio = np.max(counts) / (patch_size * patch_size)
            if ratio >= min_class_ratio and class_count[dominant] < max_per_class:
                patches.append(patch)
                labels.append(dominant - 1)  # tree -> 0, grass -> 1
                class_count[dominant] += 1
    return np.stack(patches), np.array(labels)


# === Simple 3D CNN with Attention Head ===
class SRPANet(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, bands))
        self.se_fc = nn.Sequential(
            nn.Linear(bands, bands // 4),
            nn.ReLU(),
            nn.Linear(bands // 4, bands),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (batch_size, bands)
        attention = self.se_fc(x)  # (batch_size, bands)
        return attention


# === Main SRPA Score Generation ===
patches, labels = extract_patches(vnir, mask, patch_size=50, stride=25)
patches = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W, B)
labels = torch.tensor(labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRPANet(bands=patches.shape[-1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# === Train small 3D CNN (attention will learn band importance) ===
model.train()
for epoch in range(5):  # fast debug mode
    idx = torch.randperm(patches.size(0))
    xb, yb = patches[idx], labels[idx]
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    attn = model(xb)
    loss = criterion(attn, yb)
    loss.backward()
    optimizer.step()

# === Extract Attention Scores ===
model.eval()
with torch.no_grad():
    attention_scores = model(patches.to(device))
    attention_scores = attention_scores.mean(dim=0).cpu().numpy()  # average across patches

# === Save SRPA Attention Scores ===
os.makedirs("outputs/band_scores", exist_ok=True)
np.save("outputs/band_scores/srpa_scores.npy", attention_scores)

print("✅ SRPA attention scores saved to outputs/band_scores/srpa_scores.npy")
