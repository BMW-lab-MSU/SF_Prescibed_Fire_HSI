import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# === Internal Dataset ===
class SRPADataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx]).float().unsqueeze(0)
        y = torch.tensor(self.labels[idx]).long()
        return x, y


# === SE Block ===
class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ✅ No pooling
        x = torch.sigmoid(self.fc2(x))  # Output attention weights (0–1)
        return x


# === Lightweight 3D CNN with SE ===
class SRPA3DCNN(nn.Module):
    def __init__(self, num_bands, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, num_bands))
        self.se = SEBlock(num_bands)
        self.fc = nn.Linear(16 * num_bands, num_classes)

    def forward(self, x):  # x: (B, 1, H, W, B)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 8, H/2, W/2, B)
        x = F.relu(self.conv2(x))  # (B, 16, H/2, W/2, B)

        # Global pooling to get feature map of shape (B, 16, 1, 1, B)
        pooled = self.global_pool(x)  # (B, 16, 1, 1, B)
        pooled = pooled.squeeze(2).squeeze(2)  # (B, 16, B)

        # ✅ Correct attention input: mean over channels → shape: (B, B)
        attn_input = pooled.mean(dim=1)  # (B, B)
        attn_weights = self.se(attn_input)  # (B, B)

        # Flatten for final classification
        feat = pooled.permute(0, 2, 1)  # (B, B, 16)
        feat_flat = feat.reshape(feat.size(0), -1)  # (B, B*16)

        return self.fc(feat_flat), attn_weights


# === Redundancy Matrix ===
def compute_redundancy_matrix(patches):
    """
    Computes spectral correlation between bands across all patches
    Returns a vector of average correlation of each band with others
    """
    N, H, W, B = patches.shape
    X = patches.reshape(-1, B)  # shape: (N*H*W, B)

    # Downsample to avoid memory overflow (e.g., 100k random pixels)
    if X.shape[0] > 100_000:
        idx = np.random.choice(X.shape[0], 100_000, replace=False)
        X = X[idx]

    # Compute band-wise correlation matrix (273 × 273)
    corr = np.corrcoef(X, rowvar=False)

    # Compute mean redundancy score for each band (exclude self-correlation)
    redundancy = (np.sum(np.abs(corr), axis=1) - 1) / (B - 1)
    return redundancy


# === SRPA Band Selector ===
def srpa_selection(patches, labels, num_classes=3, k=273, batch_size=16, lambda_penalty=0.3):
    print("k value in SRPA Selection :: ", k)
    num_bands = patches.shape[-1]
    model = SRPA3DCNN(num_bands, num_classes)
    loader = DataLoader(SRPADataset(patches, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for 2 quick epochs
    model.train()
    for _ in range(2):
        for x, y in loader:
            out, _ = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Collect attention weights
    model.eval()
    attn_scores = []
    with torch.no_grad():
        for x, _ in loader:
            _, attn = model(x)
            attn_scores.append(attn.mean(dim=0).cpu().numpy())
    attn_mean = np.mean(np.vstack(attn_scores), axis=0)

    # Penalize redundancy
    redundancy = compute_redundancy_matrix(patches[:, :, :, :])
    penalty = redundancy.mean(axis=0)
    srpa_scores = attn_mean - lambda_penalty * penalty

    top_k = np.argsort(srpa_scores)[::-1][:k]
    return top_k, srpa_scores
