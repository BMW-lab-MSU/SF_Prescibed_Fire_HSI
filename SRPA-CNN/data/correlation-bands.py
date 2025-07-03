import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spectral import open_image

# === Load VNIR Cube ===
hdr_path = "VNIR.hdr"
vnir = open_image(hdr_path).load()  # shape: (H, W, B)

# === Preprocess ===
H, W, B = vnir.shape
vnir_flat = vnir.reshape(-1, B)

# Remove all-zero pixels
valid_pixels = ~np.all(vnir_flat == 0, axis=1)
vnir_valid = vnir_flat[valid_pixels]

# === Sample for performance ===
np.random.seed(42)
sampled = vnir_valid[np.random.choice(vnir_valid.shape[0], size=10000, replace=False)]

# === Compute Correlation Matrix ===
corr_matrix = np.corrcoef(sampled.T)

# === Plot Heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False,
            square=True, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap of All 273 VNIR Bands")
plt.tight_layout()
plt.show()
