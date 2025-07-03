import numpy as np
import os
from spectral import open_image
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

# === Load VNIR and Mask ===
hdr_path = "VNIR.hdr"
vnir = open_image(hdr_path).load()  # (H, W, B)
mask = np.array(Image.open("segmentation_mask.png"))


# === Dice Score Function ===
def compute_dice_score(band_img, mask_edge):
    band_smoothed = gaussian_filter(band_img, sigma=1.0)
    band_edge = sobel(band_smoothed)
    band_edge_bin = (band_edge > np.percentile(band_edge, 95)).astype(np.uint8)
    band_edge_bin = np.squeeze(band_edge_bin)

    mask_edge_bin = (mask_edge > np.percentile(mask_edge, 95)).astype(np.uint8)

    intersection = np.sum(band_edge_bin * mask_edge_bin)
    total = np.sum(band_edge_bin) + np.sum(mask_edge_bin)
    return (2. * intersection / total) if total > 0 else 0.0


# === Create mask edge ===
mask_binary = (mask > 0).astype(np.uint8)
mask_edge = sobel(mask_binary)

# === Score all bands ===
H, W, B = vnir.shape
ssep_scores = []

for b in range(B):
    band_img = vnir[:, :, b]
    score = compute_dice_score(band_img, mask_edge)
    ssep_scores.append(score)

ssep_scores = np.array(ssep_scores)

# === Save scores ===
os.makedirs("outputs/band_scores", exist_ok=True)
np.save("outputs/band_scores/ssep_scores.npy", ssep_scores)

print("✅ SSEP Dice scores saved to: outputs/band_scores/ssep_scores.npy")
