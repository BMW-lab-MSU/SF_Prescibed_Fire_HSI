from spectral import open_image
from PIL import Image
import numpy as np
import pandas as pd
import os


def load_vnir(hdr_path):
    return open_image(hdr_path).load()


def load_mask(mask_path):
    return np.array(Image.open(mask_path))


def flatten_data(vnir, mask):
    vnir_np = np.array(vnir)  # (H, W, B)
    H, W, B = vnir_np.shape
    vnir_flat = vnir_np.reshape(-1, B)
    mask_flat = mask.reshape(-1)
    return vnir_flat, mask_flat


def extract_patches(vnir_cube, mask, patch_size=25, stride=25, min_class_ratio=0.6, balance_classes=False,
                    max_per_class=None, selected_bands=None):
    print("patch_size", patch_size)
    print("stride", stride)
    print("balance_classes", balance_classes)
    print("max_per_class", max_per_class)
    import numpy as np
    from collections import defaultdict

    H, W, B = vnir_cube.shape
    patches, labels = [], []
    class_patch_dict = defaultdict(list)
    print("are we here :: ")
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if selected_bands is not None:
                patch = vnir_cube[i:i + patch_size, j:j + patch_size, selected_bands]
            else:
                patch = vnir_cube[i:i + patch_size, j:j + patch_size, :]

            patch_mask = mask[i:i + patch_size, j:j + patch_size]
            classes, counts = np.unique(patch_mask, return_counts=True)
            dominant_class = classes[np.argmax(counts)]
            # if dominant_class == 0 or np.max(counts) / patch_mask.size < min_class_ratio:
            if np.max(counts) / patch_mask.size < min_class_ratio:
                continue
            if balance_classes:
                class_patch_dict[dominant_class].append((patch, dominant_class))
            else:
                patches.append(patch)
                labels.append(dominant_class)

    print("are we end here :: ")
    if balance_classes:
        for cls, patch_list in class_patch_dict.items():
            selected = patch_list[:max_per_class] if max_per_class else patch_list
            for patch, lbl in selected:
                patches.append(patch)
                labels.append(lbl)

    return np.stack(patches), np.array(labels)


def save_band_scores_to_csv(selected_bands, band_scores, save_path):
    """
    Save the selected bands, wavelengths, and band scores to a CSV file.
    """
    hdr_path = "data/VNIR.hdr"
    wavelengths = np.array([float(w) for w in open_image(hdr_path).metadata['wavelength']])

    df = pd.DataFrame({
        'Band_Index': selected_bands,  # Band indices sorted best to worst
        'Wavelength_nm': wavelengths[selected_bands],  # Wavelength for each band
        'Attn_Score': band_scores[selected_bands]  # Attn score for each band
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved full band scores CSV at: {save_path}")
