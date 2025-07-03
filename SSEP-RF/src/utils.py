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


def save_band_scores_to_csv(selected_bands, band_scores, save_path):
    """
    Save the selected bands, wavelengths, and band scores to a CSV file.
    """
    hdr_path = "data/VNIR.hdr"
    wavelengths = np.array([float(w) for w in open_image(hdr_path).metadata['wavelength']])

    df = pd.DataFrame({
        'Band_Index': selected_bands,  # Band indices sorted best to worst
        'Wavelength_nm': wavelengths[selected_bands],  # Wavelength for each band
        'Dice_Score': band_scores[selected_bands]  # Dice score for each band
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved full band scores CSV at: {save_path}")
