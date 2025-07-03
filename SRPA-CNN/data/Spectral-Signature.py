# import numpy as np
# import pandas as pd
# from spectral import open_image
# from PIL import Image
# import os
#
# # === Load VNIR Cube and Segmentation Mask ===
# hdr_path = "VNIR.hdr"
# vnir = open_image(hdr_path).load()  # shape: (H, W, 273)
# mask = np.array(Image.open("segmentation_mask.png"))  # shape: (H, W)
#
# # === Load wavelengths from .hdr ===
# wavelengths = np.array([float(w) for w in open_image(hdr_path).metadata['wavelength']])
#
# # === Collect 10 pixels per class (Tree=1, Grass=2, Soil=0) ===
# class_names = {0: "Soil", 1: "Tree", 2: "Grass"}
# spectra = []
#
# for class_id, label in class_names.items():
#     coords = np.argwhere(mask == class_id)
#     selected_coords = coords[:10] if len(coords) >= 10 else coords
#     for idx, (r, c) in enumerate(selected_coords):
#         spectrum = vnir[r, c, :]
#         row = {
#             "Class": label,
#             "Pixel_Index": f"{label}_{idx}"
#         }
#         for i, wl in enumerate(wavelengths):
#             row[f"Band_{int(wl)}nm"] = spectrum[i]
#         spectra.append(row)
#
# # === Save to CSV ===
# df = pd.DataFrame(spectra)
# os.makedirs("outputs/reflectance", exist_ok=True)
# df.to_csv("outputs/reflectance/pixel_spectral_signatures.csv", index=False)
# print("✅ Reflectance CSV saved to outputs/reflectance/pixel_spectral_signatures.csv")


import numpy as np
import pandas as pd
from spectral import open_image
from PIL import Image

# === Load Data ===
hdr_path = "VNIR.hdr"
vnir = open_image(hdr_path).load()
mask = np.array(Image.open("segmentation_mask.png"))
wavelengths = [float(w) for w in open_image(hdr_path).metadata['wavelength']]

# === Extract 10 pixels per class ===
class_ids = {0: "Soil", 1: "Tree", 2: "Grass"}
data_rows = []

for class_id, class_name in class_ids.items():
    coords = np.argwhere(mask == class_id)
    selected_coords = coords[:10]
    for i, (r, c) in enumerate(selected_coords):
        spectrum = vnir[r, c, :]
        row = {
            "Class": class_name,
            "Pixel_Index": f"{class_name}_{i}",
        }
        for j, wl in enumerate(wavelengths):
            row[f"Band_{int(wl)}nm"] = spectrum[j]
        data_rows.append(row)

# === Save Clean CSV ===
df = pd.DataFrame(data_rows)
df.to_csv("outputs/reflectance/pixel_spectral_signatures.csv", index=False)
print("✅ Cleaned CSV written!")
