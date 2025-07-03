import numpy as np
from models.classifiers import evaluate_classifier
from bandselectors.ssep import ssep_selection
from utils import load_vnir, load_mask, flatten_data, save_band_scores_to_csv


def run_ssep_pipeline(vnir_path, mask_path, k=20, model_type='svm'):
    print("Loading data...")
    vnir = load_vnir(vnir_path)  # shape: (H, W, B)
    mask = load_mask(mask_path)  # shape: (H, W)
    vnir_np, mask_flat = flatten_data(vnir, mask)

    print("Running SSEP selection...")
    # selected_bands, _ = ssep_selection(vnir, mask, k=k, method='overlap')
    # selected_bands = [243, 246, 257, 241, 251, 247, 249, 253, 266, 239, 237, 242, 269, 245, 265, 240, 250, 272,248, 233] # top 20
    # selected_bands = [243, 246, 257, 241, 251, 247, 249, 253, 266, 239, 237, 242, 269, 245, 265, 240, 250, 272, 248,233]

    selected_bands = [243, 246, 257, 241, 251, 247, 249, 253, 266, 239, 237, 242, 269, 245, 265, 240, 250, 272, 248, 233, 262, 260, 244, 255, 258, 235, 261, 254, 270, 229, 268, 263, 231, 234, 252, 238, 232, 236, 259, 256, 227, 267, 271, 225, 228, 264, 230, 226, 223, 224]
    print("size :: ", len(selected_bands))

    print("Top-K bands:", selected_bands)

    # Extract selected features from flattened data
    X_selected = vnir_np[:, selected_bands]
    y_labeled = mask_flat

    # Filter out background (0)
    valid_idx = np.where(y_labeled > 0)[0]
    X_valid = X_selected[valid_idx]
    y_valid = y_labeled[valid_idx]

    print("Evaluating classifier...")
    acc, f1 = evaluate_classifier(X_valid, y_valid, model_type=model_type, debug_mode=False)
    print(f"[SSEP] Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    return selected_bands, acc, f1


def run_ssep_pipeline_getBands(vnir_path, mask_path, k=273):
    print("Loading data...")
    vnir = load_vnir(vnir_path)  # shape: (H, W, B)
    mask = load_mask(mask_path)  # shape: (H, W)
    vnir_np, mask_flat = flatten_data(vnir, mask)

    print("Running SSEP selection...")
    print("K values::: ", k)
    selected_bands, band_scores = ssep_selection(vnir, mask, k=273, method='dice')

    print(len(selected_bands))
    print(len(band_scores))

    # Print Top-K Bands
    print("\n✅ Top 10 bands:", selected_bands[:10])
    print("✅ Top 20 bands:", selected_bands[:20])
    print("✅ Top 30 bands:", selected_bands[:30])
    print("✅ Top 40 bands:", selected_bands[:40])
    print("✅ Top 50 bands:", selected_bands[:50])

    save_band_scores_to_csv(
        selected_bands,
        band_scores,
        save_path="outputs/band_scores/ssep_full_band_scores_sorted.csv"
    )


# # Example usage
# if __name__ == "__main__":
#     vnir_path = "data/VNIR.hdr"
#     mask_path = "data/segmentation_mask.png"
#     run_ssep_pipeline(vnir_path, mask_path, k=20, model_type='svm')
