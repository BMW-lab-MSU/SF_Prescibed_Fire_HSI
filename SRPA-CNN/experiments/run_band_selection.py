import numpy as np
from models.classifiers import evaluate_classifier
from bandselectors.ssep import ssep_selection
from utils import load_vnir, load_mask, flatten_data


def run_ssep_pipeline(vnir_path, mask_path, k=20, model_type='svm'):
    print("Loading data...")
    vnir = load_vnir(vnir_path)  # shape: (H, W, B)
    mask = load_mask(mask_path)  # shape: (H, W)
    vnir_np, mask_flat = flatten_data(vnir, mask)

    print("Running SSEP selection...")
    selected_bands, _ = ssep_selection(vnir, mask, k=k, method='overlap')
    print("Top-k bands:", selected_bands)

    # Extract selected features from flattened data
    X_selected = vnir_np[:, selected_bands]
    y_labeled = mask_flat

    # Filter out background (0)
    valid_idx = np.where(y_labeled > 0)[0]
    X_valid = X_selected[valid_idx]
    y_valid = y_labeled[valid_idx]

    print("Evaluating classifier...")
    acc, f1 = evaluate_classifier(X_valid, y_valid, model_type=model_type, use_subset=True)
    print(f"[SSEP] Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    return selected_bands, acc, f1


# # Example usage
# if __name__ == "__main__":
#     vnir_path = "data/VNIR.hdr"
#     mask_path = "data/segmentation_mask.png"
#     run_ssep_pipeline(vnir_path, mask_path, k=20, model_type='svm')


from src.bandselectors.srpa import srpa_selection
from src.utils import extract_patches
from src.models.classifiers import evaluate_classifier


def run_srpa_pipeline(vnir_np, mask, k=20, debug_mode=True, evaluate_with_pixel_classifier=False):
    patches, labels = extract_patches(
        vnir_np, mask,
        patch_size=25,
        stride=50,
        min_class_ratio=0.6,
        balance_classes=True,  # balance_classes=debug_mode, when debug_mode is true
        # max_per_class=500 if debug_mode else None # when debug_mode is true
        max_per_class=None  # full train
    )
    print(f"✅ Patches: {patches.shape}, Labels: {np.unique(labels, return_counts=True)}")
    # top_k, scores = srpa_selection(patches, labels, num_classes=3, k=k)

    # top_k = [80, 136, 114, 164, 121, 41, 94, 64, 18, 244, 210, 71, 23, 30, 223, 131, 176, 122, 199, 69] # this top k=20 from the SRPA
    # print("🎯 SRPA top bands:", top_k)
    # print("🎯 SRPA top scores:", scores)

    top_k = [223, 170, 53, 33, 162, 81, 136, 86, 158, 72, 149, 259, 131, 55, 183, 238, 272, 31, 221, 225, 32, 271, 260, 156, 126, 76, 155, 182, 105, 255, 130, 243, 172, 135, 205, 89,18, 154, 16, 160, 50, 157, 13, 269, 78, 186, 80, 249, 113, 132]
    print("🎯 SSrP top bands size :: ", len(top_k))
    print("🎯 SSrP top bands:", top_k)
    if evaluate_with_pixel_classifier:
        print("🎯 Insider evaluate_with_pixel_classifier:", evaluate_with_pixel_classifier)
        # Optional classifier evaluation
        H, W, B = vnir_np.shape
        vnir_flat = vnir_np.reshape(-1, B)
        mask_flat = mask.reshape(-1)
        valid_idx = mask_flat > 0

        X_valid = vnir_flat[valid_idx][:, top_k]
        y_valid = mask_flat[valid_idx]

        acc, f1 = evaluate_classifier(X_valid, y_valid, model_type='rf')
        print("🎯 SSrP rf acc:", acc)
        print("🎯 SSrP rf f1:", f1)
        return top_k, acc, f1
    else:
        print("🎯 ELSE Insider evaluate_with_pixel_classifier:", evaluate_with_pixel_classifier)
        return top_k, None, None


#    os.makedirs("outputs/band_scores", exist_ok=True)
#  np.save("outputs/band_scores/srpa_top_k.npy", top_k)
#  np.save("outputs/band_scores/srpa_scores.npy", scores)


def run_srpa_pipeline_get_bands(vnir_np, mask, k=273, debug_mode=True, ):
    patches, labels = extract_patches(
        vnir_np, mask,
        patch_size=25,
        stride=25,
        min_class_ratio=0.6,
        balance_classes=True,
        # balance_classes=debug_mode, when debug_mode is true, balance_classes=False for full traini
        max_per_class=2000  # if debug_mode else None # when debug_mode is true
        # max_per_class=None  # full train
    )
    print(f"✅ Patches: {patches.shape}, Labels: {np.unique(labels, return_counts=True)}")
    selected_bands, band_scores = srpa_selection(patches, labels, num_classes=3, k=k)

    print(len(selected_bands))
    print(len(band_scores))

    # Print Top-K Bands
    print("\n✅ Top 10 bands:", selected_bands[:10])
    print("✅ Top 20 bands:", selected_bands[:20])
    print("✅ Top 30 bands:", selected_bands[:30])
    print("✅ Top 40 bands:", selected_bands[:40])
    print("✅ Top 50 bands:", selected_bands[:50])

    from src.utils import save_band_scores_to_csv

    # Save full sorted SRPA scores into CSV
    # selected_bands = np.argsort(scores)[::-1]  # Sort all bands, not just top-k
    save_band_scores_to_csv(
        selected_bands,
        band_scores,
        save_path="outputs/band_scores/srpa_full_band_scores_sorted.csv"
    )

    # Optional: print Top-k bands
    print("\n✅ Top 10 bands (SRPA):", selected_bands[:10])
    print("✅ Top 20 bands (SRPA):", selected_bands[:20])
    print("✅ Top 30 bands (SRPA):", selected_bands[:30])
    print("✅ Top 40 bands (SRPA):", selected_bands[:40])
    print("✅ Top 50 bands (SRPA):", selected_bands[:50])

    return selected_bands, band_scores, None
