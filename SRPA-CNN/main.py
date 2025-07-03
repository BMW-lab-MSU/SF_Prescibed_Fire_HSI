import numpy as np

from experiments.run_band_selection import run_ssep_pipeline, run_srpa_pipeline
from src.bandselectors.ssep import ssep_selection
from src.models.classifiers import evaluate_classifier
from src.utils import load_vnir, load_mask, flatten_data, extract_patches
from src.trainers.final_cnn_trainer import train_final_3dcnn

# === Settings ===
k = 10
srpa_top_k = [223, 170, 53, 33, 162, 81, 136, 86, 158, 72] # top 10 SRPA bands 3dcnn


def run_final_classifier_with_top_bands(debug_mode=False):
    import os

    project_dir = os.path.dirname(os.path.abspath(__file__))
    vnir_path = os.path.join(project_dir, "data", "VNIR.hdr")
    mask_path = os.path.join(project_dir, "data", "segmentation_mask.png")
    # vnir_path = "./data/VNIR.hdr"
    # mask_path = "./data/segmentation_mask.png"

    vnir_np = load_vnir(vnir_path)
    mask = load_mask(mask_path)

    patches, labels = extract_patches(
        vnir_np, mask,
        patch_size=25, stride=25,
        min_class_ratio=0.6,
        selected_bands=srpa_top_k,
        balance_classes=False,
        max_per_class=None  # if debug_mode else None
    )
    print(f"\n✅ Patches shape: {patches.shape}")  # (N, 50, 50, K)
    print(f"✅ Labels shape: {labels.shape}")  # (N,)
    print(f"✅ Unique labels: {np.unique(labels, return_counts=True)}")
    model, acc, f1 = train_final_3dcnn(
        patches, labels,
        num_classes=3,
        epochs=20,
        batch_size=16,
        save_path="outputs/models/final_3dcnn_srpa_classifier.pth"
    )

    return model, acc, f1


def run_fullband_classifier(debug_mode=False):
    from src.trainers.final_cnn_trainer import train_final_3dcnn
    from src.utils import load_vnir, load_mask, extract_patches
    import os

    project_dir = os.path.dirname(os.path.abspath(__file__))
    vnir_path = os.path.join(project_dir, "data", "VNIR.hdr")
    mask_path = os.path.join(project_dir, "data", "segmentation_mask.png")

    # vnir_path = "data/VNIR.hdr"
    # mask_path = "data/segmentation_mask.png"

    vnir_np = load_vnir(vnir_path)
    mask = load_mask(mask_path)

    patches, labels = extract_patches(
        vnir_np, mask,
        patch_size=50, stride=25,
        min_class_ratio=0.6,
        selected_bands=None,  # ✅ Use all 273 bands
        balance_classes=True,
        max_per_class=500 if debug_mode else None
    )

    print(
        f"\n✅ Full Band Patches: {patches.shape}, Labels: {labels.shape}, Unique: {np.unique(labels, return_counts=True)}")

    model, acc, f1 = train_final_3dcnn(
        patches, labels,
        num_classes=3,
        epochs=20,
        batch_size=16,
        save_path="outputs/models/final_3dcnn_fullband_classifier.pth"
    )

    return model, acc, f1


if __name__ == "__main__":
    print("\n=== Running Final 3D CNN Classifier using SRPA Bands ===")
    model, acc, f1 = run_final_classifier_with_top_bands(debug_mode=False)
    print(f"\n✅ Final 3D CNN (SRPA bands) → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # print("\n=== Running Final 3D CNN Classifier using All 273 Bands ===")
    # model_full, acc_full, f1_full = run_fullband_classifier(debug_mode=False)
    #
    # print(f"📊 Full Bands → Acc: {acc_full:.4f}, F1: {f1_full:.4f}")
