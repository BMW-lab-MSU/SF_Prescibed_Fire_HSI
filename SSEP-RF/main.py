from experiments.run_band_selection import run_ssep_pipeline, run_ssep_pipeline_getBands
# from experiments.run_srpa import run_srpa_pipeline
# from experiments.run_cwbi_le import run_cwbi_le_pipeline
from src.bandselectors.ssep import ssep_selection
from src.models.classifiers import evaluate_classifier
from src.utils import load_vnir, load_mask, flatten_data


def run_all_band_selection_methods():
    vnir_path = "data/VNIR.hdr"
    mask_path = "data/segmentation_mask.png"
    k = 20

    results = []

    # 1. SSEP
    print("\n=== Running SSEP ===")
    ssep_bands, ssep_acc, ssep_f1 = run_ssep_pipeline(vnir_path, mask_path, k, model_type='svm')
    results.append(('SSEP', ssep_bands, ssep_acc, ssep_f1))

    # 2. SRPA (Placeholder)
    # print("\n=== Running SRPA ===")
    # srpa_bands, srpa_acc, srpa_f1 = run_srpa_pipeline(vnir_path, mask_path, k)
    # results.append(('SRPA', srpa_bands, srpa_acc, srpa_f1))

    # 3. CWBI-LE (Placeholder)
    # print("\n=== Running CWBI-LE ===")
    # cwbi_bands, cwbi_acc, cwbi_f1 = run_cwbi_le_pipeline(vnir_path, mask_path, k)
    # results.append(('CWBI-LE', cwbi_bands, cwbi_acc, cwbi_f1))

    # Print summary
    print("\n\n=== Band Selection Comparison ===")
    for method, bands, acc, f1 in results:
        print(f"{method:10} | Accuracy: {acc:.4f} | F1-score: {f1:.4f} | Top Bands: {bands[:5]}...")


def check_size():
    vnir_path = "data/VNIR.hdr"
    mask_path = "data/segmentation_mask.png"
    print("Loading data...check_size.....")
    vnir = load_vnir(vnir_path)  # shape: (H, W, B)
    mask = load_mask(mask_path)  # shape: (H, W)
    vnir_np, mask_flat = flatten_data(vnir, mask)

    print("VNIR shape:", vnir.shape)
    print("Mask shape:", mask.shape)


def run_all_band_selection_methods_getBands():
    vnir_path = "data/VNIR.hdr"
    mask_path = "data/segmentation_mask.png"
    k = 273

    print("\n=== Running SSEP Only bands===")
    ssep_bands = run_ssep_pipeline_getBands(vnir_path, mask_path, k)

    print("main:", ssep_bands)


if __name__ == "__main__":
    run_all_band_selection_methods()
    # check_size()
    # run_all_band_selection_methods_getBands()
