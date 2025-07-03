from builtins import print

from experiments.run_band_selection import run_ssep_pipeline, run_srpa_pipeline, run_srpa_pipeline_get_bands
# from experiments.run_srpa import run_srpa_pipeline
# from experiments.run_cwbi_le import run_cwbi_le_pipeline
from src.bandselectors.ssep import ssep_selection
from src.models.classifiers import evaluate_classifier
from src.utils import load_vnir, load_mask, flatten_data


def run_all_band_selection_methods():
    vnir_path = "data/VNIR.hdr"
    mask_path = "data/segmentation_mask.png"
    k = 20

    print("before load :: ")
    vnir_np = load_vnir(vnir_path)
    mask = load_mask(mask_path)
    print("after load :: ")
    print("vnir_np size :: ", len(vnir_np))
    print("mask size :: ", len(mask))
    results = []

    # 1. SSEP
    # print("\n=== Running SSEP ===")
    # ssep_bands, ssep_acc, ssep_f1 = run_ssep_pipeline(vnir_path, mask_path, k, model_type='svm')
    # results.append(('SSEP', ssep_bands, ssep_acc, ssep_f1))

    # 2. SRPA (Placeholder)
    # print("\n=== Running SRPA ===")
    # srpa_bands, srpa_acc, srpa_f1 = run_srpa_pipeline(vnir_np, mask, k=20, debug_mode=False,
    #                                                   evaluate_with_pixel_classifier=True)
    # results.append(('SRPA', srpa_bands, srpa_acc, srpa_f1))

    # print("\n=== Running SRPA === for get all bands")
    # srpa_bands, srpa_acc, srpa_f1 = run_srpa_pipeline_get_bands(vnir_np, mask, k=273, debug_mode=False)
    # results.append(('SRPA', srpa_bands, srpa_acc, srpa_f1))

    print("\n=== Running SRPA ===")
    srpa_bands, srpa_acc, srpa_f1 = run_srpa_pipeline(vnir_np, mask, k=273, debug_mode=False, evaluate_with_pixel_classifier=True)
    results.append(('SRPA', srpa_bands, srpa_acc, srpa_f1))

    # 3. CWBI-LE (Placeholder)
    # print("\n=== Running CWBI-LE ===")
    # cwbi_bands, cwbi_acc, cwbi_f1 = run_cwbi_le_pipeline(vnir_path, mask_path, k)
    # results.append(('CWBI-LE', cwbi_bands, cwbi_acc, cwbi_f1))

    # Print summary
    # print("\n\n=== Band Selection Comparison ===")
    # for method, bands, acc, f1 in results:
    #     print(f"{method:10} | Accuracy: {acc:.4f} | F1-score: {f1:.4f} | Top Bands: {bands[:5]}...")


if __name__ == "__main__":
    run_all_band_selection_methods()
