import numpy as np
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def compute_mask_edge(mask):
    """
    Compute binary edge map from the labeled mask using Sobel filter.
    """
    mask_float = mask.astype(float)
    edge = sobel(mask_float)
    return (edge > 0).astype(np.uint8)


def compute_band_edge_score(band_img, mask_edge, method='ssim'):
    """
    Compute similarity score between a band edge map and the mask edge map.
    """
    band_smoothed = gaussian_filter(band_img, sigma=1.0)
    band_edge = sobel(band_smoothed)
    band_edge_bin = (band_edge > np.percentile(band_edge, 95)).astype(np.uint8)
    band_edge_bin = np.squeeze(band_edge_bin)
    print("Band edge shape:", band_edge_bin.shape)
    print("Mask edge shape:", mask_edge.shape)
    if method == 'ssim':
        score, _ = ssim(band_edge_bin, mask_edge, full=True)
        return score
    elif method == 'dice':
        intersection = np.sum(band_edge_bin * mask_edge)
        total = np.sum(band_edge_bin) + np.sum(mask_edge)
        if total == 0:
            return 0
        return (2. * intersection) / total
    elif method == 'overlap':
        intersection = np.sum(band_edge_bin * mask_edge)
        union = np.sum(band_edge_bin) + np.sum(mask_edge) - intersection
        if union == 0:
            return 0
        return intersection / union
    else:
        raise ValueError(f"Unsupported method: {method}")


def ssep_selection(vnir_np, mask, k=20, method='ssim'):
    """
    SSEP band selection: selects bands that best preserve edge structure.
    Args:
        vnir_np: (H, W, B) VNIR hyperspectral cube
        mask: (H, W) label mask
        k: number of bands to select
        method: 'ssim' | 'dice' | 'overlap'
    Returns:
        top_k_indices: indices of top-k selected bands
        band_scores: full array of band scores
    """
    H, W, B = vnir_np.shape
    print(f"Running SSEP on cube of shape {vnir_np.shape}...")

    # Compute edge from label mask
    mask_edge = compute_mask_edge(mask)

    band_scores = np.zeros(B)

    for i in tqdm(range(B), desc="Scoring bands (SSEP)"):
        band_img = vnir_np[:, :, i]
        score = compute_band_edge_score(band_img, mask_edge, method=method)
        band_scores[i] = score

    top_k_indices = np.argsort(band_scores)[::-1][:k]
    return top_k_indices, band_scores
