import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from spectral import open_image

# === Load VNIR Cube ===
hdr_path = "VNIR.hdr"
vnir = open_image(hdr_path).load()  # shape: (H, W, B)
H, W, B = vnir.shape

# === Preprocessing ===
vnir_flat = vnir.reshape(-1, B)
valid_idx = ~np.all(vnir_flat == 0, axis=1)
vnir_valid = vnir_flat[valid_idx]

# === Each band as a vector (273 bands × N pixels)
band_vectors = vnir_valid.T  # shape: (273, N)

# === t-SNE Projection of Bands ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
band_tsne = tsne.fit_transform(band_vectors)

# === Build NetworkX Graph with High-Correlation Edges ===
corr_matrix = np.corrcoef(band_vectors)
threshold = 0.95

G = nx.Graph()
for i in range(B):
    G.add_node(i, pos=band_tsne[i])

for i in range(B):
    for j in range(i + 1, B):
        if corr_matrix[i, j] > threshold:
            G.add_edge(i, j)

# === Plot t-SNE with Correlation Graph ===
plt.figure(figsize=(12, 10))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, node_color='skyblue', edge_color='gray', with_labels=True, node_size=50, font_size=6)
plt.title("t-SNE Projection of VNIR Bands with Correlation Edges (threshold > 0.95)")
plt.tight_layout()
plt.show()