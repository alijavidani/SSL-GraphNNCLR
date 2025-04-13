import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    # If umap is not installed, the user won't be able to use the "umap" method

def compare_tsne_enhancement(
    before_embeddings,
    after_embeddings,
    labels,
    method="umap",
    subset_size=10000,
    random_seed=42,
    perplexity=30,
    out_file="comparison_2d.png"
):
    """
    Visualize the effect of enhancement (e.g., SelfGNN) using t-SNE or UMAP in a side-by-side comparison.

    :param before_embeddings: torch.Tensor of shape [N, D], before enhancement
    :param after_embeddings: torch.Tensor of shape [N, D], after enhancement
    :param labels: torch.Tensor of shape [N], class labels
    """
    before_np = before_embeddings#.cpu().numpy()
    after_np = after_embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    print(f"Projecting BEFORE enhancement ({method})...")
    before_2d, indices_b = project_embeddings_2d(
        before_np,
        method=method,
        subset_size=subset_size,
        random_seed=random_seed,
        perplexity=perplexity,
    )
    print(f"Projecting AFTER enhancement ({method})...")
    after_2d, indices_a = project_embeddings_2d(
        after_np,
        method=method,
        subset_size=subset_size,
        random_seed=random_seed,
        perplexity=perplexity,
    )

    labels_b = labels_np[indices_b]
    labels_a = labels_np[indices_a]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].scatter(
        before_2d[:, 0], before_2d[:, 1],
        c=labels_b,
        s=3,
        cmap='tab20'
    )
    axes[0].set_title("Before Enhancement")
    axes[0].axis('off')

    axes[1].scatter(
        after_2d[:, 0], after_2d[:, 1],
        c=labels_a,
        s=3,
        cmap='tab20'
    )
    axes[1].set_title("After Enhancement")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison figure to: {out_file}")


def project_embeddings_2d(
    embeddings,
    method="tsne",
    subset_size=10000,
    random_seed=42,
    perplexity=30
):
    """
    Project the given high-dimensional embeddings into 2D using either t-SNE or UMAP.

    :param embeddings: np.array of shape [N, D]
    :param method: "tsne" or "umap"
    :param subset_size: how many points to sample for the projection
    :param random_seed: for reproducibility in random sampling and initialization
    :param perplexity: t-SNE parameter (ignored if method="umap")
    :return: projected_data of shape [subset_size, 2] (or [N, 2] if subset_size >= N)
    :raises ValueError: if method="umap" but umap-learn is not installed
    """
    N = embeddings.shape[0]
    # Subsample for feasibility
    if subset_size < N:
        random.seed(random_seed)
        indices = random.sample(range(N), subset_size)
        selected = embeddings[indices]
    else:
        indices = range(N)
        selected = embeddings

    # actual dimensionality reduction
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=random_seed)
        proj = tsne.fit_transform(selected)
    elif method == "umap":
        if not HAS_UMAP:
            raise ValueError("UMAP is not installed. Please install with `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, random_state=random_seed)
        proj = reducer.fit_transform(selected)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'tsne' or 'umap'.")

    return proj, indices


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare embeddings before and after enhancement using t-SNE or UMAP.")
    parser.add_argument("--method", type=str, default="umap", choices=["tsne", "umap"])
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--out_file", type=str, default="comparison_2d.png")
    parser.add_argument("--before_path", type=str, default="/home/alij/SSL-GraphNNCLR/saved_graphs/Cifar10/graphs_with_test_y.pth")
    parser.add_argument("--after_path", type=str, default="/home/alij/SelfGNN/data/custom_with_test_Cifar10/embeddings/embeddings.pth")
    args = parser.parse_args()

    before = torch.load(args.before_path)
    after = torch.load(args.after_path)

    teacher_graph_before = before["teacher_graph"].cpu()
    before_embeddings = teacher_graph_before.x.cpu().numpy()

    after_embeddings = after["embeddings"].cpu()
    labels = after["labels"].cpu()  # assumes labels are shared

    compare_tsne_enhancement(
        before_embeddings,
        after_embeddings,
        labels,
        method=args.method,
        subset_size=args.subset_size,
        random_seed=args.random_seed,
        perplexity=args.perplexity,
        out_file=args.out_file
    )
