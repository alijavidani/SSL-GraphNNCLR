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


def visualize_teacher_student(
    teacher_graph, 
    student_graph, 
    labels, 
    method="tsne",
    subset_size=10000,
    random_seed=42,
    perplexity=30,
    out_file="teacher_student_2d.png"
):
    """
    :param teacher_graph: has teacher_graph.x, shape [N, dim], teacher_graph.y for labels
    :param student_graph: has student_graph.x, shape [N, dim], student_graph.y
    :param labels: array of shape [N], integer labels
    :param method: "tsne" or "umap"
    :param subset_size: how many points to sample for the 2D projection
    :param random_seed: for reproducible sampling/initialization
    :param perplexity: TSNE param (ignored if method="umap")
    :param out_file: where to save the resulting figure
    """
    # Convert to numpy
    teacher_x = teacher_graph.x.cpu().numpy()
    student_x = student_graph.x.cpu().numpy()
    labels = np.array(labels)

    N = teacher_x.shape[0]
    assert student_x.shape[0] == N == labels.shape[0], "Lengths must match"

    # auto-infer num_classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} unique classes from 'labels'.")

    print(f"Projecting TEACHER (method={method}) ...")
    teacher_2d, indices_t = project_embeddings_2d(
        teacher_x,
        method=method,
        subset_size=subset_size,
        random_seed=random_seed,
        perplexity=perplexity
    )
    # get corresponding labels
    labels_t_sub = labels[indices_t]

    print(f"Projecting STUDENT (method={method}) ...")
    student_2d, indices_s = project_embeddings_2d(
        student_x,
        method=method,
        subset_size=subset_size,
        random_seed=random_seed,
        perplexity=perplexity
    )
    labels_s_sub = labels[indices_s]

    # plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # teacher
    sc1 = axes[0].scatter(
        teacher_2d[:, 0], teacher_2d[:, 1],
        c=labels_t_sub,
        s=3,
        cmap='tab20'  # repeats if >20 classes
    )
    axes[0].set_title(f"Teacher ({method.upper()})")
    axes[0].axis('off')

    # student
    sc2 = axes[1].scatter(
        student_2d[:, 0], student_2d[:, 1],
        c=labels_s_sub,
        s=3,
        cmap='tab20'
    )
    axes[1].set_title(f"Student ({method.upper()})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare teacher & student embeddings in 2D via t-SNE or UMAP.")
    parser.add_argument("--pth_file", type=str,
                        default="/home/alij/SSL-GraphNNCLR/saved_graphs/Cifar10/graphs_with_test_y.pth",
                        help="Path to the .pth file containing teacher/student graphs")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"],
                        help="Projection method: 'tsne' or 'umap'")
    parser.add_argument("--subset_size", type=int, default=10000,
                        help="Number of points to sample for the 2D projection")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sampling and initialization")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity param (ignored if method='umap')")
    parser.add_argument("--out_file", type=str, default="teacher_student_2d.png",
                        help="Output filename for the figure")
    args = parser.parse_args()

    # # 1) Load .pth
    # graphs_dict = torch.load(args.pth_file)
    # student_graph = graphs_dict["student_graph"]
    # teacher_graph = graphs_dict["teacher_graph"]

    # # 2) Extract labels from teacher_graph.y (or student_graph.y)
    # labels = teacher_graph.y.cpu().numpy()

    # # 3) Visualize
    # visualize_teacher_student(
    #     teacher_graph=teacher_graph,
    #     student_graph=student_graph,
    #     labels=labels,
    #     method=args.method,
    #     subset_size=args.subset_size,
    #     random_seed=args.random_seed,
    #     perplexity=args.perplexity,
    #     out_file=args.out_file
    # )

    saved_data = torch.load("/home/alij/SelfGNN/data/custom_with_test_Cifar10/embeddings/embeddings.pth")
    embeddings = saved_data["embeddings"].cpu()
    labels = saved_data["labels"].cpu()
    train_mask = saved_data["train_mask"].cpu()
    val_mask = saved_data["val_mask"].cpu()

    teacher_2d, indices_t = project_embeddings_2d(
        embeddings,
        method=args.method,
        subset_size=args.subset_size,
        random_seed=args.random_seed,
        perplexity=args.perplexity,
    )

    # plot side by side
    fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    # teacher
    t = axes.scatter(
        teacher_2d[:, 0], teacher_2d[:, 1],
        c=labels[indices_t],
        s=3,
        cmap='tab20'  # repeats if >20 classes
    )
    axes.set_title(f"Teacher ({args.method.upper()})")
    axes.axis('off')
    out_file="embedding.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_file}")

if __name__ == "__main__":
    main()
