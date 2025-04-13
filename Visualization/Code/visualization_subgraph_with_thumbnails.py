import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import torch
import argparse
import time
import tarfile
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from adasim_utils.parser import get_args_parser
from adasim_utils.dataset import DatasetFolderAdaSim
from augmentation import DataAugmentationDINO
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import networkx as nx
from torch.utils.data import DataLoader, DistributedSampler
import os
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import networkx as nx
from PIL import Image
import numpy as np
import json

##########################
# 1) Define train_adasim
##########################
def train_adasim(args):
    """
    Your original training function, with minimal changes.
    We ensure that args.dist_url etc. exist by adding them in get_args().
    """
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args
    )

    # Example code for extracting .tar if needed
    if len(args.untar_path) > 0 and args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]

    if args.data_path.endswith('.tar'):
        # If you're actually dealing with a .tar
        if int(args.gpu) == 0:
            with tarfile.open(args.data_path, 'r') as f:
                f.extractall(args.untar_path)
            print("Time taken for untar:")
        args.data_path = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_train')
        args.data_path_val = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_val')
    else:
        # For ImageNet100, presumably you have train/val subfolders
        args.data_path = os.path.join(args.untar_path, 'train')
        args.data_path_val = os.path.join(args.untar_path, 'val')

    torch.distributed.barrier()  # wait for all procs if in distributed

    # Build dataset & sampler
    dataset = DatasetFolderAdaSim(
        args.data_path,
        args,
        transform=transform,
        return_index_instead_of_target=True
    )
    sampler = DistributedSampler(dataset, shuffle=True)

    # Build DataLoader
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ... your training / fine-tuning code would go here ...
    # For demonstration, we just return the dataset object so we can use it later
    return dataset


##########################
# 2) BFS + shell layout + thumbnail visualization
##########################
def build_subgraph_2hop_and_shells(edge_index, center_node):
    """
    Builds a directed DiGraph from edge_index, does BFS up to 2 hops from center_node,
    returns (subG, shells).
    """
    G = nx.DiGraph()
    num_edges = edge_index.size(1)
    for i in range(num_edges):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(src, dst)

    # BFS up to distance=2
    bfs_result = nx.single_source_shortest_path_length(G, center_node, cutoff=2)
    sub_nodes = list(bfs_result.keys())
    subG = G.subgraph(sub_nodes).copy()

    # separate by distance
    layer0 = [center_node]
    layer1 = [n for n, dist in bfs_result.items() if dist == 1]
    layer2 = [n for n, dist in bfs_result.items() if dist == 2]

    shells = []
    if layer0:
        shells.append(layer0)
    if layer1:
        shells.append(layer1)
    if layer2:
        shells.append(layer2)

    return subG, shells


def draw_subgraph_with_thumbnails(ax, subG, shells, center_node, dataset, zoom=1.0):
    pos = nx.shell_layout(subG, nlist=shells)

    # --- Get label info for incorrect neighbor highlighting ---
    labels = getattr(dataset, 'targets', None) or getattr(dataset, 'labels', None)
    label_lookup = labels if isinstance(labels, list) else labels.tolist() if labels is not None else None

    group_size = None
    if label_lookup is None:
        print("[Warning] Labels not found; cannot infer class groups.")
    else:
        try:
            center_label = label_lookup[center_node]
            sorted_labels = np.array(label_lookup)
            group_indices = np.where(sorted_labels == center_label)[0]
            group_indices.sort()
            diffs = np.diff(group_indices)
            most_common_diff = np.bincount(diffs).argmax() if len(diffs) > 0 else None
            group_size = most_common_diff
        except Exception as e:
            print(f"[Warning] Error determining group size: {e}")

    # --- 1) Draw thumbnails first ---
    for node in subG.nodes():
        x, y = pos[node]
        try:
            img_path = dataset.samples[node][0]
        except IndexError:
            continue
        if not os.path.isfile(img_path):
            continue
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_img = pil_img.resize((120, 120), Image.BICUBIC)
            arr_img = np.array(pil_img)
        except Exception:
            continue

        imagebox = offsetbox.OffsetImage(arr_img, zoom=zoom)

        if node == center_node or label_lookup is None or group_size is None:
            edge_color = 'black'
        else:
            same_class = (label_lookup[node] == label_lookup[center_node])
            edge_color = 'black' if same_class else 'red'

        ab = offsetbox.AnnotationBbox(
            imagebox,
            (x, y),
            frameon=True,
            boxcoords="data",
            pad=0.3 if edge_color == 'red' else 0.2,
            bboxprops=dict(edgecolor=edge_color, linewidth=2)
        )
        ax.add_artist(ab)

    # --- 2) Draw arrows on top ---
    edges = nx.draw_networkx_edges(
        subG,
        pos,
        ax=ax,
        edge_color='gray',
        arrowstyle='-|>',
        arrows=True,
        arrowsize=10,
        connectionstyle='arc3,rad=0.05',
        min_source_margin=10,
        min_target_margin=10
    )
    if edges is not None:
        for artist in edges:
            artist.set_zorder(10)

    # --- 3) Adaptive label placement ---
    center_x, center_y = pos[center_node]
    for node in subG.nodes():
        x, y = pos[node]
        dx = x - center_x
        dy = y - center_y
        angle = np.arctan2(dy, dx)

        offset_x, offset_y = 0.0, 0.0
        ha, va = 'center', 'center'
        radius = 0.07

        if -np.pi/8 <= angle < np.pi/8:
            offset_x, ha = radius, 'left'
        elif np.pi/8 <= angle < 3*np.pi/8:
            offset_x, offset_y = radius * 0.7, radius * 0.7
            ha, va = 'left', 'bottom'
        elif 3*np.pi/8 <= angle < 5*np.pi/8:
            offset_y, va = radius, 'bottom'
        elif 5*np.pi/8 <= angle < 7*np.pi/8:
            offset_x, offset_y = -radius * 0.7, radius * 0.7
            ha, va = 'right', 'bottom'
        elif angle >= 7*np.pi/8 or angle < -7*np.pi/8:
            offset_x, ha = -radius, 'right'
        elif -7*np.pi/8 <= angle < -5*np.pi/8:
            offset_x, offset_y = -radius * 0.7, -radius * 0.7
            ha, va = 'right', 'top'
        elif -5*np.pi/8 <= angle < -3*np.pi/8:
            offset_y, va = -radius, 'top'
        elif -3*np.pi/8 <= angle < -np.pi/8:
            offset_x, offset_y = radius * 0.7, -radius * 0.7
            ha, va = 'left', 'top'

        ax.text(
            x + offset_x, y + offset_y,
            str(node),
            fontsize=8,
            ha=ha,
            va=va,
            color='black'
        )

    ax.set_aspect('equal')
    ax.axis('off')


def get_class_name_from_path(img_path):
    """Extract the class folder from a full image path."""
    return os.path.basename(os.path.dirname(img_path))

def get_readable_label(class_id, label_json_path):
    if not os.path.isfile(label_json_path):
        print(f"[Warning] Label JSON not found: {label_json_path}")
        return class_id, None

    with open(label_json_path, "r") as f:
        label_map = json.load(f)
    return class_id, label_map.get(class_id, None)

def visualize_2hop_node(
    node_idx,
    student_graph,
    teacher_graph,
    dataset,
    out_file="2hop_subgraph.png",
    zoom=0.25,
    label_json_path="/home/alij/SSL-GraphNNCLR/imagenet-100_labels.json"
):
    # Extract class from the center node's image path
    img_path = dataset.samples[node_idx][0]
    class_id = get_class_name_from_path(img_path)
    class_id, class_name = get_readable_label(class_id, label_json_path)
    readable = f"{class_id} → {class_name}" if class_name else class_id

    # Build subgraphs
    subG_stu, shells_stu = build_subgraph_2hop_and_shells(student_graph.edge_index, node_idx)
    subG_tea, shells_tea = build_subgraph_2hop_and_shells(teacher_graph.edge_index, node_idx)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Set the figure-wide title and adjust spacing
    fig.suptitle(f"2-hop subgraph for node {node_idx} ({readable})", fontsize=16, y=1.02)  # <-- pushes it upward

    draw_subgraph_with_thumbnails(axes[0], subG_stu, shells_stu, node_idx, dataset, zoom)
    axes[0].set_title(f"Student (2-hop)")

    draw_subgraph_with_thumbnails(axes[1], subG_tea, shells_tea, node_idx, dataset, zoom)
    axes[1].set_title(f"Teacher (2-hop)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # <-- prevents suptitle from overlapping subplot titles
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2-hop figure to: {out_file}")


##########################
# 3) Main script
##########################
def main():
    # We'll extend the parser from your adasim_utils.parser.get_args_parser
    parser = get_args_parser()

    args = parser.parse_args()
    # Possibly override some defaults for testing
    if not hasattr(args, "untar_path"):
        args.untar_path = "/home/alij/Datasets/ImageNet100"
    if not hasattr(args, "gpu"):
        args.gpu = 0

    args.untar_path = "/home/alij/Datasets/ImageNet100"

    # 1) Train or at least build the dataset:
    dataset = train_adasim(args)

    # 2) Load your adjacency from .pth
    pth_file = "/home/alij/SSL-GraphNNCLR/saved_graphs/ImageNet100/ImageNet100_graphs_epn15.pth"
    graphs_dict = torch.load(pth_file)
    student_graph = graphs_dict["student_graph"]
    teacher_graph = graphs_dict["teacher_graph"]

    # 3) Visualize a node's 2-hop subgraph
    node_idx = 14200
    visualize_2hop_node(
        node_idx=node_idx,
        student_graph=student_graph,
        teacher_graph=teacher_graph,
        dataset=dataset,
        out_file=f"2hop_node{node_idx}.png",
        zoom=0.25
    )

if __name__ == "__main__":
    main()
