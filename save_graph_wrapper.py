import torch
import os
import argparse
from save_graph import save_graph
from torch_geometric.data import Data

def save_graphs_from_training(teacher_graph, student_graph, teacher_graph_test=None, student_graph_test=None, 
                             dataset_name=None, model_name=None, edges_per_node=1, patch_size=None, output_dir=None,
                             dataset_path=None, train_dir="train", val_dir="val", recount_classes=False):
    """
    Save graphs from training using the save_graph module.
    
    Args:
        teacher_graph: Teacher graph from training (or combined train+test graph)
        student_graph: Student graph from training (or combined train+test graph)
        teacher_graph_test: Teacher test graph from training (or None if already combined)
        student_graph_test: Student test graph from training (or None if already combined)
        dataset_name: Name of the dataset (cifar10, imagenet100, or imagenet1k)
        model_name: Model architecture name
        edges_per_node: Number of edges per node used in training
        patch_size: Optional patch size for ViT models
        output_dir: Directory to save graphs (default: ./saved_graphs)
        dataset_path: Path to the dataset root directory (needed for ImageNet-1K)
        train_dir: Name of the training directory (default: "train")
        val_dir: Name of the validation directory (default: "val")
        recount_classes: Force recount of class images even if cached information exists
    """
    # Create arguments object to pass to save_graph
    class Args:
        pass
    
    args = Args()
    args.dataset_name = dataset_name.lower() if dataset_name else "imagenet1k"
    args.model_name = model_name if model_name else "vit_base"
    args.edges_per_node = edges_per_node
    args.teacher_graph = teacher_graph
    args.student_graph = student_graph
    
    # Handle case where test graphs are None (already combined with training graphs)
    if teacher_graph_test is None or student_graph_test is None:
        print("Using pre-combined graphs (train+test already concatenated)")
        # Set dummy test graphs to avoid errors in save_graph
        # These won't be used for concatenation since the input graphs are already combined
        dummy_node_count = 1
        dummy_dim = teacher_graph.x.shape[1] if hasattr(teacher_graph, 'x') else 64
        
        dummy_test_graph = Data(num_nodes=dummy_node_count)
        dummy_test_graph.x = torch.zeros((dummy_node_count, dummy_dim))
        dummy_test_graph.edge_index = torch.zeros(2, dummy_node_count, dtype=torch.long)
        
        args.teacher_graph_test = dummy_test_graph
        args.student_graph_test = dummy_test_graph
        # Set a flag to indicate pre-combined graphs
        args.graphs_already_combined = True
    else:
        args.teacher_graph_test = teacher_graph_test
        args.student_graph_test = student_graph_test
        args.graphs_already_combined = False
    
    # Set dataset path and related parameters for ImageNet-1K
    args.dataset_path = dataset_path
    args.train_dir = train_dir
    args.val_dir = val_dir
    args.recount_classes = recount_classes
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "./saved_graphs"
    args.output_dir = output_dir
    
    # Determine patch size if not provided (for ViT models)
    if patch_size is None:
        patch_size = "16"  # Default patch size
    
    # Create model shorthand for filename
    if "vit" in args.model_name:
        if args.model_name == "vit_base":
            model_short = "vitb"
        elif args.model_name == "vit_small":
            model_short = "vits"
        elif args.model_name == "vit_tiny":
            model_short = "vitt"
        else:
            model_short = args.model_name
    else:
        model_short = args.model_name
    
    # Create filename and output path
    filename = f"{args.dataset_name}_{model_short}{patch_size}_epn{args.edges_per_node}.pth"
    args.output_path = os.path.join(args.output_dir, args.dataset_name, filename)
    
    # Call save_graph function
    save_graph(args)
    
    return args.output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test save graphs wrapper")
    parser.add_argument("--dataset_name", type=str, default="imagenet1k",
                        choices=["cifar10", "imagenet100", "imagenet1k"],
                        help="Dataset name")
    parser.add_argument("--model_name", type=str, default="vit_small",
                        help="Model architecture name")
    parser.add_argument("--edges_per_node", type=int, default=5,
                        help="Number of edges per node")
    parser.add_argument("--dataset_path", type=str, default="/amin/imagenet/imagenet",
                        help="Path to dataset root directory (needed for ImageNet-1K)")
    parser.add_argument("--train_dir", type=str, default="train",
                        help="Name of the training directory")
    parser.add_argument("--val_dir", type=str, default="val",
                        help="Name of the validation directory")
    parser.add_argument("--recount_classes", action="store_true",
                        help="Force recount of class images even if cached info exists")
    parser.add_argument("--test", action="store_true",
                        help="Run with test data instead of loading from files")
    
    args = parser.parse_args()
    
    if args.test:
        # Create dummy test graphs for testing
        print("Creating test graphs...")
        
        # Get dataset configuration based on dataset name
        dataset_configs = {
            "cifar10": {"num_train": 50000, "num_val": 10000, "feature_dim": 384},
            "imagenet100": {"num_train": 130000, "num_val": 5000, "feature_dim": 768},
            "imagenet1k": {"num_train": 1281167, "num_val": 50000, "feature_dim": 768}
        }
        
        config = dataset_configs[args.dataset_name.lower()]
        num_train = config["num_train"]
        num_val = config["num_val"]
        feature_dim = config["feature_dim"]
        
        # Create dummy teacher graph
        teacher_graph = Data(num_nodes=num_train)
        teacher_graph.x = torch.randn(num_train, feature_dim)
        teacher_graph.edge_index = torch.zeros(2, num_train * args.edges_per_node, dtype=torch.long)
        
        # Create dummy student graph
        student_graph = Data(num_nodes=num_train)
        student_graph.x = torch.randn(num_train, feature_dim)
        student_graph.edge_index = torch.zeros(2, num_train * args.edges_per_node, dtype=torch.long)
        
        # Create dummy teacher test graph
        teacher_graph_test = Data(num_nodes=num_val)
        teacher_graph_test.x = torch.randn(num_val, feature_dim)
        teacher_graph_test.edge_index = torch.zeros(2, num_val * args.edges_per_node, dtype=torch.long)
        
        # Create dummy student test graph
        student_graph_test = Data(num_nodes=num_val)
        student_graph_test.x = torch.randn(num_val, feature_dim) 
        student_graph_test.edge_index = torch.zeros(2, num_val * args.edges_per_node, dtype=torch.long)
        
        # Save the graphs
        output_path = save_graphs_from_training(
            teacher_graph, student_graph, teacher_graph_test, student_graph_test,
            args.dataset_name, args.model_name, args.edges_per_node,
            dataset_path=args.dataset_path, train_dir=args.train_dir, 
            val_dir=args.val_dir, recount_classes=args.recount_classes
        )
        
        print(f"Test graphs saved to: {output_path}")
    else:
        print("Please use --test flag to run a test, or import and call save_graphs_from_training() function") 