import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="0"
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch_geometric.loader as pyg_loader

import numpy as np

import torch

import models
import utils
import data

import os.path as osp
import os
import sys

class ModelTrainer:
    
    """
    A utility class for training SelfGNN
    """

    def __init__(self, args):
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        self._device = torch.device(
            utils.get_device_id(torch.cuda.is_available()))
        self._aug = utils.Augmentations(method=args.aug)

        self._dataset = data.Dataset(root=args.root, name=args.name, num_parts=args.init_parts,
                                     final_parts=args.final_parts, augmentation=self._aug, graph_type=args.graph_type)

        # esi=torch.load('/home/alij/SSL-GraphNNCLR/saved_graphs/ImageNet100/ImageNet100_graphs_epn15.pth')
        # ali=torch.load('/home/alij/SSL-GraphNNCLR/saved_graphs/Cifar10/graphs_with_test_y.pth')
        # self._dataset.data.x=ali['student_graph'].x
        # self._dataset.data.edge_index=ali['student_graph'].edge_index
        # self._dataset.data.x2=ali['teacher_graph'].x
        # self._dataset.data.edge_index2=ali['teacher_graph'].edge_index

        # esi1=self._dataset.data.edge_index[:,1]
        # esi2=self._dataset.data.edge_index2[:,1]
        # print('khaaaaaaaaaaar',esi1==esi2)

        # Use a reasonable batch size to prevent memory issues
        batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size > 0 else 128

        self._loader = DataLoader(
            dataset=self._dataset, batch_size=batch_size)  # [self._dataset.data]
        print(f"Data Augmentation method {args.aug}")
        print(f"Data: {self._dataset.data}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x.shape[1]] + hidden_layers
        self._norm_config = utils.get_norm_configs(args.norms)
        self._model = models.SelfGNN(
            layer_config=layers, 
            dropout=args.dropout, 
            gnn_type=args.model,
            heads=args.heads,
            jk_mode=args.jk_mode,
            mask_ratio=args.mask_ratio,
            feature_masking_weight = args.feature_masking_weight,
            **self._norm_config
        ).to(self._device)
        
        print(self._model)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(), lr=args.lr)

    def train(self):
        """
        Trains SelfGNN in a self-supervised fashion
        """
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                batch_data.to(self._device)
                v1_output, v2_output, loss = self._model(
                    x1=batch_data.x, x2=batch_data.x2, edge_index_v1=batch_data.edge_index, edge_index_v2=batch_data.edge_index2,
                    edge_weight_v1=batch_data.edge_attr, edge_weight_v2=batch_data.edge_attr)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()
                sys.stdout.write('\rEpoch {}/{}, batch {}/{}, loss {:.4f}'.format(epoch + 1, self._args.epochs, bc + 1,
                                                                                  self._dataset.final_parts, loss.data))
                sys.stdout.flush()
            if (epoch + 1) % self._args.cache_step == 0:
                path = osp.join(self._dataset.model_dir,
                                f"model.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
        print()
        
    def infer_embeddings(self):
        """
        Infers node embeddings from the trained SelfGNN model.
        Original method suitable for smaller datasets.
        """
        
        outputs = []
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._val_mask = self._test_mask = None
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            with torch.no_grad():  # Add no_grad to save memory during inference
                v1_output, v2_output, _ = self._model(
                    x1=batch_data.x, x2=batch_data.x2,
                    edge_index_v1=batch_data.edge_index,
                    edge_index_v2=batch_data.edge_index2,
                    edge_weight_v1=batch_data.edge_attr,
                    edge_weight_v2=batch_data.edge_attr,
                    inference_mode=True)
            emb = torch.cat([v1_output, v2_output], dim=1).detach()
            y = batch_data.y.detach()
            trm = batch_data.train_mask.detach()
            dem = batch_data.val_mask.detach()
            # tem = batch_data.test_mask.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
                self._train_mask, self._val_mask = trm, dem#, self._test_mask, tem
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])
                self._train_mask = torch.cat([self._train_mask, trm])
                self._val_mask = torch.cat([self._val_mask, dem])
                # self._test_mask = torch.cat([self._test_mask, tem])

        # Save embeddings to a .pth file
        save_path = osp.join(self._args.root, self._args.name, 'embeddings', "embeddings.pth")
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        torch.save({
            "embeddings": self._embeddings.cpu(),  # Move to CPU before saving
            "labels": self._labels.cpu(),
            "train_mask": self._train_mask.cpu(),
            "val_mask": self._val_mask.cpu(),
            # "test_mask": self._test_mask.cpu()  # Uncomment if test_mask is used
        }, save_path)
        print(f"Embeddings saved to {save_path}")

    def infer_large_embeddings(self):
        """
        Memory-efficient version for inferring node embeddings from large graphs.
        Recommended for datasets like ImageNet1K with >1M nodes.
        Uses neighbor sampling to avoid OOM errors.
        """
        # No need to import here since it's imported at the top
        
        # Set model to evaluation mode
        self._model.train(False)
        
        # Get the total number of nodes from the dataset
        total_nodes = self._dataset.data.num_nodes
        
        # Correctly calculate embedding dimensions based on model structure
        # When using jumping knowledge with concat mode, embedding dimension is multiplied
        hidden_dim = self._model.student_encoder.stacked_gnn[-1].out_channels
        if self._args.jk_mode == "concat":
            # For concat mode, need to account for all layers
            num_layers = len(self._model.student_encoder.stacked_gnn)
            hidden_dim = hidden_dim * num_layers
        
        # For both student and teacher representations
        emb_dim = hidden_dim * 2  # *2 for v1_output and v2_output concat
        
        print(f"Calculated embedding dimension: {emb_dim}")
        
        # Pre-allocate tensors on CPU to save memory
        self._embeddings = torch.zeros((total_nodes, emb_dim), device='cpu')
        self._labels = torch.zeros(total_nodes, dtype=torch.long, device='cpu')
        self._train_mask = torch.zeros(total_nodes, dtype=torch.bool, device='cpu')
        self._val_mask = torch.zeros(total_nodes, dtype=torch.bool, device='cpu')
        
        # Get data
        data = self._dataset.data
        
        # Create a range of all node indices
        all_nodes = torch.arange(total_nodes, device='cpu')
        
        # Process in smaller chunks for node sampling
        chunk_size = min(500, total_nodes)  # Using very small chunks to avoid memory issues
        
        for chunk_start in range(0, total_nodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_nodes)
            print(f"\rProcessing nodes {chunk_start} to {chunk_end} of {total_nodes}", end="")
            
            # Try GPU processing first, fall back to CPU if necessary
            try:
                # Process this chunk with GPU
                self._process_chunk_gpu(chunk_start, chunk_end, all_nodes, data, emb_dim)
            except Exception as e:
                # If any error occurs during GPU processing, try CPU
                print(f"\nError during GPU processing: {str(e)}")
                print("Falling back to CPU processing for this chunk")
                self._process_chunk_cpu(chunk_start, chunk_end, all_nodes, data, emb_dim)
        
        print("\nFinished embedding generation")

        # Create embeddings directory if it doesn't exist
        save_dir = osp.join('/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/SelfGNN', 'embeddings')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save embeddings in smaller chunks to avoid file size issues
        self._save_embeddings_in_chunks(save_dir)
        
    def _save_embeddings_in_chunks(self, save_dir, chunk_size=100000):
        """Helper method to save embeddings in smaller chunks"""
        import datetime
        
        total_nodes = self._embeddings.shape[0]
        
        print(f"Saving embeddings in chunks of {chunk_size} nodes...")
        
        # Save metadata separately (helps with reassembly later)
        metadata_path = osp.join(save_dir, "metadata.pth")
        try:
            torch.save({
                "total_nodes": total_nodes,
                "embedding_dim": self._embeddings.shape[1],
                "num_chunks": (total_nodes + chunk_size - 1) // chunk_size,
                "chunk_size": chunk_size
            }, metadata_path)
            print(f"Saved metadata to {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
        
        # Save each chunk
        success_count = 0
        for chunk_start in range(0, total_nodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_nodes)
            chunk_id = chunk_start // chunk_size
            
            # Prepare chunk data
            chunk_data = {
                "embeddings": self._embeddings[chunk_start:chunk_end],
                "labels": self._labels[chunk_start:chunk_end],
                "train_mask": self._train_mask[chunk_start:chunk_end],
                "val_mask": self._val_mask[chunk_start:chunk_end],
                "start_idx": chunk_start,
                "end_idx": chunk_end
            }
            
            # Save this chunk
            chunk_path = osp.join(save_dir, f"embeddings_chunk_{chunk_id}.pth")
            try:
                torch.save(chunk_data, chunk_path)
                success_count += 1
                print(f"\rSaved chunk {chunk_id+1}/{(total_nodes + chunk_size - 1) // chunk_size}", end="")
            except Exception as e:
                print(f"\nError saving chunk {chunk_id}: {str(e)}")
                
                # Try with pickle protocol 2 (more compatible but larger files)
                try:
                    print("Trying with pickle protocol 2...")
                    torch.save(chunk_data, chunk_path, _use_new_zipfile_serialization=False, pickle_protocol=2)
                    success_count += 1
                    print(f"Successfully saved chunk {chunk_id} with protocol 2")
                except Exception as e2:
                    print(f"Still failed with protocol 2: {str(e2)}")
        
        print(f"\nSuccessfully saved {success_count} chunks out of {(total_nodes + chunk_size - 1) // chunk_size}")
        print(f"Embeddings saved to {save_dir}/")
        
        # Save a small summary file to indicate completion
        summary_path = osp.join(save_dir, "summary.txt")
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Total nodes: {total_nodes}\n")
                f.write(f"Embedding dimension: {self._embeddings.shape[1]}\n")
                f.write(f"Chunks saved: {success_count}/{(total_nodes + chunk_size - 1) // chunk_size}\n")
                f.write(f"Dataset: {self._args.name}\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Error saving summary: {str(e)}")

    def _process_chunk_gpu(self, chunk_start, chunk_end, all_nodes, data, emb_dim):
        """Helper method to process a chunk on GPU"""
        # No need to import here since it's imported at the top
        
        with torch.no_grad():
            # Get current batch of nodes
            batch_nodes = all_nodes[chunk_start:chunk_end].to(self._device)
            
            # Use just 1 hop for memory efficiency
            num_hops = 1
            
            # Process student graph
            edge_index_v1 = data.edge_index.to(self._device)
            x1 = data.x.to(self._device)
            
            student_subset, student_edge_index, _, _ = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=num_hops,
                edge_index=edge_index_v1,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            x1_sub = x1[student_subset]
            
            # Process teacher graph
            edge_index_v2 = data.edge_index2.to(self._device)
            x2 = data.x2.to(self._device)
            
            teacher_subset, teacher_edge_index, _, _ = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=num_hops,
                edge_index=edge_index_v2,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            x2_sub = x2[teacher_subset]
            
            # The first len(batch_nodes) nodes are the original nodes
            batch_local_indices = torch.arange(len(batch_nodes), device=self._device)
            
            # Forward pass
            v1_output, v2_output, _ = self._model(
                x1=x1_sub, 
                x2=x2_sub,
                edge_index_v1=student_edge_index,
                edge_index_v2=teacher_edge_index,
                edge_weight_v1=None,
                edge_weight_v2=None,
                inference_mode=True
            )
            
            # Combine outputs
            batch_emb = torch.cat([v1_output[batch_local_indices], v2_output[batch_local_indices]], dim=1)
            
            # Store results - make sure everything is on CPU
            self._embeddings[chunk_start:chunk_end] = batch_emb.cpu()
            batch_nodes_cpu = batch_nodes.cpu()
            self._labels[chunk_start:chunk_end] = data.y[batch_nodes_cpu]
            self._train_mask[chunk_start:chunk_end] = data.train_mask[batch_nodes_cpu]
            self._val_mask[chunk_start:chunk_end] = data.val_mask[batch_nodes_cpu]
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
    def _process_chunk_cpu(self, chunk_start, chunk_end, all_nodes, data, emb_dim):
        """Helper method to process a chunk on CPU, used as fallback"""
        # No need to import here since it's imported at the top
        
        with torch.no_grad():
            # Move model to CPU
            cpu_model = self._model.cpu()
            
            # Get batch nodes - stay on CPU for everything
            batch_nodes = all_nodes[chunk_start:chunk_end]  # already on CPU
            
            # Process with minimal hops
            edge_index_v1 = data.edge_index.cpu()
            x1 = data.x.cpu()
            
            student_subset, student_edge_index, _, _ = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=1,
                edge_index=edge_index_v1,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            x1_sub = x1[student_subset]
            
            # Teacher graph
            edge_index_v2 = data.edge_index2.cpu()
            x2 = data.x2.cpu()
            
            teacher_subset, teacher_edge_index, _, _ = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=1,
                edge_index=edge_index_v2,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            x2_sub = x2[teacher_subset]
            
            # Get local indices
            batch_local_indices = torch.arange(len(batch_nodes), device='cpu')
            
            # Process on CPU
            v1_output, v2_output, _ = cpu_model(
                x1=x1_sub, 
                x2=x2_sub,
                edge_index_v1=student_edge_index,
                edge_index_v2=teacher_edge_index,
                edge_weight_v1=None,
                edge_weight_v2=None,
                inference_mode=True
            )
            
            # Combine outputs
            batch_emb = torch.cat([v1_output[batch_local_indices], v2_output[batch_local_indices]], dim=1)
            
            # Store results - already on CPU
            self._embeddings[chunk_start:chunk_end] = batch_emb
            self._labels[chunk_start:chunk_end] = data.y[batch_nodes]
            self._train_mask[chunk_start:chunk_end] = data.train_mask[batch_nodes]
            self._val_mask[chunk_start:chunk_end] = data.val_mask[batch_nodes]
            
            # Move model back to GPU
            self._model = cpu_model.to(self._device)

    def evaluate_epoch(self, epoch):
        """
        Evaluates SelfGNN saved after a given training epoch
        :param epoch: The epoch to be evaluated
        :return: The validation and test accuracy of the model saved at a given epoch
        """
        path = osp.join(self._dataset.model_dir, f"model.ep.{epoch}.pt")
        self._model.load_state_dict(
            torch.load(path, map_location=self._device))
        self.infer_embeddings()
        dev_acu = self.evaluate_semi(partition="dev")
        test_acu = self.evaluate_semi(partition="test")
        return dev_acu, test_acu

    def search_best_epoch(self):
        """
        Searches for the best epoch that leads to the best validation accuracy. 
        Used for hyperparameter tuning
        
        """
        model_files = os.listdir(self._dataset.model_dir)
        results = []
        best_epoch = -1, (0,), (0,)
        for i, model_file in enumerate(model_files):
            if model_file.endswith(".pt"):
                substr = model_file.split(".")
                epoch = int(substr[substr.index("ep") + 1])
                dev_acu, test_acu = self.evaluate_epoch(epoch)
                results.append([epoch, dev_acu, test_acu])
                if dev_acu[0] > best_epoch[1][0]:
                    best_epoch = epoch, dev_acu, test_acu
                print(epoch, dev_acu, test_acu)

        dev_accuracy, dev_std = best_epoch[1]
        test_accuracy, test_std = best_epoch[2]
        print(f"The best epoch is: {best_epoch[0]}")
        print(f"with validation accuracy: {dev_accuracy} std: {dev_std}")
        print(f"with test accuracy: {test_accuracy} std: {test_std}")
        path = osp.join(self._dataset.result_dir, "results-bn.txt")
        has_header = False
        algorithm = f"SelfGNN-{str(self._aug)}"
        with open(path, "a") as f:
            # Entry Dataset,Group,DevAccuracy,DevStd,TestAccuracy,TestStd,Algorithm
            f.write(
                f"{self._dataset.name.title()},{self._args.model.upper()},{dev_accuracy},{dev_std},{test_accuracy},{test_std},{algorithm}\n")

    
    def evaluate_semi(self, split="dev"):
        """
        Evaluates SelfGNN on a given split in a semi-supervised fashion
        :param split: The split to be evaluated
        :return: Accuracy along with the standard deviation
        
        Note: Used mainly for hyperparameter search
        """
        if split == "train":
            mask = self._train_mask
        elif split == "dev":
            mask = self._val_mask
        else:
            mask = self._test_mask
        
        features = self._embeddings[mask].detach().cpu().numpy()
        labels = self._labels[mask].detach().cpu().numpy()
        accuracy, std = utils.evaluate(features, labels)
        return accuracy, std
    
    def evaluate(self):
        """
        Evaluates SelfGNN on train and validation splits.
        Uses chunk-by-chunk processing for ImageNet1K to save memory.
        """
        print("Evaluating ...")
        
        # Special handling for ImageNet1K: process chunks one at a time
        if self._args.name.lower() == 'imagenet1k':
            return self._evaluate_large_dataset_by_chunks()
        
        # Regular evaluation for other datasets
        if not hasattr(self, '_embeddings') or self._embeddings is None:
            print("Loading embeddings...")
            self._load_chunked_embeddings()
        
        # Set test_mask to None if it doesn't exist
        if not hasattr(self, '_test_mask'):
            self._test_mask = None
            print("No test mask found, evaluation will be performed only on validation set.")
        
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]
        
        dev_accs, test_accs = [], []
        args = self._args

        # Determine how many iterations based on shape of train_mask
        if len(self._train_mask.shape) == 1:
            iters = 20
        else:
            iters = self._train_mask.shape[1]

        for i in range(iters):
            classifier = models.LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

            # Grab the appropriate column of masks if they're 2D, otherwise use them as is
            train_mask, val_mask, test_mask = index_mask(
                self._train_mask, self._val_mask, self._test_mask, index=i
            )

            # Train logistic regression on train_mask
            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[train_mask], self._labels[train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on val (dev_mask)
            classifier.eval()
            dev_logits, _ = classifier(self._embeddings[val_mask], self._labels[val_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            dev_acc = (dev_preds == self._labels[val_mask]).float().mean().cpu().item() * 100
            dev_accs.append(dev_acc)

            # Evaluate on test if test_mask is not None
            if test_mask is not None:
                test_logits, _ = classifier(self._embeddings[test_mask], self._labels[test_mask])
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == self._labels[test_mask]).float().mean().cpu().item() * 100
                test_accs.append(test_acc)

                print(
                    f"Iteration {i + 1:02d} of LR: "
                    f"Val Acc {dev_acc:.2f}, Test Acc {test_acc:.2f}"
                )
            else:
                print(
                    f"Iteration {i + 1:02d} of LR: "
                    f"Val Acc {dev_acc:.2f}, No Test Set"
                )

        # Convert dev_accs to numpy
        dev_accs = np.array(dev_accs)
        dev_acc_mean, dev_acc_std = dev_accs.mean(), dev_accs.std()

        # If we never had a test_mask, test_accs might be empty
        if test_accs:
            test_accs = np.array(test_accs)
            test_acc_mean, test_acc_std = test_accs.mean(), test_accs.std()
        else:
            test_acc_mean, test_acc_std = None, None

        nc = self._norm_config
        path = osp.join(
            self._dataset.result_dir, 
            f"results-norm.encoder.{nc['encoder_norm']}.projection."
            f"{nc['prj_head_norm']}.prediction.{nc['prd_head_norm']}.txt"
        )

        # Write results to file
        with open(path, 'w') as f:
            if test_acc_mean is not None:
                f.write(f"{args.name},{args.model},{dev_acc_mean:.4f},{dev_acc_std:.2f},{test_acc_mean:.4f},{test_acc_std:.2f}")
            else:
                # If no test set, just write NA for test results
                f.write(f"{args.name},{args.model},{dev_acc_mean:.4f},{dev_acc_std:.2f},NA,NA")

        print(f"Average validation accuracy: {dev_acc_mean:.2f} ± {dev_acc_std:.2f}")
        if test_acc_mean is not None:
            print(f"Average test accuracy: {test_acc_mean:.2f} ± {test_acc_std:.2f}")
        else:
            print("Test mask was None, skipping test accuracy.")

        return dev_acc_mean, dev_acc_std, test_acc_mean, test_acc_std
        
    def _load_chunked_embeddings(self):
        """Load embeddings from saved chunks"""
        import glob
        
        # Find the directory containing the chunks
        save_dir = osp.join('/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/SelfGNN', 'embeddings')
        if not osp.exists(save_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {save_dir}")
            
        # Try to load metadata first
        metadata_path = osp.join(save_dir, "metadata.pth")
        if osp.exists(metadata_path):
            try:
                metadata = torch.load(metadata_path)
                total_nodes = metadata["total_nodes"]
                embedding_dim = metadata["embedding_dim"]
                print(f"Found metadata: {total_nodes} nodes with {embedding_dim} dimensions")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                # Try to infer from chunks
                chunk_files = sorted(glob.glob(osp.join(save_dir, "embeddings_chunk_*.pth")))
                if not chunk_files:
                    raise FileNotFoundError(f"No embedding chunks found in {save_dir}")
                    
                # Load first chunk to get dimensions
                first_chunk = torch.load(chunk_files[0])
                embedding_dim = first_chunk["embeddings"].shape[1]
                
                # Guess total nodes from last chunk
                last_chunk = torch.load(chunk_files[-1])
                total_nodes = last_chunk["end_idx"]
        else:
            # No metadata - infer from chunks
            chunk_files = sorted(glob.glob(osp.join(save_dir, "embeddings_chunk_*.pth")))
            if not chunk_files:
                raise FileNotFoundError(f"No embedding chunks found in {save_dir}")
                
            # Load first chunk to get dimensions
            first_chunk = torch.load(chunk_files[0])
            embedding_dim = first_chunk["embeddings"].shape[1]
            
            # Guess total nodes from last chunk
            last_chunk = torch.load(chunk_files[-1])
            total_nodes = last_chunk["end_idx"]
        
        # Initialize empty tensors
        self._embeddings = torch.zeros((total_nodes, embedding_dim), device='cpu')
        self._labels = torch.zeros(total_nodes, dtype=torch.long, device='cpu')
        self._train_mask = torch.zeros(total_nodes, dtype=torch.bool, device='cpu')
        self._val_mask = torch.zeros(total_nodes, dtype=torch.bool, device='cpu')
        
        # Load all chunks
        chunk_files = sorted(glob.glob(osp.join(save_dir, "embeddings_chunk_*.pth")))
        print(f"Loading {len(chunk_files)} embedding chunks...")
        
        for i, chunk_file in enumerate(chunk_files):
            try:
                chunk_data = torch.load(chunk_file)
                start_idx = chunk_data["start_idx"]
                end_idx = chunk_data["end_idx"]
                
                # Copy data from chunk to main tensors
                self._embeddings[start_idx:end_idx] = chunk_data["embeddings"]
                self._labels[start_idx:end_idx] = chunk_data["labels"]
                self._train_mask[start_idx:end_idx] = chunk_data["train_mask"]
                self._val_mask[start_idx:end_idx] = chunk_data["val_mask"]
                
                print(f"\rLoaded chunk {i+1}/{len(chunk_files)}", end="")
            except Exception as e:
                print(f"\nError loading chunk {chunk_file}: {str(e)}")
        
        print("\nAll embedding chunks loaded successfully")

    def _evaluate_large_dataset_by_chunks(self):
        """
        Memory-efficient evaluation for ImageNet1K.
        Processes one chunk at a time instead of loading all embeddings.
        """
        import glob
        
        # Find the directory containing the chunks
        save_dir = osp.join('/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/SelfGNN', 'embeddings')
        if not osp.exists(save_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {save_dir}")
        
        # Get metadata
        metadata_path = osp.join(save_dir, "metadata.pth")
        if osp.exists(metadata_path):
            metadata = torch.load(metadata_path)
            total_nodes = metadata["total_nodes"]
            embedding_dim = metadata["embedding_dim"]
            print(f"Found metadata: {total_nodes} nodes with {embedding_dim} dimensions")
        else:
            # Infer from chunks
            chunk_files = sorted(glob.glob(osp.join(save_dir, "embeddings_chunk_*.pth")))
            if not chunk_files:
                raise FileNotFoundError(f"No embedding chunks found in {save_dir}")
            print(f"No metadata found. Will process {len(chunk_files)} chunks.")
        
        # Get all chunk files
        chunk_files = sorted(glob.glob(osp.join(save_dir, "embeddings_chunk_*.pth")))
        print(f"Processing {len(chunk_files)} embedding chunks one at a time...")
        
        # Setup for evaluation
        args = self._args
        dev_accs, test_accs = [], []
        iters = 20  # Number of trials for evaluation
        
        # For each evaluation trial, we'll process all chunks
        for trial in range(iters):
            print(f"\nEvaluation trial {trial+1}/{iters}")
            
            # Initialize model and optimizer for this trial
            classifier = None
            optimizer = None
            
            # First pass: Process each chunk for training
            print("Training phase - processing chunks:")
            for i, chunk_file in enumerate(chunk_files):
                print(f"\nLoading chunk {i+1}/{len(chunk_files)} for training...")
                
                # Load chunk
                chunk_data = torch.load(chunk_file)
                
                # Extract data for this chunk
                embeddings = chunk_data["embeddings"].to(self._device)
                labels = chunk_data["labels"].to(self._device)
                train_mask = chunk_data["train_mask"].to(self._device)
                val_mask = chunk_data["val_mask"].to(self._device)
                
                # Initialize classifier on first chunk
                if classifier is None:
                    num_class = torch.unique(labels).shape[0]
                    classifier = models.LogisticRegression(embeddings.shape[1], num_class).to(self._device)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
                    print(f"Initialized classifier with {embeddings.shape[1]} input features and {num_class} classes")
                
                # Train on this chunk (only on train mask)
                if train_mask.sum() > 0:
                    print(f"Training on {train_mask.sum().item()} samples from chunk {i+1}")
                    for _ in range(10):  # Fewer iterations per chunk
                        classifier.train()
                        logits, loss = classifier(embeddings[train_mask], labels[train_mask])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # Clear GPU memory
                del embeddings, labels, train_mask, val_mask, chunk_data
                torch.cuda.empty_cache()
            
            # Second pass: Evaluate on validation sets from all chunks
            print("\nEvaluation phase - processing chunks:")
            all_val_preds = []
            all_val_labels = []
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"\nLoading chunk {i+1}/{len(chunk_files)} for evaluation...")
                
                # Load chunk
                chunk_data = torch.load(chunk_file)
                
                # Extract validation data
                val_mask = chunk_data["val_mask"]
                if val_mask.sum() > 0:
                    embeddings = chunk_data["embeddings"][val_mask].to(self._device)
                    labels = chunk_data["labels"][val_mask].to(self._device)
                    
                    # Evaluate
                    classifier.eval()
                    with torch.no_grad():
                        val_logits, _ = classifier(embeddings, labels)
                    val_preds = torch.argmax(val_logits, dim=1).cpu()
                    
                    # Store predictions and labels for accuracy calculation
                    all_val_preds.append(val_preds)
                    all_val_labels.append(labels.cpu())
                    
                    print(f"Evaluated {val_mask.sum().item()} validation samples from chunk {i+1}")
                
                # Clear GPU memory
                del chunk_data
                if val_mask.sum() > 0:
                    del embeddings, labels, val_mask
                torch.cuda.empty_cache()
            
            # Calculate accuracy for this trial
            all_val_preds = torch.cat(all_val_preds)
            all_val_labels = torch.cat(all_val_labels)
            val_acc = (all_val_preds == all_val_labels).float().mean().item() * 100
            dev_accs.append(val_acc)
            
            print(f"Trial {trial+1}/{iters}: Val Acc = {val_acc:.2f}%")
            
            # Clean up
            del classifier, optimizer
            torch.cuda.empty_cache()
        
        # Calculate overall results
        dev_accs = np.array(dev_accs)
        dev_acc_mean, dev_acc_std = dev_accs.mean(), dev_accs.std()
        
        # No test_mask, so test accuracy is None
        test_acc_mean, test_acc_std = None, None
        
        # Save results
        nc = self._norm_config
        path = osp.join(
            self._dataset.result_dir, 
            f"results-norm.encoder.{nc['encoder_norm']}.projection."
            f"{nc['prj_head_norm']}.prediction.{nc['prd_head_norm']}.txt"
        )
        
        # Write results to file
        with open(path, 'w') as f:
            f.write(f"{args.name},{args.model},{dev_acc_mean:.4f},{dev_acc_std:.2f},NA,NA")
        
        print(f"Average validation accuracy: {dev_acc_mean:.2f} ± {dev_acc_std:.2f}")
        print("Chunked evaluation completed successfully!")
        
        return dev_acc_mean, dev_acc_std, test_acc_mean, test_acc_std

def set_random_seed(seed):
    """
    Set seeds for reproducibility across random, numpy, and torch.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_evaluate(self, data, seed=42):
    """
    Evaluates SelfGNN using node features (x) from the Data object as embeddings.

    If 'val_mask' or 'train_mask' is not already present, creates splits with a 70/10/20 ratio.
    If 'test_mask' is missing entirely, we skip test accuracy reporting.
    
    :param data: PyTorch Geometric Data object with x, y, and optional train/val/test masks
    :param seed: Random seed for reproducibility
    :return: (dev_acc, dev_std, test_acc, test_std) if test_mask is found,
             otherwise (dev_acc, dev_std, None, None) for test scores.
    """
    print("Evaluating with Improved Raw Method...")
    set_random_seed(seed)

    # 1) Use node features as embeddings
    self._embeddings = torch.cat([data.x, data.x2], dim=1)  # Node features
    self._labels = data.y      # Node labels
    num_nodes = data.num_nodes

    # 2) Ensure device compatibility
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self._embeddings = self._embeddings.to(self._device)
    self._labels = self._labels.to(self._device)

    # 3) Check for existing masks
    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask"):
        print("No predefined train/val_mask found. Creating 70/10/20 splits (class-stratified).")
        # Create train and val masks (and test if we want)
        unique_labels = self._labels.unique()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)  # We'll create it by default

        for label in unique_labels:
            class_indices = (self._labels == label).nonzero(as_tuple=True)[0]
            class_indices = class_indices[torch.randperm(class_indices.size(0))]  # Shuffle
            num_class_samples = class_indices.size(0)

            num_train = int(0.7 * num_class_samples)
            num_val = int(0.1 * num_class_samples)
            num_test = num_class_samples - num_train - num_val

            train_mask[class_indices[:num_train]] = True
            val_mask[class_indices[num_train:num_train + num_val]] = True
            test_mask[class_indices[num_train + num_val:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask  # We'll assume you want a test split if you didn't have any masks
    else:
        print("Using predefined train/val_mask from the data object.")
        train_mask = data.train_mask
        val_mask = data.val_mask

        # For the test mask, if it does NOT exist, set it to None
        if hasattr(data, "test_mask"):
            test_mask = data.test_mask
        else:
            test_mask = None

    # Move masks to device
    train_mask = train_mask.to(self._device)
    val_mask = val_mask.to(self._device)
    if test_mask is not None:
        test_mask = test_mask.to(self._device)

    # 4) Logistic Regression Evaluation
    emb_dim = self._embeddings.shape[1]
    num_class = self._labels.unique().shape[0]
    dev_accs, test_accs = [], []
    args = self._args
    iters = 20  # Number of training iterations for evaluation

    for i in range(iters):
        classifier = models.LogisticRegression(emb_dim, num_class).to(self._device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(100):
            classifier.train()
            logits, loss = classifier(self._embeddings[train_mask], self._labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation
        classifier.eval()
        with torch.no_grad():
            dev_logits, _ = classifier(self._embeddings[val_mask], self._labels[val_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        dev_acc = (dev_preds == self._labels[val_mask]).float().mean().cpu().item() * 100
        dev_accs.append(dev_acc)

        # Evaluate on test only if we have a test_mask
        if test_mask is not None:
            with torch.no_grad():
                test_logits, _ = classifier(self._embeddings[test_mask], self._labels[test_mask])
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == self._labels[test_mask]).float().mean().cpu().item() * 100
            test_accs.append(test_acc)
            print(f"Iteration {i+1:02d}: Val Acc {dev_acc:.2f}%, Test Acc {test_acc:.2f}%")
        else:
            # No test mask => just print validation accuracy
            print(f"Iteration {i+1:02d}: Val Acc {dev_acc:.2f}%")

    # 5) Compute Mean and Standard Deviation
    dev_accs = np.array(dev_accs)
    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()

    if test_mask is not None and len(test_accs) > 0:
        test_accs = np.array(test_accs)
        test_acc, test_std = test_accs.mean(), test_accs.std()
    else:
        test_acc, test_std = None, None

    # 6) Save results
    nc = self._norm_config
    path = osp.join(
        self._dataset.result_dir, 
        f"results-norm.encoder.{nc['encoder_norm']}.projection.{nc['prj_head_norm']}.prediction.{nc['prd_head_norm']}.txt"
    )

    # If no test mask, we skip test acc in the file
    with open(path, 'w') as f:
        if test_acc is not None:
            f.write(f"{args.name},{args.model},{dev_acc:.4f},{dev_std:.2f},{test_acc:.4f},{test_std:.2f}")
        else:
            f.write(f"{args.name},{args.model},{dev_acc:.4f},{dev_std:.2f},NA,NA")

    print(f'Average validation accuracy: {dev_acc:.2f} with std: {dev_std:.2f}')
    if test_acc is not None:
        print(f'Average test accuracy: {test_acc:.2f} with std: {test_std:.2f}')
    else:
        print('No test mask available; skipping test accuracy report.')

    return dev_acc, dev_std, test_acc, test_std


def index_mask(train_mask, val_mask=None, test_mask=None, index=0):
    """
    Returns train_mask, val_mask, test_mask for the specified index if 2D.
    If mask is None or 1D, return as is. If test_mask is None, remains None.
    """
    if len(train_mask.shape) > 1:
        train_mask = train_mask[:, index]
    if val_mask is not None and len(val_mask.shape) > 1:
        val_mask = val_mask[:, index]
    # test_mask might be None; only slice if it's not None and > 1D
    if test_mask is not None and len(test_mask.shape) > 1:
        test_mask = test_mask[:, index]

    return train_mask, val_mask, test_mask


# def index_mask(train_mask, val_mask=None, test_mask=None, index=0):
#     train_mask = train_mask if len(train_mask.shape) == 1 else train_mask[:, index]
#     val_mask = val_mask if val_mask is None or len(val_mask.shape) == 1 else val_mask[:, index]
#     test_mask = test_mask if test_mask is None or len(test_mask.shape) == 1 else test_mask[:, index]
#     return train_mask, val_mask, test_mask

        
def train_search_eval(args):
    """
    Pipeline for tuning
    """
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.search_best_epoch()
    

def train_eval(args):
    """
    SelfGNN training and evaluation pipeline
    """
    # pretrained_state = torch.load("/home/alij/SelfGNN/data/custom_with_test_Cifar10/model/model.ep.1000.pt")
    # print(pretrained_state.keys())
    # trainer._model.load_state_dict(torch.load("/home/alij/SelfGNN/data/custom_with_test_Cifar10/model/model.ep.1000.pt"))

    trainer = ModelTrainer(args)
    trainer.train()
    trainer.infer_embeddings()
    trainer.evaluate()
    # raw_evaluate(trainer, trainer._dataset.data)

    # try:
    #     trainer = ModelTrainer(args)
        
    #     # Check if model should be trained
    #     if args.train_model:
    #         print(f"Training model as requested via --train_model flag...")
    #         trainer.train()
    #     else:
    #         print(f"Skipping model training (use --train_model flag to train)")
        
    #     # Custom embedding path check
    #     custom_embedding_path = '/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/SelfGNN/embeddings'
    #     print(f"Checking for embeddings in custom location: {custom_embedding_path}")
        
    #     # Use different embedding methods based on dataset size
    #     data_name = args.name.lower()
    #     total_nodes = trainer._dataset.data.num_nodes
        
    #     # Determine if we should generate embeddings
    #     default_embeddings_exist = os.path.exists(osp.join(args.root, args.name, 'embeddings'))
    #     custom_embeddings_exist = os.path.exists(custom_embedding_path)
        
    #     embeddings_exist = default_embeddings_exist or custom_embeddings_exist
        
    #     if custom_embeddings_exist:
    #         print(f"Found embeddings in custom location: {custom_embedding_path}")
        
    #     should_generate = (not embeddings_exist) or args.train_model or args.regenerate_embeddings
        
    #     if should_generate:
    #         reason = "not found" if not embeddings_exist else "regeneration requested"
    #         print(f"Generating embeddings ({reason})...")
            
    #         # Use the large embedding method for ImageNet or any dataset with >500K nodes
    #         if 'imagenet' in data_name or total_nodes > 500000:
    #             print(f"Using memory-efficient embedding for large dataset: {data_name} with {total_nodes} nodes")
    #             trainer.infer_large_embeddings()
    #         else:
    #             print(f"Using standard embedding for dataset: {data_name} with {total_nodes} nodes")
    #             trainer.infer_embeddings()
    #     else:
    #         print(f"Found existing embeddings. Skipping embedding generation.")
    #         print(f"(Use --regenerate_embeddings flag to force regeneration)")
        
    #     # Now evaluate the model using the embeddings (will load from chunks if needed)
    #     print("Evaluating model performance...")
    #     try:
    #         trainer.evaluate()
    #     except Exception as eval_error:
    #         print(f"Error during evaluation: {str(eval_error)}")
    #         print("Trying to continue with what we have...")
            
    # except Exception as e:
    #     import traceback
    #     print(f"Error in train_eval: {str(e)}")
    #     traceback.print_exc()
    #     print("\nTrying to salvage partial results if possible...")
        
    #     # Try to create a trainer and evaluate with whatever embeddings we might have
    #     try:
    #         trainer = ModelTrainer(args)
    #         trainer.evaluate()
    #     except Exception as recovery_error:
    #         print(f"Recovery also failed: {str(recovery_error)}")
    #         print("Please check the embeddings directory for partially saved results.")

def main():
    
    args = utils.parse_args()
    print(args)
    train_eval(args)


if __name__ == "__main__":
    main()
