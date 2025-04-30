from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, Actor
from torch_geometric.data import Data, ClusterData, InMemoryDataset
from torch_geometric.utils import subgraph

from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch

import os.path as osp
import sys

import utils


# class Dataset(InMemoryDataset):

#     """
#     A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
#     """

#     def __init__(self, root="data", name='cora', num_parts=1, final_parts=1, augmentation=None, transform=None,
#                  pre_transform=None):
#         self.num_parts = num_parts
#         self.final_parts = final_parts
#         self.augmentation = augmentation
#         super().__init__(root=osp.join(root, name), transform=transform, pre_transform=pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
        
#     def download(self):
#         utils.create_dirs(self.dirs)
#         dataset = fetch_dataset(*osp.split(self.root))
#         utils.create_masks(data=dataset.data)
#         data = dataset.data
#         edge_attr = torch.ones(data.edge_index.shape[1]) if data.edge_attr is None else data.edge_attr
#         data.edge_attr = edge_attr
#         torch.save((data, dataset.slices), self.processed_paths[1])

#     def process(self):
#         """
#         Process either a full batch or cluster data.

#         :return:
#         """
#         data, _ = torch.load(self.processed_paths[1])
#         if self.num_parts == 1:
#             data_list = self.process_full_batch_data(data)
#         else:
#             data_list = self.process_cluster_data(data)
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])
        
#     def process_full_batch_data(self, view1data):
#         """
#         Augmented view data generation using the full-batch data.

#         :param view1data:
#         :return:
#         """
#         print("Processing full batch data")
#         view2data = view1data
#         # view1data = view1data.cpu()
#         # view2data = view1data if self.augmentation is None else self.augmentation(view1data)
#         diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
#         if diff > 0:
#             """
#             Data augmentation on the features could lead to mismatch between the shape of the two views,
#             hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
#             reflect on the original data)
#             """
#             smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
#             smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
#             view1data.x = F.normalize(view1data.x)
#             view2data.x = F.normalize(view2data.x)
        
#         nodes = torch.tensor(np.arange(view1data.num_nodes), dtype=torch.long)
#         data = Data(nodes=nodes, edge_index=view1data.edge_index, edge_index2=view2data.edge_index,
#                     edge_attr=view1data.edge_attr,
#                     edge_attr2=view2data.edge_attr, x=view1data.x, x2=view2data.x, y=view1data.y,
#                     train_mask=view1data.train_mask,
#                     val_mask=view1data.val_mask, num_nodes=view1data.num_nodes)#, test_mask=view1data.test_mask
#         return [data]

#     def process_cluster_data(self, data):
#         """
#         Data processing for ClusterSelfGNN. First the data object will be clustered according to the number of partition
#         specified by this class. Then, we randomly sample a number of clusters and merge them together. Finally, data 
#         augmentation is applied each of the final clusters. This is a simple strategy motivated by ClusterGCN and 
#         employed to improve the scalability of SelfGNN.

#         :param data: A PyTorch Geometric Data object
#         :return: a list of Data objects depending on the final number of clusters.
#         """
#         data_list = []
#         clusters = []
#         num_parts, cluster_size = self.num_parts, self.num_parts // self.final_parts

#         # Cluster the data
#         # cd = ClusterData(data, num_parts=num_parts)
#         # for i in range(1, cd.partptr.shape[0]):
#         #     cls_nodes = cd.perm[cd.partptr[i - 1]: cd.partptr[i]]
#         #     clusters.append(cls_nodes)

#         cd = ClusterData(data, num_parts=num_parts)
#         # Access `partptr` and `perm` from the `partition` attribute
#         partptr = cd.partition.partptr  # Partition pointers
#         perm = cd.partition.node_perm   # Node permutation for clustering

#         clusters = []  # Store the node indices for each cluster
#         for i in range(1, partptr.numel()):
#             # Extract node indices for the current cluster
#             cls_nodes = perm[partptr[i - 1]: partptr[i]]
#             clusters.append(cls_nodes)

#         # Randomly merge clusters and apply transformation
#         np.random.shuffle(clusters)
#         for i in tqdm(range(0, len(clusters), cluster_size), "Processing clusters"):
#             end = i + cluster_size if len(clusters) - i > cluster_size else len(clusters)
#             cls_nodes = torch.cat(clusters[i:end]).unique()

#             x = data.x[cls_nodes]
#             y = data.y[cls_nodes]
#             train_mask = data.train_mask[cls_nodes]
#             val_mask = data.val_mask[cls_nodes]
#             # test_mask = data.test_mask[cls_nodes]
#             edge_index, edge_attr = subgraph(cls_nodes, data.edge_index, relabel_nodes=True)
#             view1data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=cls_nodes.shape[0])
#             view2data = view1data
#             # view2data = view1data if self.augmentation is None else self.augmentation(view1data)
#             if not hasattr(view2data, "edge_attr") or view2data.edge_attr is None:
#                 view2data.edge_attr = torch.ones(view2data.edge_index.shape[1])
#             diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
#             if diff > 0:
#                 smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
#                 smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
#                 view1data.x = F.normalize(view1data.x)
#                 view2data.x = F.normalize(view2data.x)
#             new_data = Data(y=y, x=view1data.x, x2=view2data.x, edge_index=view1data.edge_index,
#                             edge_index2=view2data.edge_index,
#                             edge_attr=view1data.edge_attr, edge_attr2=view2data.edge_attr, train_mask=train_mask,
#                             val_mask=val_mask, num_nodes=cls_nodes.shape[0], nodes=cls_nodes) #, test_mask=test_mask
#             data_list.append(new_data)
#         print()
#         return data_list
    
#     @property
#     def name(self):
#         return osp.split(self.root)[1]

#     @property
#     def raw_file_names(self):
#         return []

#     @property
#     def processed_file_names(self):
#         if self.num_parts == 1:
#             return [f'data.aug.{self.augmentation.method}.pt', "data.pth"]
#         else:
#             return [f'data.aug.{self.augmentation.method}.ip.{self.num_parts}.fp.{self.final_parts}.pt', "data.pth"]

#     @property
#     def model_dir(self):
#         return osp.join(self.root, "model")

#     @property
#     def result_dir(self):
#         return osp.join(self.root, "result")

#     @property
#     def dirs(self):
#         return [self.root, self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]


# class CustomGraphDataset(InMemoryDataset):
#     def __init__(self, root, graph_type='teacher_graph', transform=None, pre_transform=None):
#         """
#         :param root: Root directory where dataset is stored
#         :param graph_type: Specify which graph to load ('student_graph' or 'teacher_graph')
#         :param transform: Transformations to apply to data
#         :param pre_transform: Pre-transformations before processing
#         """
#         self.graph_type = graph_type  # Store which graph to load
#         super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['data.pth']  # Ensure your dataset is saved as 'data.pt' in the raw directory

#     @property
#     def processed_file_names(self):
#         return [f'{self.graph_type}.pth']  # Processed dataset is saved based on graph type

#     def download(self):
#         # Ensure the dataset exists
#         if not osp.exists(self.raw_paths[0]):
#             raise FileNotFoundError(f"Dataset not found at {self.raw_paths[0]}")

#     def process(self):
#         # Load raw dataset
#         raw_data_path = self.raw_paths[0]
#         raw_data = torch.load(raw_data_path)

#         if self.graph_type not in raw_data:
#             raise ValueError(f"Graph type '{self.graph_type}' not found in dataset. "
#                              f"Available keys: {list(raw_data.keys())}")

#         # Extract the desired graph
#         graph_data = raw_data[self.graph_type]

#         # Ensure it's wrapped in a list (InMemoryDataset expects a list)
#         if not isinstance(graph_data, list):
#             graph_data = [graph_data]

#         # Collate and save
#         data, slices = self.collate(graph_data)
#         torch.save((data, slices), self.processed_paths[0])

########################################################################################################################################
#Mine Produced:
class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", name='cora', num_parts=1, final_parts=1, augmentation=None, transform=None,
                 pre_transform=None, graph_type='teacher_graph'):
        self.num_parts = num_parts
        self.final_parts = final_parts
        self.augmentation = augmentation
        self.graph_type = graph_type
        super().__init__(root=osp.join(root, name), transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def download(self):
        utils.create_dirs(self.dirs)
        dataset = fetch_dataset(*osp.split(self.root))
        if self.graph_type == 'student_graph' or self.graph_type == 'teacher_graph':
            utils.create_masks(data=dataset.data[self.graph_type])
            edge_attr = torch.ones(dataset.data[self.graph_type].edge_index.shape[1]) if dataset.data[self.graph_type].edge_attr is None else dataset.data[self.graph_type].edge_attr
            dataset.data[self.graph_type].edge_attr = edge_attr
            # torch.save((dataset.data[self.graph_type], dataset.slices), self.processed_paths[1])
        else:
            utils.create_masks(data=dataset.data['teacher_graph'])
            edge_attr = torch.ones(dataset.data['teacher_graph'].edge_index.shape[1]) if dataset.data['teacher_graph'].edge_attr is None else dataset.data['teacher_graph'].edge_attr
            dataset.data['teacher_graph'].edge_attr = edge_attr
            dataset.data['student_graph'].edge_attr = edge_attr
        torch.save((dataset.data, dataset.slices), self.processed_paths[1])

    def process(self):
        """
        Process either a full batch or cluster data.

        :return:
        """
        data, _ = torch.load(self.processed_paths[1])
        if self.num_parts == 1:
            data_list = self.process_full_batch_data(data)
        else:
            data_list = self.process_cluster_data(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Processing full batch data")
        if self.graph_type == "teacher_graph" or self.graph_type == "student_graph":
            view1data = data[self.graph_type]
            view2data = view1data if self.augmentation is None else self.augmentation(view1data)

        elif self.graph_type == "both":
            view1data = data['teacher_graph']
            view2data = data['student_graph']
            # view2data = view2data if self.augmentation is None else self.augmentation(view2data)
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
        if diff > 0:
            """
            Data augmentation on the features could lead to mismatch between the shape of the two views,
            hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
            reflect on the original data)
            """
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)
        
        nodes = torch.tensor(np.arange(view1data.num_nodes), dtype=torch.long)
        data = Data(nodes=nodes, edge_index=view1data.edge_index, edge_index2=view2data.edge_index,
                    edge_attr=view1data.edge_attr,
                    edge_attr2=view2data.edge_attr, x=view1data.x, x2=view2data.x, y=view1data.y,
                    train_mask=view1data.train_mask,
                    val_mask=view1data.val_mask, num_nodes=view1data.num_nodes)#, test_mask=view1data.test_mask
        return [data]

    def process_cluster_data(self, data):
        """
        Data processing for ClusterSelfGNN. The data object is clustered according to `num_parts`,
        and clusters are randomly merged together. If `graph_type='both'`, teacher and student graphs
        will be clustered **identically**, ensuring matching clusters.
        
        :param data: A PyTorch Geometric Data object
        :return: a list of Data objects depending on the final number of clusters.
        """
        data_list = []
        num_parts, cluster_size = self.num_parts, self.num_parts // self.final_parts

        if self.graph_type in ["teacher_graph", "student_graph"]:
            clusters = []
            cd = ClusterData(data[self.graph_type], num_parts=num_parts)
            partptr = cd.partition.partptr  # Partition pointers
            perm = cd.partition.node_perm   # Node permutation for clustering

            for i in range(1, partptr.numel()):
                cls_nodes = perm[partptr[i - 1]: partptr[i]]
                clusters.append(cls_nodes)

            # Shuffle clusters randomly
            np.random.shuffle(clusters)

            # Merge clusters into larger partitions
            for i in tqdm(range(0, len(clusters), cluster_size), "Processing clusters"):
                end = min(i + cluster_size, len(clusters))
                cls_nodes = torch.cat(clusters[i:end]).unique()

                x = data[self.graph_type].x[cls_nodes]
                y = data[self.graph_type].y[cls_nodes]
                train_mask = data[self.graph_type].train_mask[cls_nodes]
                val_mask = data[self.graph_type].val_mask[cls_nodes]

                edge_index, edge_attr = subgraph(cls_nodes, data[self.graph_type].edge_index, relabel_nodes=True)
                view1data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=cls_nodes.shape[0])
                view2data = view1data if self.augmentation is None else self.augmentation(view1data)

                if not hasattr(view2data, "edge_attr") or view2data.edge_attr is None:
                    view2data.edge_attr = torch.ones(view2data.edge_index.shape[1])

                diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
                if diff > 0:
                    smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
                    smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
                    view1data.x = F.normalize(view1data.x)
                    view2data.x = F.normalize(view2data.x)

                new_data = Data(y=y, x=view1data.x, x2=view2data.x, edge_index=view1data.edge_index,
                                edge_index2=view2data.edge_index,
                                edge_attr=view1data.edge_attr, edge_attr2=view2data.edge_attr, train_mask=train_mask,
                                val_mask=val_mask, num_nodes=cls_nodes.shape[0], nodes=cls_nodes)
                data_list.append(new_data)

        elif self.graph_type == "both":
            clusters = []
            cd = ClusterData(data['teacher_graph'], num_parts=num_parts)  # Use teacher graph for clustering

            partptr = cd.partition.partptr  # Partition pointers
            perm = cd.partition.node_perm   # Node permutation for clustering

            for i in range(1, partptr.numel()):
                cls_nodes = perm[partptr[i - 1]: partptr[i]]
                clusters.append(cls_nodes)

            # Shuffle clusters the same way for both graphs
            indices = np.arange(len(clusters))
            np.random.shuffle(indices)
            clusters = [clusters[i] for i in indices]

            # Merge clusters into larger partitions
            for i in tqdm(range(0, len(clusters), cluster_size), "Processing clusters"):
                end = min(i + cluster_size, len(clusters))
                cls_nodes = torch.cat(clusters[i:end]).unique()

                # Extract data for teacher graph
                x_teacher = data['teacher_graph'].x[cls_nodes]
                y_teacher = data['teacher_graph'].y[cls_nodes]
                train_mask_teacher = data['teacher_graph'].train_mask[cls_nodes]
                val_mask_teacher = data['teacher_graph'].val_mask[cls_nodes]
                edge_index_teacher, edge_attr_teacher = subgraph(cls_nodes, data['teacher_graph'].edge_index, relabel_nodes=True)
                view1data = Data(edge_index=edge_index_teacher, x=x_teacher, edge_attr=edge_attr_teacher, num_nodes=cls_nodes.shape[0])

                # Extract data for student graph (same cluster nodes)
                x_student = data['student_graph'].x[cls_nodes]
                y_student = data['student_graph'].y[cls_nodes]
                train_mask_student = data['student_graph'].train_mask[cls_nodes]
                val_mask_student = data['student_graph'].val_mask[cls_nodes]
                edge_index_student, edge_attr_student = subgraph(cls_nodes, data['student_graph'].edge_index, relabel_nodes=True)
                view2data = Data(edge_index=edge_index_student, x=x_student, edge_attr=edge_attr_student, num_nodes=cls_nodes.shape[0])

                # Normalize and pad if necessary
                diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
                if diff > 0:
                    smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
                    smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
                    view1data.x = F.normalize(view1data.x)
                    view2data.x = F.normalize(view2data.x)

                # Create final data object
                new_data = Data(y=y_teacher, x=view1data.x, x2=view2data.x, edge_index=view1data.edge_index,
                                edge_index2=view2data.edge_index,
                                edge_attr=view1data.edge_attr, edge_attr2=view2data.edge_attr, train_mask=train_mask_teacher,
                                val_mask=val_mask_teacher, num_nodes=cls_nodes.shape[0], nodes=cls_nodes)
                data_list.append(new_data)

        print()
        return data_list
    
    @property
    def name(self):
        return osp.split(self.root)[1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.num_parts == 1:
            return [f'data.aug.{self.augmentation.method}.pt', "data.pth"]
        else:
            return [f'data.aug.{self.augmentation.method}.ip.{self.num_parts}.fp.{self.final_parts}.pt', "data.pth"]

    @property
    def model_dir(self):
        return osp.join(self.root, "model")

    @property
    def result_dir(self):
        return osp.join(self.root, "result")

    @property
    def dirs(self):
        return [self.root, self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        :param root: Root directory where dataset is stored
        :param transform: Transformations to apply to data
        :param pre_transform: Pre-transformations before processing
        """
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pth']  # Ensure your dataset is saved as 'data.pt' in the raw directory

    @property
    def processed_file_names(self):
        return ['processed.pth']  

    def download(self):
        # Ensure the dataset exists
        if not osp.exists(self.raw_paths[0]):
            raise FileNotFoundError(f"Dataset not found at {self.raw_paths[0]}")

    def process(self):
        # Load raw dataset
        raw_data_path = self.raw_paths[0]
        raw_data = torch.load(raw_data_path)

        graph_data = {}
        if 'teacher_graph' in raw_data:
            # Extract the desired graph
            graph_data['teacher_graph'] = raw_data['teacher_graph']

        if 'student_graph' in raw_data:
            # Extract the desired graph
            graph_data['student_graph'] = raw_data['student_graph']

        # Ensure it's wrapped in a list (InMemoryDataset expects a list)
        if not isinstance(graph_data, list):
            graph_data = [graph_data]

        # Collate and save
        data, slices = self.collate(graph_data)
        torch.save((data, slices), self.processed_paths[0])

def fetch_dataset(root, name):
    """
    Fetchs datasets from the PyTorch Geometric library
    
    :param root: A path to the root directory a dataset will be placed
    :param name: Name of the dataset. Currently, the following names are supported
                'cora', 'citeseer', "pubmed", 'Computers', "Photo", 'CS',  'Physics'
    :return: A PyTorch Geometric dataset
    """
    print(name.lower())
    if name.lower() in {'cora', 'citeseer', "pubmed"}:
        return Planetoid(root=root, name=name)
    elif name.lower() in {'computers', "photo"}:
        return Amazon(root=root, name=name)
    elif name.lower() in {'cs',  'physics'}:
        return Coauthor(root=root, name=name)
    elif name.lower() == "wiki":
        return WikiCS(osp.join(root, "WikiCS"))
    elif name.lower() == "actor":
        return Actor(osp.join(root, name))
    elif name.lower() == "custom":  # Add this block for your dataset
        print("Loading custom dataset...")
        return CustomGraphDataset(root=osp.join(root, "custom"), graph_type='teacher_graph')
    elif name.lower() == "imagenet100":  # Add this block for your dataset
        print("Loading custom dataset...")
        return CustomGraphDataset(root=osp.join(root, "Imagenet100"))
    elif name.lower() == "cifar10":  # Add this block for your dataset
        print("Loading custom dataset...")
        return CustomGraphDataset(root=osp.join(root, "Cifar10"))
    elif name.lower() == "imagenet1k":  # Add this block for your dataset
        print("Loading custom dataset...")
        return CustomGraphDataset(root=osp.join(root, "Imagenet1K"))
    else:
        raise ValueError(f"Dataset '{name}' not supported!")
