import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class CombinedModel(nn.Module):
    def __init__(self, backbone_model, graph_model, args):
        super(CombinedModel, self).__init__()
        self.backbone = backbone_model
        self.graph_model = graph_model
        self.args = args

        # Initialize any additional required components or state variables here
        # For example, features, nn_tensor, sim_tensor, etc.

    def forward(self, images, indices, same_im_bool, features, nn_tensor,
                sim_tensor, bootstrap_myself_tensor, graph,
                nn_matrix_cpu_flag, is_student=True, fp16_scaler=None):
        # Process images through the backbone model
        output = self.backbone(images)

        # Update graph and state
        graph, sims_knn_local, indices_knn_local, feats_local = update_graph(
            output, indices, features, graph, nn_matrix_cpu_flag, is_student, self.args
        )

        # Forward pass through the graph model
        device = graph.x.device
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

        # For the teacher model, use torch.no_grad()
        if not is_student:
            with torch.no_grad():
                node_embeddings, global_embedding = self.graph_model(
                    graph.x, graph.edge_index, batch
                )
        else:
            node_embeddings, global_embedding = self.graph_model(
                graph.x, graph.edge_index, batch
            )

        return (output, node_embeddings, global_embedding, graph, sims_knn_local, indices_knn_local, feats_local)


def update_graph(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, requires_grad, args):
    if not requires_grad:
        # Use torch.no_grad() only when gradients are not required (for teacher)
        with torch.no_grad():
            # output = output.detach()  # Ensure teacher output is detached
            # Rest of the code remains the same for teacher
            return _update_graph_inner(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, args)
    else:
        # For student, do not use torch.no_grad()
        return _update_graph_inner(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, args)


def _update_graph_inner(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, args):
    """
    Refactored to mirror the logic & style of 'update_graph_from_model_multi_gpu'.
    Returns: (graph, sims_knn_local, indices_knn_local, feats_local)
    """

    # === 1) Normalize Features ===
    feats_local = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
    if args.nn_rep_type == "mean":
        feats_local = feats_local.mean(dim=0)
        feats_local = F.normalize(feats_local, p=2, dim=-1)
    elif args.nn_rep_type == "first":
        feats_local = feats_local[0]
    elif args.nn_rep_type == "second":
        feats_local = feats_local[1]
    else:
        raise NotImplementedError("Invalid nn_rep_type for graph update.")

    # === 2) Compute kNN ===
    # feats_local: [batch_size, feat_dim]
    # features:    [num_nodes, feat_dim]
    similarity = feats_local @ features.T
    sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

    # === 3) Prepare Local Edge Updates (Vectorized) ===
    batch_size = indices_local.size(0)
    world_size = dist.get_world_size()

    # If we do NOT want to update edges (e.g. teacher might skip), set edge_counts increment to 0
    # so that the final index_copy_ won't change them.
    edge_increment = args.edges_per_node if teacher_nn_matrix_cpu_flag else 0

    # 3A) Compute reserved_indices_local
    start_indices_local = indices_local * args.edges_per_node * args.vote_nn_nb
    edge_positions_local = graph.edge_counts[indices_local].unsqueeze(1) \
                           + torch.arange(args.edges_per_node, device=indices_local.device).unsqueeze(0)
    edge_positions_local %= (args.edges_per_node * args.vote_nn_nb)
    reserved_indices_local = (start_indices_local.unsqueeze(1) + edge_positions_local).flatten()

    # # 3B) Build new_edges_local
    # #     src: indices_local repeated edges_per_node times (optionally + offset if you want len_train)
    # #     dst: top-K neighbor indices
    # src_nodes_local = indices_local.repeat_interleave(args.edges_per_node)  # + len_train if needed
    # dst_nodes_local = indices_knn_local[:, :args.edges_per_node].flatten()
    # new_edges_local = torch.stack([src_nodes_local, dst_nodes_local], dim=0)
    # # shape [2, batch_size * edges_per_node]

    valid_neighbors_list = []
    for i in range(batch_size):
        # row of candidate neighbors (shape ~ [edges_per_node+5])
        row_neighbors = indices_knn_local[i]
        # remove any that match the source node:
        row_neighbors = row_neighbors[row_neighbors != indices_local[i]]
        # keep only up to edges_per_node after removing self-loops
        row_neighbors = row_neighbors[: args.edges_per_node]
        valid_neighbors_list.append(row_neighbors)

    # 3) Stack into a [batch_size, edges_per_node] tensor
    valid_neighbors = torch.stack(valid_neighbors_list, dim=0)  # [batch_size, edges_per_node]

    # 4) Flatten to match the original shape
    dst_nodes_local = valid_neighbors.flatten()  # [batch_size * edges_per_node]

    # 5) src_nodes_local stays the same, but slice it if you want to match lengths exactly
    src_nodes_local = indices_local.repeat_interleave(args.edges_per_node)
    src_nodes_local = src_nodes_local[: dst_nodes_local.size(0)]  # safety if any row lost neighbors

    # 6) Finally, build new_edges_local
    new_edges_local = torch.stack([src_nodes_local, dst_nodes_local], dim=0)


    # 3C) Compute new edge_counts
    edge_counts_local = (graph.edge_counts[indices_local] + edge_increment) \
                        % (args.edges_per_node * args.vote_nn_nb)

    # === 4) All-Gather the Local Updates ===
    all_indices          = gather_all_gpus_graph(indices_local)         # shape [world_size * batch_size]
    all_feats            = gather_all_gpus_graph(feats_local)           # shape [world_size * batch_size, feat_dim]
    all_resv_indices     = gather_all_gpus_graph(reserved_indices_local)# shape [world_size * batch_size * edges_per_node]
    all_new_edges        = gather_all_gpus_graph(new_edges_local)       # shape [2 * world_size * batch_size * edges_per_node]
    all_edge_counts      = gather_all_gpus_graph(edge_counts_local)     # shape [world_size * batch_size]

    # Reshape `all_new_edges` to get [2, total_edges] with rank 0 block first, then rank 1, etc.
    all_new_edges = all_new_edges.view(world_size, 2, batch_size * args.edges_per_node)
    all_new_edges = all_new_edges.permute(1, 0, 2)   # => [2, world_size, batch_size*edges_per_node]
    all_new_edges = all_new_edges.reshape(2, -1)     # => [2, total_edges]

    # === 5) Apply Updates with index_copy_ on ALL GPUs ===
    # 5.1) Node features
    graph.x.index_copy_(0, all_indices, all_feats)

    # 5.2) edge_counts
    graph.edge_counts.index_copy_(0, all_indices, all_edge_counts)

    # 5.3) reserved_edge_index along dim=1
    graph.reserved_edge_index.index_copy_(1, all_resv_indices, all_new_edges)

    # 5.4) Update graph.edge_index
    graph.edge_index = graph.reserved_edge_index.clone()

    return graph, sims_knn_local, indices_knn_local, feats_local

def gather_all_gpus_graph(tensor):
    """
    Gathers `tensor` from all GPUs along dim=0 and concatenates.
    Assumes each GPU has the same shape for `tensor`.
    If shapes differ, you need a more advanced gather.
    """
    world_size = dist.get_world_size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)



# def _update_graph_inner(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, args):
#     """
#     Example code that skips self-loops by searching for the next neighbor
#     if the top-k neighbor equals the source node.
#     """

#     # === 1. Normalize features ===
#     feats_local = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
#     if args.nn_rep_type == "mean":
#         feats_local = feats_local.mean(dim=0)
#         feats_local = F.normalize(feats_local, p=2, dim=-1)
#     elif args.nn_rep_type == "first":
#         feats_local = feats_local[0]
#     elif args.nn_rep_type == "second":
#         feats_local = feats_local[1]
#     else:
#         raise NotImplementedError("Invalid nn_rep_type for graph update.")

#     # === 2. Compute extended kNN so we can skip self-loops ===
#     # We ask for k = edges_per_node + 1 neighbors to have an extra candidate.
#     batch_size = feats_local.shape[0]
#     world_size = dist.get_world_size()
#     k_for_topk = args.edges_per_node + 1
#     similarity = feats_local @ features.T  # [batch_size, num_nodes]

#     sims_knn_all, indices_knn_all = similarity.topk(dim=-1, k=k_for_topk)
#     # sims_knn_all, indices_knn_all each: [batch_size, (edges_per_node+1)]

#     # === 3. Filter out self-loops, keep only edges_per_node neighbors ===
#     indices_no_self, sims_no_self = [], []
#     for i in range(batch_size):
#         row_neighbors = indices_knn_all[i]  # shape [(edges_per_node+1)]
#         row_sims = sims_knn_all[i]
#         # Skip if neighbor == i-th sample in the local batch (self loop)
#         # 'i' here corresponds to feats_local[i], but note:
#         # 'indices_local[i]' is the actual "global node index" for row i.
#         # If you want to skip exactly if 'neighbor == indices_local[i]', do:
#         #   mask = (row_neighbors != indices_local[i])
#         # If your row i is already the global node, do:
#         mask = (row_neighbors != indices_local[i])

#         # Now pick only the first edges_per_node from the masked neighbors.
#         valid_neighbors = row_neighbors[mask][: args.edges_per_node]
#         valid_sims = row_sims[mask][: args.edges_per_node]

#         # If there's a chance you might not have enough neighbors after filtering,
#         # you could pad or handle that case. 
#         indices_no_self.append(valid_neighbors)
#         sims_no_self.append(valid_sims)

#     # Stack them back into [batch_size, edges_per_node]
#     indices_knn_local = torch.stack(indices_no_self, dim=0)
#     sims_knn_local = torch.stack(sims_no_self, dim=0)

#     # === 4. Proceed with your usual edge updates ===
#     # For instance, if teacher_nn_matrix_cpu_flag is True, update graph edge storage:
#     if teacher_nn_matrix_cpu_flag:
#         # Vectorized approach to update edges/counters
#         batch_sz = indices_local.size(0)

#         start_indices_local = indices_local * args.edges_per_node * args.vote_nn_nb
#         edge_positions_local = (
#             graph.edge_counts[indices_local].unsqueeze(1)
#             + torch.arange(args.edges_per_node, device=indices_local.device).unsqueeze(0)
#         )
#         edge_positions_local %= (args.edges_per_node * args.vote_nn_nb)
#         reserved_indices_local = (start_indices_local.unsqueeze(1) + edge_positions_local).flatten()

#         # Build src/dst
#         src_nodes_local = indices_local.repeat_interleave(args.edges_per_node)
#         dst_nodes_local = indices_knn_local.flatten()
#         new_edges_local = torch.stack([src_nodes_local, dst_nodes_local], dim=0)

#         # Update edge_counts
#         edge_counts_local = (
#             graph.edge_counts[indices_local] + args.edges_per_node
#         ) % (args.edges_per_node * args.vote_nn_nb)

#     all_indices          = gather_all_gpus_graph(indices_local)         # shape [world_size * batch_size]
#     all_feats            = gather_all_gpus_graph(feats_local)           # shape [world_size * batch_size, feat_dim]
#     all_resv_indices     = gather_all_gpus_graph(reserved_indices_local)# shape [world_size * batch_size * edges_per_node]
#     all_new_edges        = gather_all_gpus_graph(new_edges_local)       # shape [2 * world_size * batch_size * edges_per_node]
#     all_edge_counts      = gather_all_gpus_graph(edge_counts_local)     # shape [world_size * batch_size]

#     # Reshape `all_new_edges` to get [2, total_edges] with rank 0 block first, then rank 1, etc.
#     all_new_edges = all_new_edges.view(world_size, 2, batch_size * args.edges_per_node)
#     all_new_edges = all_new_edges.permute(1, 0, 2)   # => [2, world_size, batch_size*edges_per_node]
#     all_new_edges = all_new_edges.reshape(2, -1)     # => [2, total_edges]

#     # === 5) Apply Updates with index_copy_ on ALL GPUs ===
#     # 5.1) Node features
#     graph.x.index_copy_(0, all_indices, all_feats)

#     # 5.2) edge_counts
#     graph.edge_counts.index_copy_(0, all_indices, all_edge_counts)

#     # 5.3) reserved_edge_index along dim=1
#     graph.reserved_edge_index.index_copy_(1, all_resv_indices, all_new_edges)

#     # 5.4) Update graph.edge_index
#     graph.edge_index = graph.reserved_edge_index.clone()

#     return graph, sims_knn_local, indices_knn_local, feats_local


# def _update_graph_inner(output, indices_local, features, graph, teacher_nn_matrix_cpu_flag, args):
#     # Normalize features
#     feats_local = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
#     if args.nn_rep_type == "mean":
#         feats_local = feats_local.mean(dim=0)
#         feats_local = F.normalize(feats_local, p=2, dim=-1)
#     elif args.nn_rep_type == "first":
#         feats_local = feats_local[0]
#     elif args.nn_rep_type == "second":
#         feats_local = feats_local[1]
#     else:
#         raise NotImplemented

#     # Compute kNN
#     similarity = feats_local @ features.T
#     sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

#     if teacher_nn_matrix_cpu_flag:
#         # Initialize reserved edge storage if not already done
#         # if not hasattr(graph, "reserved_edge_index"):
#         #     num_nodes = graph.x.size(0)
#         #     max_edges_per_node = args.edges_per_node * args.vote_nn_nb
#         #     # Reserve space for edges: [2, num_nodes * max_edges_per_node]
#         #     graph.reserved_edge_index = torch.zeros(2, num_nodes * max_edges_per_node, dtype=torch.long, device=output.device)
#         #     graph.edge_counts = torch.zeros(num_nodes, dtype=torch.long, device=output.device)  # Track edge counts for each node

#         # Add new edges for each node
#         num_nodes = graph.x.size(0)
#         for i, src_node in enumerate(indices_local):
#             start_idx = src_node * args.edges_per_node * args.vote_nn_nb
#             end_idx = start_idx + args.edges_per_node * args.vote_nn_nb

#             # Get reserved slots for the node
#             reserved_edges = graph.reserved_edge_index[:, start_idx:end_idx]

#             # Add new edges to the reserved slots, avoiding self-loops
#             edge_pos = graph.edge_counts[src_node]  # Start from the current edge count
#             for dst_node in indices_knn_local[i, :args.edges_per_node]:
#                 if dst_node == src_node:  # Skip self-loops
#                     continue

#                 # Add the edge and increment the edge position
#                 reserved_edges[:, edge_pos % (args.edges_per_node * args.vote_nn_nb)] = torch.tensor([src_node, dst_node], device=output.device)
#                 edge_pos += 1

#             # Update edge count
#             graph.edge_counts[src_node] = edge_pos % (args.edges_per_node * args.vote_nn_nb)
#         # Update the graph's edge_index to reflect reserved edges
#         graph.edge_index = graph.reserved_edge_index.clone()

#         # Optionally, update node features
#         graph.x[indices_local] = feats_local

#     # Synchronize graph updates across all GPUs
#     graph = synchronize_graph(graph, indices_local)

#     return graph, sims_knn_local, indices_knn_local, feats_local


# def synchronize_graph(graph, indices_local):

    
#     # Each GPU creates a tensor with its rank
#     # rank = torch.distributed.get_rank()
#     # local_tensor = torch.tensor([rank], device=torch.device(f'cuda:{rank}'))

#     # # Gather data from all GPUs
#     # gathered_tensor = gather_all_gpus(local_tensor)

#     # if rank == 0:  # Only print on rank 0 to avoid duplicated output
#     #     print("Gathered tensor:", gathered_tensor.tolist())

#     """Synchronize graph state (edge_index, edge_counts, x) across all GPUs deterministically."""
#     world_size = torch.distributed.get_world_size()

#     # Gather global indices for the updated nodes
#     indices_global = gather_all_gpus(indices_local)
#     indices_global_split = torch.split(indices_global, indices_local.size(0), dim=0)

#     # Gather and update edge_index
#     edge_index_all = gather_all_gpus(graph.edge_index)
#     edge_index_split_size = edge_index_all.size(0) // world_size
#     edge_index_all_split = torch.split(edge_index_all, edge_index_split_size, dim=0)

#     # Filter edges based on global indices
#     for gpu_idx, global_indices in enumerate(indices_global_split):
#         edge_index_gpu = edge_index_all_split[gpu_idx]
#         graph.edge_index=edge_index_gpu

#     # Gather and update edge_counts
#     edge_counts_all = gather_all_gpus(graph.edge_counts)
#     edge_counts_split_size = edge_counts_all.size(0) // world_size
#     edge_counts_all_split = torch.split(edge_counts_all, edge_counts_split_size, dim=0)

#     for gpu_idx, global_indices in enumerate(indices_global_split):
#         graph.edge_counts[global_indices] = edge_counts_all_split[gpu_idx][global_indices]

#     # Gather and update x (node features)
#     x_all = gather_all_gpus(graph.x)
#     x_split_size = x_all.size(0) // world_size
#     x_all_split = torch.split(x_all, x_split_size, dim=0)

#     for gpu_idx, global_indices in enumerate(indices_global_split):
#         graph.x[global_indices] = x_all_split[gpu_idx][global_indices]

#     return graph


# def gather_all_gpus(local_tensor):
#     feats_all = torch.empty(dist.get_world_size(), *local_tensor.shape, dtype=local_tensor.dtype,
#                             device=local_tensor.device)
#     output_l = list(feats_all.unbind(0))
#     output_all_reduce = torch.distributed.all_gather(output_l, local_tensor, async_op=True)
#     output_all_reduce.wait()
#     return torch.cat(output_l)

# def _update_graph_inner(output, indices_local, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, same_im_bool, graph, teacher_nn_matrix_cpu_flag, args):
#     # Note Be careful about teacher_output[0]:
#     feats_local = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
#     if args.nn_rep_type == "mean":
#         feats_local = feats_local.mean(dim=0)
#         feats_local = F.normalize(feats_local, p=2, dim=-1)
#     elif args.nn_rep_type == "first":
#         feats_local = feats_local[0]
#     elif args.nn_rep_type == "second":
#         feats_local = feats_local[1]
#     else:
#         raise NotImplemented

#     # Compute knn
#     similarity = feats_local @ features.T
#     sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

#     # indices_batch_all = gather_all_gpus(indices_local)
#     # features_batch_all = gather_all_gpus(feats_local)
#     # indices_knn_batch_all = gather_all_gpus(indices_knn_local)
#     # sims_knn_batch_all = gather_all_gpus(sims_knn_local)

#     # # Convert feats_local to the same type as features
#     # feats_local = feats_local.to(features.dtype)
#     # features_batch_all = features_batch_all.to(features.dtype)
#     # sims_knn_batch_all = sims_knn_batch_all.to(sim_tensor.dtype)

#     # features.index_copy_(0, indices_batch_all, features_batch_all).cpu()
#     # nn_tensor.index_copy_(0, indices_batch_all, indices_knn_batch_all).cpu()
#     # sim_tensor.index_copy_(0, indices_batch_all, sims_knn_batch_all).cpu()
#     # bootstrap_myself_tensor[indices_batch_all.cpu()] = gather_all_gpus(same_im_bool).cpu()

#     # Ensure no self-loops are included in the edge construction
#     src_nodes = indices_local.unsqueeze(1).repeat(1, args.edges_per_node).flatten()
#     dst_nodes = indices_knn_local[:, :args.edges_per_node].clone()  # Clone for in-place modifications

#     # Replace self-loops with an additional neighbor
#     for i, neighbors in enumerate(dst_nodes):
#         for j, neighbor in enumerate(neighbors):
#             if neighbor == indices_local[i]:  # Self-loop detected
#                 # Replace with the next neighbor in the kNN list, if available
#                 for additional_neighbor in indices_knn_local[i, args.edges_per_node:]:
#                     if additional_neighbor not in dst_nodes[i]:  # Ensure it's not already included
#                         dst_nodes[i, j] = additional_neighbor
#                         break

#     edge_index_new = torch.stack([src_nodes, dst_nodes.flatten()], dim=0)

#     if teacher_nn_matrix_cpu_flag:
#         # Create edge indices
#         # src_nodes = indices_local.unsqueeze(1).repeat(1, args.edges_per_node).flatten()
#         # dst_nodes = indices_knn_local[:, :args.edges_per_node].flatten()
#         # edge_index_new = torch.stack([src_nodes, dst_nodes], dim=0)

#         # Update the graph's edge_index
#         graph.edge_index = torch.cat([graph.edge_index.to(edge_index_new.device), edge_index_new], dim=1)

#         # Implement fixed-size global edge buffer
#         # Define E_max (maximum number of edges)
#         num_nodes = graph.x.size(0)
#         E_max = num_nodes * args.edges_per_node * args.vote_nn_nb  # args.w is the desired average number of edges per node
#         # Check if total number of edges exceeds E_max
#         num_edges = graph.edge_index.size(1)
#         if num_edges > E_max:
#             # Number of edges to remove
#             num_edges_to_remove = num_edges - E_max
#             # Remove the oldest edges from the beginning
#             graph.edge_index = graph.edge_index[:, num_edges_to_remove:]

#         # Optionally, update node features in the graph
#         graph.x[indices_local] = feats_local#.requires_grad_(True)
#     return graph, sims_knn_local, indices_knn_local, feats_local