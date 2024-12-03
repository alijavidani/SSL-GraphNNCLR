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
            output, indices, features, nn_tensor,
            sim_tensor, bootstrap_myself_tensor, same_im_bool,
            graph, nn_matrix_cpu_flag, is_student, self.args
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


def update_graph(output, indices_local, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, same_im_bool, graph, teacher_nn_matrix_cpu_flag, requires_grad, args):
    if not requires_grad:
        # Use torch.no_grad() only when gradients are not required (for teacher)
        with torch.no_grad():
            # output = output.detach()  # Ensure teacher output is detached
            # Rest of the code remains the same for teacher
            return _update_graph_inner(output, indices_local, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, same_im_bool, graph, teacher_nn_matrix_cpu_flag, args)
    else:
        # For student, do not use torch.no_grad()
        return _update_graph_inner(output, indices_local, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, same_im_bool, graph, teacher_nn_matrix_cpu_flag, args)


def _update_graph_inner(output, indices_local, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, same_im_bool, graph, teacher_nn_matrix_cpu_flag, args):
    # Note Be careful about teacher_output[0]:
    feats_local = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
    if args.nn_rep_type == "mean":
        feats_local = feats_local.mean(dim=0)
        feats_local = F.normalize(feats_local, p=2, dim=-1)
    elif args.nn_rep_type == "first":
        feats_local = feats_local[0]
    elif args.nn_rep_type == "second":
        feats_local = feats_local[1]
    else:
        raise NotImplemented

    # Compute knn
    similarity = feats_local @ features.T
    sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

    # indices_batch_all = gather_all_gpus(indices_local)
    # features_batch_all = gather_all_gpus(feats_local)
    # indices_knn_batch_all = gather_all_gpus(indices_knn_local)
    # sims_knn_batch_all = gather_all_gpus(sims_knn_local)

    # # Convert feats_local to the same type as features
    # feats_local = feats_local.to(features.dtype)
    # features_batch_all = features_batch_all.to(features.dtype)
    # sims_knn_batch_all = sims_knn_batch_all.to(sim_tensor.dtype)

    # features.index_copy_(0, indices_batch_all, features_batch_all).cpu()
    # nn_tensor.index_copy_(0, indices_batch_all, indices_knn_batch_all).cpu()
    # sim_tensor.index_copy_(0, indices_batch_all, sims_knn_batch_all).cpu()
    # bootstrap_myself_tensor[indices_batch_all.cpu()] = gather_all_gpus(same_im_bool).cpu()

    if teacher_nn_matrix_cpu_flag:
        # Create edge indices
        src_nodes = indices_local.unsqueeze(1).repeat(1, args.edges_per_node).flatten()
        dst_nodes = indices_knn_local[:, :args.edges_per_node].flatten()
        edge_index_new = torch.stack([src_nodes, dst_nodes], dim=0)

        # Update the graph's edge_index
        graph.edge_index = torch.cat([graph.edge_index.to(edge_index_new.device), edge_index_new], dim=1)

        # Implement fixed-size global edge buffer
        # Define E_max (maximum number of edges)
        num_nodes = graph.x.size(0)
        E_max = num_nodes * args.edges_per_node * args.vote_nn_nb  # args.w is the desired average number of edges per node
        # Check if total number of edges exceeds E_max
        num_edges = graph.edge_index.size(1)
        if num_edges > E_max:
            # Number of edges to remove
            num_edges_to_remove = num_edges - E_max
            # Remove the oldest edges from the beginning
            graph.edge_index = graph.edge_index[:, num_edges_to_remove:]

        # Optionally, update node features in the graph
        graph.x[indices_local] = feats_local#.requires_grad_(True)
    return graph, sims_knn_local, indices_knn_local, feats_local