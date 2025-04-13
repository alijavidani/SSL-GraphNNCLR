import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from graph import GraphNet
# from main_adasim import writer

class AdaSimLoss(nn.Module):
    def __init__(self, out_dim, embed_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, writer=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim)) # Note
        # self.register_buffer("node_center", torch.zeros(1, embed_dim))
        # self.register_buffer("global_center", torch.zeros(1, embed_dim))

        # Warm-up for teacher temperature
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.args = args
        self.writer = writer

    def forward(self, student_output, teacher_output, epoch, it,
                # teacher_node_embeddings, teacher_global_embedding,
                # student_node_embeddings, student_global_embedding,
                indices):
        """
        Cross-entropy between softmax outputs of the teacher and student networks
        + Graph consistency loss.
        """
        # student_output, student_output_feat = student_output
        # teacher_output, teacher_output_feat = teacher_output
        # teacher_output = teacher_output.detach()

        total_loss = 0

        # 1. Graph consistency loss
        # gc_local_loss, gc_global_loss = self.graph_consistency_CE_loss(
        #     teacher_node_embeddings, teacher_global_embedding,
        #     student_node_embeddings, student_global_embedding,
        #     indices, epoch)

        # gc_local_loss2, gc_global_loss2 = self.graph_consistency_KL_loss(
        #     teacher_node_embeddings, teacher_global_embedding,
        #     student_node_embeddings, student_global_embedding,
        #     indices, epoch)
        
        # 2. AdaSim loss
        adasim_loss = self.adasim_loss(student_output, teacher_output, epoch)

        # 3. Combine both losses
        print(f'adasim_loss: {adasim_loss}') #adasim_loss: {adasim_loss}, 
       # print(f'gc_local_loss: {gc_local_loss}, gc_global_loss: {gc_global_loss}') #adasim_loss: {adasim_loss}, 
        # print(f'gc_local_loss2: {gc_local_loss}, gc_global_loss2: {gc_global_loss}') #adasim_loss: {adasim_loss}, 
        # self.writer.add_scalar('adasim_loss', adasim_loss, it)
        # self.writer.add_scalar('gc_local_loss', gc_local_loss, it)
        # self.writer.add_scalar('gc_global_loss', gc_global_loss, it)
        # self.writer.add_scalar('gc_local_loss2', gc_local_loss2, it)
        # self.writer.add_scalar('gc_global_loss2', gc_global_loss2, it)
        total_loss += adasim_loss 
        # total_loss += gc_local_loss #+ gc_local_loss2 # gc_global_loss +

        # 4. Update center
        self.update_centers(teacher_output,
                            #  teacher_node_embeddings, teacher_global_embedding
                        )

        return total_loss

    @torch.no_grad()
    def update_centers(self, teacher_output,
                        # teacher_node_embeddings, teacher_global_embedding
                        ):
        """
        Update center used for teacher output.
        """
        # Update output center
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        # Update node center
        # batch_node_center = teacher_node_embeddings.mean(dim=0, keepdim=True)
        # dist.all_reduce(batch_node_center)
        # batch_node_center = batch_node_center / dist.get_world_size()
        # self.node_center = self.node_center * self.center_momentum + batch_node_center * (1 - self.center_momentum)

        # # Update global center
        # batch_global_center = teacher_global_embedding.mean(dim=0, keepdim=True)
        # dist.all_reduce(batch_global_center)
        # batch_global_center = batch_global_center / dist.get_world_size()
        # self.global_center = self.global_center * self.center_momentum + batch_global_center * (1 - self.center_momentum)


    def adasim_loss(self, student_output, teacher_output, epoch):
        adasim_loss = 0
        
        # Student sharpening
        student_out = student_output / self.student_temp
        student_out = F.log_softmax(student_out, dim=-1)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        # Reshape student and teacher outputs
        proj_student = student_out.reshape(self.args.local_crops_number + 2, self.args.batch_size_per_gpu, -1).transpose(0, 1)
        reshaped_teacher = teacher_out.reshape(2, self.args.batch_size_per_gpu, -1).transpose(0, 1)

        # Global<->Global matching
        adasim_loss += (-proj_student[:, 0] * reshaped_teacher[:, 1]).mean(dim=0).sum()
        adasim_loss += (-proj_student[:, 1] * reshaped_teacher[:, 0]).mean(dim=0).sum()
        adasim_loss /= 2
        return adasim_loss
    
    def graph_consistency_CE_loss(self, teacher_node_embeddings, teacher_global_embedding,
                           student_node_embeddings, student_global_embedding,
                           indices, epoch):
        """
        Calculate local and global graph consistency loss using cross-entropy loss.
        """
        # Centering and sharpening for teacher outputs
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_node_embeddings = F.softmax(
            (teacher_node_embeddings - self.node_center) / teacher_temp, dim=-1
        )
        teacher_global_embedding = F.softmax(
            (teacher_global_embedding - self.global_center) / teacher_temp, dim=-1
        )

        # Student outputs
        student_node_embeddings = F.log_softmax(student_node_embeddings / self.student_temp, dim=-1)
        student_global_embedding = F.log_softmax(student_global_embedding / self.student_temp, dim=-1)

        # Compute local consistency loss (cross-entropy)
        loss_local = - (teacher_node_embeddings[indices] * student_node_embeddings[indices]).sum(dim=-1).mean()

        # Compute global consistency loss (cross-entropy)
        loss_global = - (teacher_global_embedding * student_global_embedding).sum(dim=-1).mean()

        return loss_local, loss_global


    def graph_consistency_KL_loss(self, teacher_node_embeddings, teacher_global_embedding,
                        student_node_embeddings, student_global_embedding,
                        indices, epoch):
        """
        Calculate local and global graph consistency loss using PyTorch Geometric graphs.
        """

        # Centering and sharpening for teacher outputs
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_node_embeddings = F.softmax((teacher_node_embeddings - self.node_center) / teacher_temp, dim=-1)
        teacher_global_embedding = F.softmax((teacher_global_embedding - self.global_center) / teacher_temp, dim=-1)

        # Student outputs
        student_node_embeddings = F.log_softmax(student_node_embeddings / self.student_temp, dim=-1)
        student_global_embedding = F.log_softmax(student_global_embedding / self.student_temp, dim=-1)

        # Compute local consistency loss
        loss_local = F.kl_div(
            student_node_embeddings[indices],
            teacher_node_embeddings[indices],
            reduction='batchmean',
            log_target=False
        )

        # Compute global consistency loss
        loss_global = F.kl_div(
            student_global_embedding,
            teacher_global_embedding,
            reduction='batchmean',
            log_target=False
        )

        return loss_local, loss_global

        
    def my_previous_loss(self, teacher_graph, student_graph, indices, epoch):
        """
        Calculate local and global graph consistency loss.
        """
        # Ensure that graphs are on the correct device
        device = teacher_graph.edge_index.device

        # Batch tensor for global pooling (assuming all nodes are in one graph)
        batch = torch.zeros(teacher_graph.num_nodes, dtype=torch.long, device=device)

        teacher_graph.to(device)
        student_graph.to(device)

        # Forward pass through the graph model for both teacher and student
        # teacher_node_embeddings, teacher_global_embedding = self.teacher_graph_model(
        #     teacher_graph.x, teacher_graph.edge_index, batch)
        student_node_embeddings, student_global_embedding = self.student_graph_model(
            student_graph.x, student_graph.edge_index, batch)
        # Forward pass through the teacher graph model (with no grad)
        with torch.no_grad():
            teacher_node_embeddings, teacher_global_embedding = self.teacher_graph_model(
                teacher_graph.x, teacher_graph.edge_index, batch)
            
        # Local consistency loss (cross-entropy)
        student_logits = F.log_softmax(student_node_embeddings[indices, :], dim=-1)
        teacher_embs = teacher_node_embeddings[indices, :]
        loss_local = -(teacher_embs * student_logits).sum(dim=-1).mean()

        # Global consistency loss (cross-entropy between global embeddings)
        loss_fn = nn.CrossEntropyLoss()
        loss_global = loss_fn(student_global_embedding, teacher_global_embedding)

        # Update centers
        self.update_centers(teacher_node_embeddings, teacher_global_embedding)

        return loss_local, loss_global