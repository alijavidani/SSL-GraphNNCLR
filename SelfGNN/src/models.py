from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GCN2Conv

import torch.nn.functional as F
import torch.nn as nn
import torch

from functools import wraps
import copy

torch.manual_seed(0)

"""
The following code is borrowed from BYOL

=====================Start=================
"""


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


"""
=====================End====================
"""

class Normalize(nn.Module):
    def __init__(self, dim=None, method="batch"):
        super().__init__()
        self.method = method
        if method == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif method == "layer":
            self.norm = nn.LayerNorm(dim)
        elif method == "pair":
            self.norm = self._pairnorm
        else:  # 'no' or None → Identity
            self.norm = lambda x: x

    def forward(self, x):
        return self.norm(x)

    def _pairnorm(self, x, scale=1.0):
        # PairNorm "PN" mode (scale after centering)
        mean = x.mean(dim=0, keepdim=True)  # [1, d]
        x = x - mean
        norm = x.norm(p=2, dim=1, keepdim=True) + 1e-6  # [n, 1]
        x = scale * x / norm.mean()
        return x
    

class Encoder(nn.Module):
    def __init__(self, layer_config, gnn_type, dropout=None, jk_mode="last", residual=False, **kwargs):
        super().__init__()
        self.jk_mode = jk_mode
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.residual = residual
        self.project = kwargs.get("prj_head_norm", False)

        rep_dim = layer_config[-1]
        if self.jk_mode == "concat":
            rep_dim = rep_dim * len(layer_config[1:])

        self.stacked_gnn = get_encoder(layer_config=layer_config, gnn_type=gnn_type, **kwargs)
        self.encoder_norm = Normalize(dim=rep_dim, method=kwargs["encoder_norm"])

        if self.project != "no":
            self.projection_head = nn.Sequential(
                nn.Linear(rep_dim, rep_dim),
                Normalize(dim=rep_dim, method=kwargs["prj_head_norm"]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

    def forward(self, x, edge_index, edge_weight=None):
        if self.gnn_type == "gcnii":
            # GCNII has its own internal stack
            x = self.stacked_gnn[0](x, edge_index)
            x = self.encoder_norm(x)
            return x, (self.projection_head(x) if self.project != "no" else None)
        layer_outputs = []

        for i, gnn in enumerate(self.stacked_gnn):
            x_prev = x
            if self.gnn_type in ["gat", "sage"]:
                x = gnn(x, edge_index)
            else:
                x = gnn(x, edge_index)

            # Add residual connection if enabled and dimensions match
            if self.residual and x.shape == x_prev.shape:
                x = x + x_prev

            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Jumping Knowledge
        if self.jk_mode == "concat":
            x = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == "max":
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == "sum":
            x = sum(layer_outputs)
        elif self.jk_mode == "mean":
            x = torch.stack(layer_outputs, dim=0).mean(dim=0)
        else:
            x = layer_outputs[-1]

        x = self.encoder_norm(x)
        return x, (self.projection_head(x) if self.project != "no" else None)


class SelfGNN(nn.Module):

    def __init__(self, layer_config, dropout=0.0, moving_average_decay=0.99, gnn_type='gcn', jk_mode="last", mask_ratio=0.3,feature_masking_weight=0.1, **kwargs):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.feature_masking_weight = feature_masking_weight

        # Adjust rep_dim if using Jumping Knowledge with concat mode
        rep_dim = layer_config[-1]
        if jk_mode == "concat":
            rep_dim = rep_dim * (len(layer_config) - 1)

        self.student_encoder = Encoder(
            layer_config=layer_config,
            gnn_type=gnn_type,
            dropout=dropout,
            jk_mode=jk_mode,
            **kwargs
        )

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.student_predictor = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            Normalize(dim=rep_dim, method=kwargs["prd_head_norm"]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def encode(self, x, edge_index, edge_weight=None, encoder=None):
        encoder = self.student_encoder if encoder is None else encoder
        encoder.train(self.training)
        return encoder(x, edge_index, edge_weight)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None, inference_mode=False):
        v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v1_rep, v1_student = v1_enc if v1_enc[1] is not None else (v1_enc[0], v1_enc[0])
        v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
        v2_rep, v2_student = v2_enc if v2_enc[1] is not None else (v2_enc[0], v2_enc[0])

        # If in inference mode, skip loss computation and return representations only
        if inference_mode:
            return v1_rep, v2_rep, None

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1, encoder=teacher_encoder)
            v1_teacher = v1_enc[1] if v1_enc[1] is not None else v1_enc[0]
            v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2, encoder=teacher_encoder)
            v2_teacher = v2_enc[1] if v2_enc[1] is not None else v2_enc[0]

        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())
        # total_loss = (loss1 + loss2).mean()

        loss = (loss1 + loss2).mean()

        if self.feature_masking_weight > 0:
            # Apply feature masking to x2 (you can also mask x1 if you want)
            x2_masked = self.apply_feature_mask(x2, self.mask_ratio)
            
            # Encode the masked view
            v2_masked_enc = self.encode(x=x2_masked, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
            _, v2_masked_student = v2_masked_enc if v2_masked_enc[1] is not None else (v2_masked_enc[0], v2_masked_enc[0])
            
            # Predict using the student predictor
            v2_masked_pred = self.student_predictor(v2_masked_student)
            
            # Compute loss between original unmasked student view (v1_pred) and masked one
            mask_loss = loss_fn(v1_pred, v2_masked_pred).mean()
            
            # Add to total loss
            loss = loss + self.feature_masking_weight * mask_loss
        return v1_rep, v2_rep, loss


    def apply_feature_mask(self, x, mask_ratio):
        B, D = x.shape
        device = x.device
        mask = torch.ones_like(x, device=device)
        num_mask = int(mask_ratio * D)

        for i in range(B):
            mask_idx = torch.randperm(D, device=device)[:num_mask]
            mask[i, mask_idx] = 0
        return x * mask

    def compute_feature_masking_loss(self, z1, z2):
        """
        Encourages feature consistency by penalizing variation across views for the same node.
        """
        return F.mse_loss(z1, z2)



def build_gin_mlp(in_dim, out_dim, hidden_dim=None):
    hidden_dim = hidden_dim or out_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )

class GCNIINet(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, alpha=0.1, theta=0.5, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))  # input layer

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=layer + 1, shared_weights=True, normalize=False))

        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        x_0 = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x_0)
        x = F.relu(x)
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x

def get_encoder(layer_config, gnn_type, **kwargs):
    """
    Builds the GNN backbone as required
    """
    if gnn_type == "gcn":
        return nn.ModuleList([GCNConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "sage":
        return nn.ModuleList([SAGEConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "gin":
        return nn.ModuleList([
            GINConv(build_gin_mlp(layer_config[i-1], layer_config[i]))
            for i in range(1, len(layer_config))
        ])
    elif gnn_type == "gat":
        heads = kwargs['heads'] if 'heads' in kwargs else [8] * len(layer_config)
        concat = kwargs['concat'] if 'concat' in kwargs else True
        return nn.ModuleList([
            GATConv(layer_config[i-1], layer_config[i] // heads[i-1], heads=heads[i-1], concat=concat)
            for i in range(1, len(layer_config))
        ])
    elif gnn_type == "gcnii":
        # layer_config = [input_dim, hidden_dim, num_layers] or [in, hidden, hidden, ..., hidden]
        in_channels = layer_config[0]
        hidden_channels = layer_config[1]
        num_layers = len(layer_config) - 1  # num of GCNII layers
        alpha = kwargs.get("alpha", 0.1)
        theta = kwargs.get("theta", 0.5)
        return nn.ModuleList([GCNIINet(num_layers, in_channels, hidden_channels, alpha=alpha, theta=theta, dropout=kwargs.get("dropout", 0.0))])

    
class LogisticRegression(nn.Module):
    """
    A logistic regression classifier for evaluating SelfGNN 
    """
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss