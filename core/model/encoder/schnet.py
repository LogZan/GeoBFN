import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding, Sigmoid
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from core.utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron
from ..common_model import GaussianSmearing, MLP, outer_product
from torch_scatter import scatter_softmax, scatter_sum

from .cross_vit import CrossViT

# class GaussianSmearing(torch.nn.Module):
#     def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
#         super(GaussianSmearing, self).__init__()
#         offset = torch.linspace(start, stop, num_gaussians)
#         self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
#         self.register_buffer("offset", offset)

#     def forward(self, dist):
#         dist = dist.view(-1, 1) - self.offset.view(1, -1)
#         return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):
    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer("freq_k", torch.arange(1, num_basis_k + 1).float())
        self.register_buffer("freq_l", torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis_k)
        c = torch.cos(
            angle.view(-1, 1) * self.freq_l.view(1, -1)
        )  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):
    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer("freq_k", torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = (
                C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
            )  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)
        #W = self.nn(edge_attr)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        edge_channels=100,
        cutoff=10.0,
        smooth=False,
        embedding=False,
        edge_emb=None,
        edge_activation="ReLU"
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.embedding = embedding
        n_heads = 16
        edge_feat_dim = 4
        num_r_gaussian = 20
        norm=True
        self.r_min = 0.0
        self.r_max = 10.0

        if self.embedding:
            self.node_emb = Embedding(100, hidden_channels, max_norm=10.0)

        if edge_emb is not None:
            self.edge_emb = edge_emb
            self.edge_cat = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels *2, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels))
            self.edge_d_emb = MultiLayerPerceptron(
                    1, 
                    [hidden_channels, hidden_channels], 
                    activation=edge_activation
                    )

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth
            )
            self.interactions.append(block)

        input_edge = hidden_channels * 2 + edge_channels
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1, bias=False),
        )

        self.h2x_layers = BaseH2XAttLayer(hidden_channels, hidden_channels, hidden_channels, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=edge_activation, norm=norm,
                                ew_net_type=None)
        
        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.cross_vit = CrossViT(image_size=192, num_classes=1, sm_dim=257, lg_dim=256)

    @classmethod
    def from_config(cls, config):
        if config.edge_emb:
            from edge import MLPEdgeEncoder
            edge_emb = MLPEdgeEncoder(config.hidden_dim, config.mlp_act)
        else:
            edge_emb = None

        #print(f"hidden_channels:{hidden_channels}")
        #print(f"num_filters:{hidden_channels}")
        #print(f"num_interactions:{config.num_convs}")
        #print(f"cutoff:{config.cutoff}")
        #print(f"smooth:{config.smooth_conv}")
        #print(f"embedding:{False}")
        #print(f"edge_emb:{edge_emb}")
        #print(f"edge_activation:{config.mlp_act}")

        encoder = cls(
                hidden_channels=config.hidden_dim,
                num_filters=config.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=config.hidden_dim,
                cutoff=config.cutoff,
                smooth=config.smooth_conv,
                embedding=False,
                edge_emb=edge_emb,
                edge_activation=config.mlp_act
                )
        return encoder
    
    def coord2diff(self, x, edge_index, norm_constant=1):
        row, col = edge_index
        coord_diff = x[row] - x[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + norm_constant)
        return radial, coord_diff
    
    def unsorted_segment_sum(
        self, data, segment_ids, num_segments, normalization_factor=1, aggregation_method='sum'
    ):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
        """
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        if aggregation_method == "sum":
            result = result / normalization_factor

        if aggregation_method == "mean":
            norm = data.new_zeros(result.shape)
            norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
            norm[norm == 0] = 1
            result = result / norm
        return result

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        trans = coord_diff * self.coord_mlp(input_tensor)
        agg = self.unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
        )
        coord = coord + agg
        return coord

    def forward(
        self, z, edge_index, edge_length, edge_attr=None, embed_node=False, **kwargs
    ):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long and self.embedding
            h = self.node_emb(z)
        else:
            h = z

        if edge_attr is None:
            if hasattr(kwargs, "edge_type"):
                edge_type_r, edge_type_p = kwargs["edge_type"]
                edge_emb_r = self.edge_emb(edge_type_r) 
                edge_emb_p = self.edge_emb(edge_type_p) 
                edge_d_emb = self.edge_d_emb(edge_length)
                edge_attr = self.edge_cat(
                        torch.cat(
                            [edge_d_emb * edge_emb_r, edge_d_emb * edge_emb_p],
                            -1)
                        )
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h
    

    def forward_update_pos(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long and self.embedding
            h = self.node_emb(z)
        else:
            h = z

        if edge_attr is None:
            if hasattr(kwargs, "edge_type"):
                edge_type_r, edge_type_p = kwargs["edge_type"]
                edge_emb_r = self.edge_emb(edge_type_r) 
                edge_emb_p = self.edge_emb(edge_type_p) 
                edge_d_emb = self.edge_d_emb(edge_length)
                edge_attr = self.edge_cat(
                        torch.cat(
                            [edge_d_emb * edge_emb_r, edge_d_emb * edge_emb_p],
                            -1)
                        )
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

            _, coord_diff = self.coord2diff(pos, edge_index)
            pos = self.coord_model(h, pos, edge_index, coord_diff, edge_attr)
        
        return h, pos
    

    def forward_cross_vit(
        self, z, edge_index, edge_length, edge_attr=None, embed_node=False, pos=None, time=None, rxnfp=None, segment_ids=None, **kwargs
    ):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long and self.embedding
            h = self.node_emb(z)
        elif time is not None:
            h = torch.cat([z, time], dim=-1)
            edge_attr = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), 1, device=edge_attr.device)], dim=-1)
        else:
            h = z

        if edge_attr is None:
            if hasattr(kwargs, "edge_type"):
                edge_type_r, edge_type_p = kwargs["edge_type"]
                edge_emb_r = self.edge_emb(edge_type_r) 
                edge_emb_p = self.edge_emb(edge_type_p) 
                edge_d_emb = self.edge_d_emb(edge_length)
                edge_attr = self.edge_cat(
                        torch.cat(
                            [edge_d_emb * edge_emb_r, edge_d_emb * edge_emb_p],
                            -1)
                        )
        
        x = pos
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
            h = self.cross_vit(h, rxnfp, segment_ids)

            _, coord_diff = self.coord2diff(pos, edge_index)
            pos = self.coord_model(h, pos, edge_index, coord_diff, edge_attr)
        
        return h, pos


class BaseH2XAttLayer(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = Sequential(Linear(r_feat_dim, 1), Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]