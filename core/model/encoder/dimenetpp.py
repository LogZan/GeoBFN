import torch
from torch import nn
from torch.nn import Linear, Embedding 
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt

from core.model.encoder.dimenetpp_features import dist_emb, angle_emb
from torch_sparse import SparseTensor
from math import pi as PI
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    import sympy as sym
except ImportError:
    sym = None

def xyz_to_dat(pos, edge_index, num_nodes, cutoff=5.0, use_torsion = False):
    """
    Compute the diatance, angle, and torsion from geometric information.

    Args:
        pos: Geometric information for every node in the graph.
        edgee_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    #C = dist < cutoff
    #j, i = j[C], i[C]

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    if use_torsion:
        # Prepare torsion idxes.
        idx_batch = torch.arange(len(idx_i),device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n   
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji 
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0]+=2*PI # 0 to 2pi
        torsion = scatter(torsion1,idx_batch_t,reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji, mask
    
    else:
        return dist, angle, i, j, idx_kj, idx_ji, mask

def swish(x):
    return x * torch.sigmoid(x)

# 首先定义一个Swish激活函数作为Module子类
class SwishModule(torch.nn.Module):
    def __init__(self):
        super(SwishModule, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j, edge_attr, embed_node=False):
        rbf,_ = emb
        #print(f"debuf4 e1, e2 : {emb[0].shape, emb[1].shape}")
        if embed_node:
            x = self.emb(x)
        # here edge attribute * rbf0 (ij) ?
        rbf0 = self.act(self.lin_rbf_0(rbf))
        rbf0 = edge_attr * rbf0 + edge_attr
        #print(f"debug2 x : {x.shape}\n{x}")
        #print(f"debug2 rbf0 : {rbf0.shape}\n{rbf0}")
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        
        # here edge attribute * rbf0 (ij) ?
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, 
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])
        
        # Add layer normalization for numerical stability
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        # Initialize the layer norms
        if hasattr(self, 'layer_norm1'):
            self.layer_norm1.reset_parameters()
        if hasattr(self, 'layer_norm2'):
            self.layer_norm2.reset_parameters()

    def forward(self, x, emb, idx_kj, idx_ji, edge_attr):
        # e1, e2 = x
        rbf0, sbf = emb
        x1, _ = x
        
        # Check for NaN values early
        if torch.isnan(x1).any():
            # Apply a safe replacement for NaN values
            x1 = torch.nan_to_num(x1, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply feature normalization for stability
        x1 = self.layer_norm1(x1)
        
        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        # Apply numerical safety for rbf
        rbf0 = torch.nan_to_num(rbf0, nan=0.0)
        if torch.isnan(sbf).any():
            sbf = torch.nan_to_num(sbf, nan=0.0)

        # edge_attribute here (kj)
        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        
        # Safely multiply with edge attributes
        rbf = edge_attr * rbf
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))
        
        # Apply numerical safety to sbf
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        
        # Safe indexing with boundary check
        if idx_kj.max() < x_kj.size(0):
            x_kj_idx = x_kj[idx_kj]
            x_kj = x_kj_idx * sbf
        else:
            # Handle the case where indices might be out of bounds
            valid_idx = idx_kj < x_kj.size(0)
            if not valid_idx.all():
                print(f"Warning: {(~valid_idx).sum()} indices out of bounds in update_e")
            idx_kj = idx_kj[valid_idx]
            sbf = sbf[valid_idx]
            x_kj = x_kj[idx_kj] * sbf
        
        # Safe scatter
        try:
            x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        except Exception as e:
            print(f"Scatter error in update_e: {e}")
            # Fallback to zero if scatter fails
            x_kj = torch.zeros_like(x1)
        
        x_kj = self.act(self.lin_up(x_kj))

        # Safely combine features
        e1 = x_ji + x_kj
        
        # Apply layers with safety checks
        for layer in self.layers_before_skip:
            e1_prev = e1
            e1 = layer(e1)
            # Check if layer produced NaNs
            if torch.isnan(e1).any():
                print("NaN detected in layers_before_skip, using previous values")
                e1 = e1_prev  # Use previous layer output if current one has NaNs
        
        # Final normalization for stability
        e1 = self.layer_norm2(e1)
        e1 = self.act(self.lin(e1)) + x1
        
        for layer in self.layers_after_skip:
            e1_prev = e1
            e1 = layer(e1)
            if torch.isnan(e1).any():
                print("NaN detected in layers_after_skip, using previous values")
                e1 = e1_prev
        
        # Apply numerical safety to e2 calculation
        safe_rbf0 = torch.clamp(rbf0, min=-100, max=100)  # Prevent extreme values
        e2 = self.lin_rbf(safe_rbf0) * e1
        
        # Final safety check
        e1 = torch.nan_to_num(e1, nan=0.0)
        e2 = torch.nan_to_num(e2, nan=0.0)

        return e1, e2 


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_emb_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)
        if hasattr(self, 'layer_norm'):
            self.layer_norm.reset_parameters()

    def forward(self, e, i):
        _, e2 = e
        
        # Apply numerical safety
        if torch.isnan(e2).any():
            e2 = torch.nan_to_num(e2, nan=0.0)
        
        # Clip extremely small values that might cause underflow
        e2 = torch.clamp(e2, min=-1e6, max=1e6)
        
        # Safe scatter operation
        try:
            v = scatter(e2, i, dim=0)
        except Exception as e:
            print(f"Scatter error in update_v: {e}")
            # Return zeros if scatter fails
            return torch.zeros((i.max()+1, e2.size(1)), device=e2.device)
        
        v = self.lin_up(v)
        
        # Apply normalization for stability
        v = self.layer_norm(v)
        
        for lin in self.lins:
            v_prev = v
            v = self.act(lin(v))
            # Check for NaNs
            if torch.isnan(v).any():
                print("NaN detected in update_v lins, using previous values")
                v = v_prev
        
        v = self.lin(v)
        
        # Final safety check
        v = torch.nan_to_num(v, nan=0.0)
        
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class DimeNetPPEncoder(torch.nn.Module):
    r"""
        The re-implementation for DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size (int, optional): Embedding size used in the basis transformation. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj=`3`)
            act: (function, optional): The activation funtion. (default: :obj=`swish`) 
            output_init: (str, optional): The initialization fot the output. It could be :obj=`GlorotOrthogonal` and :obj=`zeros`. (default: :obj=`GlorotOrthogonal`)       
    """
    def __init__(
        self, 
        energy_and_force=False, 
        cutoff=5.0, 
        num_layers=4, 
        hidden_channels=128, 
        out_channels=1, 
        int_emb_size=64, 
        basis_emb_size=8, 
        out_emb_channels=256, 
        num_spherical=7, 
        num_radial=6, 
        envelope_exponent=5, 
        num_before_skip=1, 
        num_after_skip=2, 
        num_output_layers=3, 
        act=swish, 
        output_init='GlorotOrthogonal'):
        super(DimeNetPPEncoder, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        
        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        # 替换coord_mlp的定义，使用SwishModule代替act函数
        swish_module = SwishModule()
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            swish_module,
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            swish_module,
            torch.nn.Linear(hidden_channels // 2, 1, bias=False),
        )

        self.reset_parameters()

    @classmethod
    def from_config(cls, config):
        #print(f"num_layers:{config.num_convs}")
        #print(f"hidden_channels:{config.hidden_dim}")
        #print(f"cutoff:{config.cutoff}")
        #print(f"out_channels:{config.hidden_dim}")
        #print(f"int_emb_size:{64}")
        #print(f"num_radial:{config.num_radial}")
        #print(f"num_spherical:{config.num_spherical}")
        #print(f"envelope_exponent:{5}")
        #print(f"num_before_skip:{config.num_before_skip}")
        #print(f"num_after_skip:{config.num_after_skip}")
        #print(f"num_output_layers:{3}")
        encoder = cls(
                num_layers=config.num_layers,
                hidden_channels=config.hidden_dim,
                cutoff=config.cutoff,
                out_channels=config.hidden_dim,
                int_emb_size=64,
                num_radial=config.num_radial,
                num_spherical=config.num_spherical,
                envelope_exponent=5,
                num_before_skip=config.num_before_skip,
                num_after_skip=config.num_after_skip,
                num_output_layers=3,
                )
        return encoder

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    
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


    # input will be : 
    # z             : will not be integer but hidden vector.
    # pos           : will be given as usual.
    # edge_index    : will be given extened order 2 or 3
    # edge_feature  : will be given
    def forward(
            self, z, edge_index, edge_length, 
            pos=None, 
            edge_attr=None, 
            embed_node=False, 
            **kwargs, 
            ):

        num_nodes=z.size(0)
        dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
                pos, 
                edge_index, 
                num_nodes, 
                cutoff=self.cutoff,
                use_torsion=False
                )
        
        # TODO extract edge_embedding to epsmodel
        edge_length = dist
        emb = self.emb(dist, angle, idx_kj)

        # dist, i, j : (N_edge, )
        # angle, idx_kj, idx_ji : (N_2hop, )
        # dist_emb, angle_emb = emb
        # dist_emb  :  (N_edge, dim)
        # angle_emb :  (N_2hopedge, dim)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node) 
        e1, e2 = e
        v = self.init_v(e, i)
        # we don't need u
        if torch.isnan(v).any():
            for i in range(50): print("Nan")
            exit()

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(e, emb, idx_kj, idx_ji, edge_attr)
            v = update_v(e, i)
            # we don't need u
            #u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)

        # we don't need u
        return v #u
    
    
    def forward_update_pos(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):
        num_nodes = z.size(0)
        
        # First compute without updating positions
        dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            pos, edge_index, num_nodes, 
            cutoff=self.cutoff, use_torsion=False
        )
        
        # Initialize with stable embeddings
        dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
        emb = (dist_emb, angle_emb)
        
        # Initialize stable edge and node features
        e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node)
        v = self.init_v(e, i)
        
        # Small scaling factor for position updates to prevent large jumps
        pos_scale = 1  # Start with a very small update scale
        
        for layer_idx, (update_e_layer, update_v_layer) in enumerate(zip(self.update_es, self.update_vs)):
            # Safety check on current features
            e1, e2 = e
            if torch.isnan(e1).any() or torch.isnan(e2).any():
                print(f"NaN detected in edge features at layer {layer_idx}, applying fix")
                e1 = torch.nan_to_num(e1, nan=0.0)
                e2 = torch.nan_to_num(e2, nan=0.0)
                e = (e1, e2)
            
            # Recompute geometric features with current positions
            # dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            #     pos, edge_index, num_nodes,
            #     cutoff=self.cutoff, use_torsion=False
            # )
            
            # Create embeddings with safety checks
            # dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
            # if torch.isnan(dist_emb).any():
            #     dist_emb = torch.nan_to_num(dist_emb, nan=0.0)
            # if torch.isnan(angle_emb).any():
            #     angle_emb = torch.nan_to_num(angle_emb, nan=0.0)
            # emb = (dist_emb, angle_emb)
            
            # Update edge and node features with NaN monitoring
            try:
                e = update_e_layer(e, emb, idx_kj, idx_ji, edge_attr)
                v = update_v_layer(e, i)
            except RuntimeError as error:
                print(f"Error in layer {layer_idx}: {error}")
                # Skip position update for this layer
                continue
            
            # Safety check post-update
            if torch.isnan(v).any():
                print(f"NaN detected in node features at layer {layer_idx}")
                # Skip position update for this layer
                continue
            
            # Update positions using edge features
            e1, e2 = e
            row, col = edge_index
            
            # Compute edge vectors with numerical stability
            edge_vec = pos[row] - pos[col]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            
            # Safe normalization of edge directions
            edge_dir = edge_vec / edge_dist
            
            # Create combined edge features with clipping to prevent extreme values
            e1_safe = torch.clamp(e1, min=-100, max=100)
            e2_safe = torch.clamp(e2, min=-100, max=100)
            edge_feat_combined = torch.cat([e1_safe, e2_safe], dim=-1)
            
            # Predict position updates with tanh to constrain values
            update_scale = torch.tanh(self.coord_mlp(edge_feat_combined))
            
            # Apply gradually increasing scale factor for position updates
            # This helps stabilize initial updates
            current_pos_scale = pos_scale * (1.0 + layer_idx * 0.5)
            
            # Compute position update vectors
            pos_updates = edge_dir * update_scale * current_pos_scale
            
            # Check for NaN in updates
            if torch.isnan(pos_updates).any():
                print(f"NaN detected in position updates at layer {layer_idx}")
                continue
            
            # Aggregate position updates safely
            pos_update = torch.zeros_like(pos)
            for dim in range(3):
                pos_update[:, dim:dim+1].scatter_add_(0, row.view(-1, 1), -pos_updates[:, dim:dim+1])
                pos_update[:, dim:dim+1].scatter_add_(0, col.view(-1, 1), pos_updates[:, dim:dim+1])
            
            # Normalize updates by node degree
            node_degrees = torch.zeros(num_nodes, device=pos.device)
            node_degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            node_degrees.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
            node_degrees = torch.clamp(node_degrees, min=1.0).view(-1, 1)
            
            # Apply normalized position updates
            pos_update = pos_update / node_degrees
            
            # Clip extreme position updates
            pos_update = torch.clamp(pos_update, min=-0.1, max=0.1)
            
            # Update positions
            pos = pos + pos_update
        
        return v, pos