import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_add_pool
from FFiNet.models.model_utils import init_weight, MLP, PositionEncoder

from typing import List, Tuple

import numpy as np


def seed_all():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

class FFiNetTrModel(nn.Module):
    def __init__(self, feature_per_layer: List, num_heads: int,
                 pred_hidden_dim: int, pred_dropout: float, pred_layers:int,
                 activation=nn.ELU(), residual: bool = True, num_tasks: int = 1,
                 bias: bool = True, dropout: float = 0.1):
        super(FFiNetTrModel, self).__init__()

        # update phase
        self.feature_per_layer = feature_per_layer
        layers = []
        for i in range(len(feature_per_layer) - 1):
            layer = FFiTrLayer(
                num_node_features=feature_per_layer[i] * (1 if i == 0 else num_heads),
                output_dim=feature_per_layer[i + 1],
                num_heads=num_heads,
                concat=True if i < len(feature_per_layer) - 2 else False,
                activation=activation,
                residual=residual,
                bias=bias,
                dropout=dropout
            )
            layers.append(layer)
        self.ffi_model = nn.Sequential(*layers)

        # readout phase
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(([feature_per_layer[-1] * 2] + [pred_hidden_dim] * pred_layers + [num_tasks]), dropout=pred_dropout)
        self.predict.apply(init_weight)
        self.weighted_output = None

    def forward(self, data: Data):
        output, _, _, _, _, _, _, _, _, _, _ = self.ffi_model((data.x, data.edge_index, data.triple_index, data.quadra_index, data.pos, data.edge_attr,
                                                        data.distance_matrix,
                                                        data.distance_matrix_angle, data.angle_matrix,
                                                        data.distance_matrix_dihedral, data.dihedral_matrix))

        weighted = self.atom_weighting(output)
        self.weighted_output = weighted
        output1 = global_max_pool(output, data.batch)
        output2 = global_add_pool(weighted * output, data.batch)
        output = torch.cat([output1, output2], dim=1)
        return self.predict(output)
        


class FFiTrLayer(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, num_heads: int,
                 activation=nn.PReLU(), concat: bool = True, residual: bool = True,
                 bias: bool = True, dropout: float = 0.1, share_weights: bool = False,
                 transformer_on : bool = False):
        super(FFiTrLayer, self).__init__()

        seed_all()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights
        self.transformer_on = transformer_on

        # Embedding by linear projection
        self.linear_src = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
            self.linear_mid1 = self.linear_src
            self.linear_mid2 = self.linear_src
        else:
            self.linear_dst = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
            self.linear_mid1 = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
            self.linear_mid2 = nn.Linear(num_node_features, output_dim * num_heads, bias=False)

        # distance embedding for bonded and unbonded atom-pair distance
        self.linear_pos_bonded = nn.Linear(2, output_dim * num_heads)
        self.linear_pos_unbonded1 = nn.Linear(3, output_dim * num_heads)
        self.linear_pos_unbonded2 = nn.Linear(3, output_dim * num_heads)
        self.linear_pos_unbonded = nn.Linear(3, output_dim * num_heads)
        
        # angle of three atoms
        self.linear_angle = MLP([2, output_dim * num_heads])
        self.linear_dihedral = MLP([6, output_dim * num_heads])

        self.linear_one_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)
        self.linear_two_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)
        self.linear_three_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)
        self.linear_allout = nn.Linear(output_dim * num_heads, output_dim * num_heads)

        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))
        self.triple_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))
        self.quadra_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        self.layer_norm = nn.LayerNorm(output_dim * num_heads) if concat else nn.LayerNorm(output_dim)

        if self.transformer_on:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model = num_heads * output_dim, 
                                                            nhead = 4, dim_feedforward = 256, 
                                                            dropout = 0.1, batch_first = True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = 1)
            self.output_token = nn.Parameter(torch.FloatTensor(1, num_heads * output_dim))
        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, num_heads * output_dim, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)
        nn.init.xavier_uniform_(self.linear_mid1.weight)
        nn.init.xavier_uniform_(self.linear_mid2.weight)

        self.linear_one_hop.apply(init_weight)
        self.linear_two_hop.apply(init_weight)
        self.linear_three_hop.apply(init_weight)
        if self.transformer_on:
            self.linear_allout.apply(init_weight)
            self.transformer_encoder.apply(init_weight)
            nn.init.xavier_uniform_(self.output_token)


        self.linear_pos_bonded.apply(init_weight)
        self.linear_pos_unbonded.apply(init_weight)
        self.linear_pos_unbonded1.apply(init_weight)
        self.linear_pos_unbonded2.apply(init_weight)
        self.linear_angle.apply(init_weight)
        self.linear_dihedral.apply(init_weight)
    
        nn.init.xavier_uniform_(self.double_attn)
        nn.init.xavier_uniform_(self.triple_attn)
        nn.init.xavier_uniform_(self.quadra_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0) 


    def forward(self, data):

        # Input preprocessing
        x, edge_index, triple_index, quadra_index, pos, edge_attr, distance_matrix1, distance_matrix_angle1, angle_matrix1, distance_matrix_dihedral1, dihedral_matrix1 = data
        dihedral_src_index, dihedral_mid2_index, dihedral_mid1_index, dihedral_dst_index = quadra_index
        angle_src_index, angle_mid_index, angle_dst_index = triple_index
        edge_src_index, edge_dst_index = edge_index
    
        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x))
        dst_projected = self.linear_dst(self.dropout(x))
        mid1_projected = self.linear_mid1(self.dropout(x))
        mid2_projected = self.linear_mid2(self.dropout(x))

        # Position encoder(See in Transformer)
        self.position_encoder = PositionEncoder(d_model=self.num_heads*self.output_dim, device=x.device)
        self.position_encoder2 = PositionEncoder(d_model=self.num_heads*self.output_dim, device=x.device)

        x_squence = torch.stack([src_projected, mid2_projected, mid1_projected, dst_projected], dim=1)
        x_squence = self.position_encoder(x_squence)

        src_projected = x_squence[:, 0, :].view(-1, self.num_heads, self.output_dim)
        mid2_projected = x_squence[:, 1, :].view(-1, self.num_heads, self.output_dim)
        mid1_projected = x_squence[:, 2, :].view(-1, self.num_heads, self.output_dim)
        dst_projected = x_squence[:, 3, :].view(-1, self.num_heads, self.output_dim)

        #######################################
        ############## Edge Attn ##############
        #######################################
        distance_matrix = self.linear_pos_bonded(distance_matrix1).view(-1, self.num_heads, self.output_dim)

        # Edge attention coefficients
        edge_attn = self.leakyReLU((mid1_projected.index_select(0, edge_src_index)
                                    + dst_projected.index_select(0, edge_dst_index)) * distance_matrix)
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = mid1_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        ########################################
        ############## Angle Attn ##############
        ########################################

        # distance of the src atom and dst atom in each triedge
        distance_matrix_angle = self.linear_pos_unbonded1(distance_matrix_angle1)
        distance_matrix_angle = distance_matrix_angle.view(-1, self.num_heads, self.output_dim)        

        angle_matrix = self.linear_angle(angle_matrix1).view(-1, self.num_heads, self.output_dim)

        # Angle attention coefficients
        angle_attn = self.leakyReLU((mid2_projected.index_select(0, angle_src_index)
                                     + dst_projected.index_select(0, angle_dst_index)
                                     + mid1_projected.index_select(0, angle_mid_index))
                                    * (angle_matrix + distance_matrix_angle))
        angle_attn = ((self.triple_attn * angle_attn).sum(-1))
        exp_angle_attn = (angle_attn - angle_attn.max()).exp()

        # sum the angle scores to destination node
        angle_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                           dtype=exp_angle_attn.dtype,
                                           device=exp_angle_attn.device)
        angle_dst_index_broadcast = angle_dst_index.unsqueeze(-1).expand_as(exp_angle_attn)
        angle_node_score_sum.scatter_add_(0, angle_dst_index_broadcast, exp_angle_attn)

        exp_angle_attn = exp_angle_attn / (angle_node_score_sum.index_select(0, angle_dst_index) + 1e-16)
        exp_angle_attn = self.dropout(exp_angle_attn).unsqueeze(-1)


        angle_x_projected = mid2_projected.index_select(0, angle_src_index) * exp_angle_attn
        angle_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                   dtype=exp_angle_attn.dtype,
                                   device=exp_angle_attn.device)
        angle_dst_index_broadcast = (angle_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(angle_x_projected)
        angle_output.scatter_add_(0, angle_dst_index_broadcast, angle_x_projected)

        ###########################################
        ############## Dihedral Attn ##############
        ###########################################

        # distance of src atom and dst atom in each quaedge
        distance_matrix_dihedral = self.linear_pos_unbonded2(distance_matrix_dihedral1)
        distance_matrix_dihedral = distance_matrix_dihedral.view(-1, self.num_heads, self.output_dim)

        dihedral_matrix = self.linear_dihedral(dihedral_matrix1).view(-1, self.num_heads, self.output_dim)

        # dihedral attention coefficients
        dihedral_attn = self.leakyReLU((src_projected.index_select(0, dihedral_src_index)
                                        + dst_projected.index_select(0, dihedral_dst_index)
                                        + mid1_projected.index_select(0, dihedral_mid1_index)
                                        + mid2_projected.index_select(0, dihedral_mid2_index))
                                       * (dihedral_matrix + distance_matrix_dihedral))
        dihedral_attn = ((self.quadra_attn * dihedral_attn).sum(-1))
        exp_dihedral_attn = (dihedral_attn - dihedral_attn.max()).exp()

        # sum the dihedral scores to destination node
        dihedral_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                              dtype=exp_dihedral_attn.dtype,
                                              device=exp_dihedral_attn.device)
        dihedral_dst_index_broadcast = dihedral_dst_index.unsqueeze(-1).expand_as(exp_dihedral_attn)
        dihedral_node_score_sum.scatter_add_(0, dihedral_dst_index_broadcast, exp_dihedral_attn)

        exp_dihedral_attn = exp_dihedral_attn / (dihedral_node_score_sum.index_select(0, dihedral_dst_index) + 1e-16)
        exp_dihedral_attn = self.dropout(exp_dihedral_attn).unsqueeze(-1)

        # aggregation
        dihedral_x_projected = src_projected.index_select(0, dihedral_src_index) * exp_dihedral_attn
        dihedral_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                      dtype=exp_dihedral_attn.dtype,
                                      device=exp_dihedral_attn.device)
        dihedral_dst_index_broadcast = (dihedral_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(dihedral_x_projected)
        dihedral_output.scatter_add_(0, dihedral_dst_index_broadcast, dihedral_x_projected)
        

        ######################################
        ############## Attn Sum ##############
        ######################################
        if self.transformer_on:
            output_stack = torch.stack((edge_output.view(-1, self.num_heads * self.output_dim), 
                                        angle_output.view(-1, self.num_heads * self.output_dim),
                                        dihedral_output.view(-1, self.num_heads * self.output_dim),
                                        self.output_token.expand(edge_output.shape[0], -1)),dim=1)

            # output_stack = self.position_encoder2(output_stack)
            output = self.transformer_encoder(output_stack)[:, -1, :].view(-1, self.num_heads, self.output_dim)

        else:
            one_hop = self.linear_one_hop(edge_output.view(-1, self.num_heads * self.output_dim)).view(-1, self.num_heads, self.output_dim)
            two_hop = self.linear_two_hop(angle_output.view(-1, self.num_heads * self.output_dim)).view(-1, self.num_heads, self.output_dim)
            three_hop = self.linear_three_hop(dihedral_output.view(-1, self.num_heads * self.output_dim)).view(-1, self.num_heads, self.output_dim)

            zero_hop = dst_projected
            
            one_hop_attn = torch.diagonal(torch.matmul(zero_hop, one_hop.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.output_dim)
            two_hop_attn = torch.diagonal(torch.matmul(zero_hop, two_hop.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.output_dim)
            three_hop_attn = torch.diagonal(torch.matmul(zero_hop, three_hop.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(self.output_dim)

            squence_attn = torch.stack([one_hop_attn, two_hop_attn, three_hop_attn], dim=0)
            squence_attn = self.dropout(torch.softmax(squence_attn, dim=0).unsqueeze(-1))

            output = squence_attn[0, :, :] * edge_output + squence_attn[1, :, :] * angle_output + squence_attn[2, :, :] * dihedral_output
            
        # for plotting
        self.attention_group = {'edge attention': exp_edge_attn.detach().to('cpu'), 
                          'angle attention': exp_angle_attn.detach().to('cpu'),
                          'dihedral attention': exp_dihedral_attn.detach().to('cpu')}
        
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(-1, self.num_heads, self.output_dim)
        
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        output = self.layer_norm(output)
        if self.activation is not None:
            output = self.activation(output)

        self.node_output = output.detach().to('cpu')
        return output, edge_index, triple_index, quadra_index, pos, edge_attr, distance_matrix1, distance_matrix_angle1, angle_matrix1, distance_matrix_dihedral1, dihedral_matrix1





