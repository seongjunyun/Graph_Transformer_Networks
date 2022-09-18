import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
from torch_scatter import scatter_add
import torch_sparse

class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers, args=None):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.args = args
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))
        self.layers = nn.ModuleList(layers)
        if args.dataset in ["PPI", "BOOK", "MUSIC"]:
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out, args=args)
        self.linear = nn.Linear(self.w_out*self.num_channels, self.num_class)

    def normalization(self, H, num_nodes):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            deg_row, deg_col = self.norm(edge.detach(), num_nodes, value)
            value = (deg_row) * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, target_x, target, num_nodes=None, eval=False, node_labels=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A, num_nodes, eval=eval)
            else:                
                H, W = self.layers[i](A, num_nodes, H, eval=eval)
            H = self.normalization(H, num_nodes)
            Ws.append(W)
        for i in range(self.num_channels):
            edge_index, edge_weight = H[i][0], H[i][1]
            if i==0:                
                X_ = self.gcn(X,edge_index=edge_index.detach(), edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                X_tmp = F.relu(self.gcn(X,edge_index=edge_index.detach(), edge_weight=edge_weight))
                X_ = torch.cat((X_,X_tmp), dim=1)

        y = self.linear(X_[target_x])
        if eval:
            return y
        else:
            if self.args.dataset == 'PPI':
                loss = self.loss(self.m(y), target)
            else:
                loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, num_nodes, H_=None, eval=False):
        if self.first == True:
            result_A = self.conv1(A, num_nodes, eval=eval)
            result_B = self.conv2(A, num_nodes, eval=eval)                
            W = [(F.softmax(self.conv1.weight, dim=1)),(F.softmax(self.conv2.weight, dim=1))]
        else:
            result_A = H_
            result_B = self.conv1(A, num_nodes, eval=eval)
            W = [(F.softmax(self.conv1.weight, dim=1))]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            mat_b = torch.sparse_coo_tensor(b_edge, b_value, (num_nodes, num_nodes)).to(a_edge.device)
            mat = torch.sparse.mm(mat_a, mat_b).coalesce()
            edges, values = mat.indices(), mat.values()
            # edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, num_nodes, num_nodes, num_nodes)
            H.append((edges, values))
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A, num_nodes, eval=eval):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))
        return results
