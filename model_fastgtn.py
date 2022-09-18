import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
import torch_sparse
from torch_geometric.utils import softmax
from utils import _norm, generate_non_local_graph


device = f'cuda' if torch.cuda.is_available() else 'cpu'

class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None):
        super(FastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        fastGTNs = []
        for i in range(args.num_FastGTN_layers):
            if i == 0:
                fastGTNs.append(FastGTN(num_edge_type, w_in, num_class, num_nodes, args))
            else:
                fastGTNs.append(FastGTN(num_edge_type, args.node_dim, num_class, num_nodes, args))
        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.linear = nn.Linear(args.node_dim, num_class)
        self.loss = nn.CrossEntropyLoss()
        if args.dataset == "PPI":
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:    
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, A, X, target_x, target, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None, epoch=None):
        if num_nodes == None:
            num_nodes = self.num_nodes
        H_, Ws = self.fastGTNs[0](A, X, num_nodes=num_nodes, epoch=epoch)
        for i in range(1, self.num_FastGTN_layers):
            H_, Ws = self.fastGTNs[i](A, H_, num_nodes=num_nodes)
        y = self.linear(H_[target_x])
        if eval:
            return y
        else:
            if self.args.dataset == 'PPI':
                loss = self.loss(self.m(y), target)
            else:
                loss = self.loss(y, target.squeeze())
        return loss, y, Ws

class FastGTN(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None):
        super(FastGTN, self).__init__()
        if args.non_local:
            num_edge_type += 1
        self.num_edge_type = num_edge_type
        self.num_channels = args.num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        args.w_in = w_in
        self.w_out = args.node_dim
        self.num_class = num_class
        self.num_layers = args.num_layers
        
        if pre_trained is None:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args))
                else:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args))
            self.layers = nn.ModuleList(layers)
        else:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args, pre_trained=pre_trained[i]))
                else:
                    layers.append(FastGTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args, pre_trained=pre_trained[i]))
            self.layers = nn.ModuleList(layers)
        
        self.Ws = []
        for i in range(self.num_channels):
            self.Ws.append(GCNConv(in_channels=self.w_in, out_channels=self.w_out).weight)
        self.Ws = nn.ParameterList(self.Ws)

        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)

        feat_trans_layers = []
        for i in range(self.num_layers+1):
            feat_trans_layers.append(nn.Sequential(nn.Linear(self.w_out, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64)))
        self.feat_trans_layers = nn.ModuleList(feat_trans_layers)

        self.args = args

        self.out_norm = nn.LayerNorm(self.w_out)
        self.relu = torch.nn.ReLU()

    def forward(self, A, X, num_nodes, eval=False, node_labels=None, epoch=None):        
        Ws = []
        X_ = [X@W for W in self.Ws]
        H = [X@W for W in self.Ws]
        
        for i in range(self.num_layers):
            if self.args.non_local:
                g = generate_non_local_graph(self.args, self.feat_trans_layers[i], torch.stack(H).mean(dim=0), A, self.num_edge_type, num_nodes)
                deg_inv_sqrt, deg_row, deg_col = _norm(g[0].detach(), num_nodes, g[1])
                g[1] = softmax(g[1],deg_row)
                if len(A) < self.num_edge_type:
                    A.append(g)
                else:
                    A[-1] = g
            
            H, W = self.layers[i](H, A, num_nodes, epoch=epoch, layer=i+1)
            Ws.append(W)
        
        for i in range(self.num_channels):
            if i==0:
                H_ = F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
            else:
                if self.args.channel_agg == 'concat':
                    H_ = torch.cat((H_,F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])), dim=1)
                elif self.args.channel_agg == 'mean':
                    H_ = H_ + F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
        if self.args.channel_agg == 'concat':
            H_ = F.relu(self.linear1(H_))
        elif self.args.channel_agg == 'mean':
            H_ = H_ /self.args.num_channels
        
        
        
        return H_, Ws

class FastGTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True, args=None, pre_trained=None):
        super(FastGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if pre_trained is not None:
            self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args, pre_trained=pre_trained.conv1)
        else:
            self.conv1 = FastGTConv(in_channels, out_channels, num_nodes, args=args)
        self.args = args
        self.feat_transfrom = nn.Sequential(nn.Linear(args.w_in, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64))
    def forward(self, H_, A, num_nodes, epoch=None, layer=None):
        result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
        W = [W1]
        Hs = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            H = torch.sparse.mm(mat_a, H_[i])
            Hs.append(H)
        return Hs, W

class FastGTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, args=None, pre_trained=None):
        super(FastGTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes

        self.reset_parameters()

        if pre_trained is not None:
            with torch.no_grad():
                self.weight.data = pre_trained.weight.data
        
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.1)
        if self.args.non_local and self.args.non_local_weight != 0:
            with torch.no_grad():
                self.weight[:,-1] = self.args.non_local_weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)               

    def forward(self, A, num_nodes, epoch=None, layer=None):
        
        weight = self.weight
        filter = F.softmax(weight, dim=1)   
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
        
        return results, filter



