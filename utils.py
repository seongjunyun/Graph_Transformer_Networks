from __future__ import division

import torch
import numpy as np
import random
import subprocess
from torch_scatter import scatter_add
import pdb
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import time


def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                dtype=dtype,
                                device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)                                                          
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
    return deg_inv_sqrt, row, col


# def sample_adj(edge_index, edge_weight, thr=0.5, sampling_type='random', binary=False):
#         # tmp = (edge_weight - torch.mean(edge_weight)) / torch.std(edge_weight)
#         if sampling_type == 'gumbel':
#             sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1, 
#                                                                     probs=edge_weight).rsample(thr=thr)
#         elif sampling_type == 'random':
#             sampled = pyro.distributions.Bernoulli(1-thr).sample(edge_weight.shape).cuda()
#         elif sampling_type == 'topk':
#             indices = torch.topk(edge_weight, k=int(edge_weight.shape[0]*0.8))[1]
#             sampled = torch.zeros_like(edge_weight)
#             sampled[indices] = 1
#         # print(sampled.sum()/edge_weight.shape[0])
#         edge_index = edge_index[:,sampled==1]
#         edge_weight = edge_weight*sampled
#         edge_weight = edge_weight[edge_weight!=0]
#         if binary:
#             return edge_index, sampled[sampled!=0]
#         else:
#             return edge_index, edge_weight


def to_heterogeneous(edge_index, num_nodes, n_id, edge_type, num_edge, device='cuda', args=None):
    # edge_index = adj[0]
    # num_nodes = adj[2][0]
    edge_type_indices = []
    # pdb.set_trace()
    for k in range(edge_index.shape[1]):
        edge_tmp = edge_index[:,k]
        e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
        edge_type_indices.append(e_type)
    edge_type_indices = np.array(edge_type_indices)
    A = []
    for e_type in range(num_edge):
        edge_tmp = edge_index[:,edge_type_indices==e_type]
        #################################### j -> i ########################################
        edge_tmp = torch.flip(edge_tmp, [0])
        #################################### j -> i ########################################
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        if args.model == 'FastGTN':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_weight=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp.to(device), value_tmp.to(device)))
    edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
    A.append([edge_tmp.to(device),value_tmp.to(device)])
    return A

# def to_heterogeneous(adj, n_id, edge_type, num_edge, device='cuda'):
#     edge_index = adj[0]
#     num_nodes = adj[2][0]
#     edge_type_indices = []
#     for k in range(edge_index.shape[1]):
#         edge_tmp = edge_index[:,k]
#         e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
#         edge_type_indices.append(e_type)
#     edge_type_indices = np.array(edge_type_indices)
#     A = []
#     for e_type in range(num_edge):
#         edge_tmp = edge_index[:,edge_type_indices==e_type]
#         value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
#         A.append((edge_tmp.to(device), value_tmp.to(device)))
#     edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
#     value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
#     A.append([edge_tmp.to(device),value_tmp.to(device)])

#     return A


def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:    
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x@x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:,:K]
    D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]

    # if len(A) < num_edge:

    #     deg_inv_sqrt, deg_row, deg_col = _norm(edge_index, num_nodes, edge_value)
    #     edge_value = deg_inv_sqrt[deg_col] * edge_value
    #     g = (edge_index, edge_value)
    #     A.append(g)
    # else:
    #     deg_inv_sqrt, deg_row, deg_col = _norm(edge_index, num_nodes, edge_value)
    #     edge_value = deg_inv_sqrt[deg_col] * edge_value
    #     g = (edge_index, edge_value)
    #     A[-1] = g