import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN
from model_fastgtn import FastGTNs
import pickle
import argparse
from torch_geometric.utils import f1_score, add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy

if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GTN',
                        help='Model')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=1,
                        help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=1,
                        help='number of FastGTN layers')

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    with open('../data/%s/node_features.pkl' % args.dataset,'rb') as f:
        node_features = pickle.load(f)
    with open('../data/%s/edges.pkl' % args.dataset,'rb') as f:
        edges = pickle.load(f)
    with open('../data/%s/labels.pkl' % args.dataset,'rb') as f:
        labels = pickle.load(f)
    if args.dataset == 'PPI':
        with open('../data/%s/ppi_tvt_nids.pkl' % args.dataset, 'rb') as fp:
            nids = pickle.load(fp)

    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes
    # build adjacency matrices for each edge type
    A = []
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        # normalize each adjacency matrix
        if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))
    
    
    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    if args.dataset == 'PPI':
        train_node = torch.from_numpy(nids[0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(labels[nids[0]]).type(torch.cuda.FloatTensor)
        valid_node = torch.from_numpy(nids[1]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(labels[nids[1]]).type(torch.cuda.FloatTensor)
        test_node = torch.from_numpy(nids[2]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(labels[nids[2]]).type(torch.cuda.FloatTensor)
        num_classes = 121
        is_ppi = True
    else:
        train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)
        num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()])+1
        is_ppi = False
    final_f1, final_micro_f1 = [], []
    tmp = None
    runs = args.runs
    if args.pre_train:
        runs += 1
        pre_trained_fastGTNs = None
    for l in range(runs):
        # initialize a model
        if args.model == 'GTN':
            model = GTN(num_edge=len(A),
                                num_channels=num_channels,
                                w_in = node_features.shape[1],
                                w_out = node_dim,
                                num_class=num_classes,
                                num_layers=num_layers,
                                num_nodes=num_nodes,
                                args=args)        
        elif args.model == 'FastGTN':
            if args.pre_train and l == 1:
                pre_trained_fastGTNs = []
                for layer in range(args.num_FastGTN_layers):
                    pre_trained_fastGTNs.append(copy.deepcopy(model.fastGTNs[layer].layers))
            while len(A) > num_edge_type:
                del A[-1]
            model = FastGTNs(num_edge_type=len(A),
                            w_in = node_features.shape[1],
                            num_class=num_classes,
                            num_nodes = node_features.shape[0],
                            args = args)
            if args.pre_train and l > 0:
                for layer in range(args.num_FastGTN_layers):
                    model.fastGTNs[layer].layers = pre_trained_fastGTNs[layer]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.cuda()
        if args.dataset == 'PPI':
            loss = nn.BCELoss()
        else:
            loss = nn.CrossEntropyLoss()
        Ws = []
        
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        
        for i in range(epochs):
            # print('Epoch ',i)
            model.zero_grad()
            model.train()
            if args.model == 'FastGTN':
                loss,y_train,W = model(A, node_features, train_node, train_target, epoch=i)
            else:
                loss,y_train,W = model(A, node_features, train_node, train_target)
            if args.dataset == 'PPI':
                y_train = (y_train > 0).detach().float().cpu()
                train_f1 = 0.0
                sk_train_f1 = sk_f1_score(train_target.detach().cpu().numpy(), y_train.numpy(), average='micro')
            else:
                train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
                sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1), average='micro')
            # print(W)
            # print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1, sk_train_f1))
            
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                if args.model == 'FastGTN':
                    val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target, epoch=i)
                else:
                    val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                if args.dataset == 'PPI':
                    val_f1 = 0.0
                    y_valid = (y_valid > 0).detach().float().cpu()
                    sk_val_f1 = sk_f1_score(valid_target.detach().cpu().numpy(), y_valid.numpy(), average='micro')
                else:
                    val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                    sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1), average='micro')
                # print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1, sk_val_f1))

                if args.model == 'FastGTN':
                    test_loss, y_test,W = model.forward(A, node_features, test_node, test_target, epoch=i)
                else:
                    test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                if args.dataset == 'PPI':
                    test_f1 = 0.0
                    y_test = (y_test > 0).detach().float().cpu()
                    sk_test_f1 = sk_f1_score(test_target.detach().cpu().numpy(), y_test.numpy(), average='micro')
                else:
                    test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                    sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1), average='micro')
                # print('Test - Loss: {}, Macro_F1: {}, Micro_F1:{} \n'.format(test_loss.detach().cpu().numpy(), test_f1, sk_test_f1))
            if sk_val_f1 > best_micro_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1
        if l == 0 and args.pre_train:
            continue
        print('Run {}'.format(l))
        print('--------------------Best Result-------------------------')
        print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_train_f1, best_micro_train_f1))
        print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_val_loss, best_val_f1, best_micro_val_f1))
        print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_test_f1, best_micro_test_f1))
        final_f1.append(best_test_f1)
        final_micro_f1.append(best_micro_test_f1)

    print('--------------------Final Result-------------------------')
    print('Test - Macro_F1: {:.4f}+{:.4f}, Micro_F1:{:.4f}+{:.4f}'.format(np.mean(final_f1), np.std(final_f1), np.mean(final_micro_f1), np.std(final_micro_f1)))
