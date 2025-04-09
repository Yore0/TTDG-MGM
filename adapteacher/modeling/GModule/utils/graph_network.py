import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.data import Data
from adapteacher.modeling.GModule.utils.attentions import MultiHeadAttention


class Topo_Graph(nn.Module):
    def __init__(self,input_ch):
        super(Topo_Graph, self).__init__()
        self.graph = GNNModel(input_ch)
        self.attention = MultiHeadAttention(input_ch, 1, dropout=0.1, version='v2')

    def forward(self, nodes, edges):
        edge_index = torch.combinations(torch.arange(0, nodes.size(0)), r=2).t().to(nodes.device)
        edge_attr = edges[edge_index[0], edge_index[1]]
        data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        x = self.graph(data)
        out, _ = self.attention([x, x, x])
        return out

class GNNModel(nn.Module):
    def __init__(self, num_features, dim=256):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        # self.conv2 = GCNConv(dim, dim)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))

        x = x + x1
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Feat2Graph(nn.Module):
    def __init__(self, num_feats):
        super(Feat2Graph, self).__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)

        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)
        return x, adj

class MAGNN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=False, init="xavier"):
        super(MAGNN, self).__init__()
        self.graph = Feat2Graph(nfeat)

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.gc3 = GraphConvolution(nhid, nfeat, init=init)
        self.dropout = dropout


    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x):
        x_in = x

        x, adj = self.graph(x)
        x = self.bottleneck(self.gc1, self.gc2, self.gc3, adj, x)
        # x = F.relu(self.gc1(x, adj))
        # x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, adj))

        x = x + x_in
        return x