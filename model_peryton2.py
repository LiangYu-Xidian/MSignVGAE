import torch
import torch.nn as nn
import torch.nn.functional as F

import args

##############################################################
# for peryton_sign3.py


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0., bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        # 下面这两行的顺序 很重要   torch只支持 sparse与dense的矩阵乘法
        x = self.w(x)
        x = adj.mm(x)
        x = self.dropout(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


class VGAE3_2(nn.Module):
    def __init__(self):
        super(VGAE3_2, self).__init__()
        self.conv1 = GraphConv(args.input_dim1_pery_sign, args.h1_pery, activation=F.relu)
        self.conv2 = GraphConv(args.h1_pery, args.h2_pery, activation=F.relu)

        self.meanGCN1 = GraphConv(args.h2_pery, args.z1_pery, activation=lambda x: x)
        self.logstdGCN1 = GraphConv(args.h2_pery, args.z1_pery, activation=lambda x: x)

        self.meanGCN2 = GraphConv(args.h2_pery, args.z2_pery, activation=lambda x: x)
        self.logstdGCN2 = GraphConv(args.h2_pery, args.z2_pery, activation=lambda x: x)

        self.meanGCN3 = GraphConv(args.h2_pery, args.z3_pery, activation=lambda x: x)
        self.logstdGCN3 = GraphConv(args.h2_pery, args.z3_pery, activation=lambda x: x)

        self.transform_z = nn.Linear(args.z1_pery + args.z2_pery + args.z3_pery, args.z1_pery + args.z2_pery + args.z3_pery)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)

        self.aux1 = nn.Linear(args.z1_pery + args.z2_pery + args.z3_pery, args.h1_pery)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.LeakyReLU(0.1)

        self.aux2 = nn.Linear(args.z1_pery + args.z2_pery + args.z3_pery, args.h2_pery)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.LeakyReLU(0.1)

    def encoder(self, adj, x):
        x1 = self.conv1(adj, x)
        x2 = self.conv2(adj, x1)
        self.mean1 = h1 = self.meanGCN1(adj, x2)
        self.logstd1 = std1 = self.logstdGCN1(adj, x2)

        self.mean2 = h2 = self.meanGCN2(adj, x2)
        self.logstd2 = std2 = self.logstdGCN2(adj, x2)

        self.mean3 = h3 = self.meanGCN3(adj, x2)
        self.logstd3 = std3 = self.logstdGCN3(adj, x2)

        self.mean = torch.cat([self.mean1, self.mean2, self.mean3], 1)
        self.logstd = torch.cat([self.logstd1, self.logstd2, self.logstd3], 1)

        z1 = self.reparameterize(h1, std1)
        z2 = self.reparameterize(h2, std2)
        z3 = self.reparameterize(h3, std3)
        z = torch.cat([z1, z2, z3], 1)

        return z, x1, x2

    def decoder_main(self, z):
        trans_z = self.transform_z(z)
        trans_z = self.dropout(trans_z)
        trans_z = self.relu(trans_z)
        pred = torch.matmul(z, trans_z.t())
        return pred

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, adj, x):
        z, x1, x2 = self.encoder(adj, x)
        x1_1 = torch.sigmoid(x1)
        x2_2 = torch.sigmoid(x2)
        reconst = self.decoder_main(z)
        return reconst, x1_1, x2_2, z


class VGAE4_2(nn.Module):
    def __init__(self):
        super(VGAE4_2, self).__init__()
        self.conv1 = GraphConv(args.input_dim2_pery_sign, args.h1_pery2, activation=F.relu)
        self.conv2 = GraphConv(args.h1_pery2, args.h2_pery2, activation=F.relu)

        self.meanGCN1 = GraphConv(args.h2_pery2, args.z1_pery2, activation=lambda x: x)
        self.logstdGCN1 = GraphConv(args.h2_pery2, args.z1_pery2, activation=lambda x: x)

        self.meanGCN2 = GraphConv(args.h2_pery2, args.z2_pery2, activation=lambda x: x)
        self.logstdGCN2 = GraphConv(args.h2_pery2, args.z2_pery2, activation=lambda x: x)

        self.meanGCN3 = GraphConv(args.h2_pery2, args.z3_pery2, activation=lambda x: x)
        self.logstdGCN3 = GraphConv(args.h2_pery2, args.z3_pery2, activation=lambda x: x)

        self.transform_z = nn.Linear(args.z1_pery2 + args.z2_pery2 + args.z3_pery2, args.z1_pery2 + args.z2_pery2 + args.z3_pery2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)

        self.aux1 = nn.Linear(args.z1_pery2 + args.z2_pery2 + args.z3_pery2, args.h1_pery2)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.LeakyReLU(0.1)

        self.aux2 = nn.Linear(args.z1_pery2 + args.z2_pery2 + args.z3_pery2, args.h2_pery2)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.LeakyReLU(0.1)

    def encoder(self, adj, x):
        x1 = self.conv1(adj, x)
        x2 = self.conv2(adj, x1)
        self.mean1 = h1 = self.meanGCN1(adj, x2)
        self.logstd1 = std1 = self.logstdGCN1(adj, x2)

        self.mean2 = h2 = self.meanGCN2(adj, x2)
        self.logstd2 = std2 = self.logstdGCN2(adj, x2)

        self.mean3 = h3 = self.meanGCN3(adj, x2)
        self.logstd3 = std3 = self.logstdGCN3(adj, x2)

        self.mean = torch.cat([self.mean1, self.mean2, self.mean3], 1)
        self.logstd = torch.cat([self.logstd1, self.logstd2, self.logstd3], 1)

        z1 = self.reparameterize(h1, std1)
        z2 = self.reparameterize(h2, std2)
        z3 = self.reparameterize(h3, std3)
        z = torch.cat([z1, z2, z3], 1)

        return z, x1, x2

    def decoder_main(self, z):
        trans_z = self.transform_z(z)
        trans_z = self.dropout(trans_z)
        trans_z = self.relu(trans_z)
        pred = torch.matmul(z, trans_z.t())
        return pred

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, adj, x):
        z, x1, x2 = self.encoder(adj, x)
        x1_1 = torch.sigmoid(x1)
        x2_2 = torch.sigmoid(x2)
        reconst = self.decoder_main(z)
        return reconst, x1_1, x2_2, z

