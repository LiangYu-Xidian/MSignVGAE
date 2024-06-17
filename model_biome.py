import torch
import torch.nn as nn
import torch.nn.functional as F

import args


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


class VGAE3(nn.Module):
    def __init__(self):
        super(VGAE3, self).__init__()
        self.conv1 = GraphConv(args.input_dim1_bio, args.h1_bio, activation=F.relu)
        self.conv2 = GraphConv(args.h1_bio, args.h2_bio, activation=F.relu)

        self.meanGCN1 = GraphConv(args.h2_bio, args.z1_bio, activation=lambda x: x)
        self.logstdGCN1 = GraphConv(args.h2_bio, args.z1_bio, activation=lambda x: x)

        self.meanGCN2 = GraphConv(args.h2_bio, args.z2_bio, activation=lambda x: x)
        self.logstdGCN2 = GraphConv(args.h2_bio, args.z2_bio, activation=lambda x: x)

        self.meanGCN3 = GraphConv(args.h2_bio, args.z3_bio, activation=lambda x: x)
        self.logstdGCN3 = GraphConv(args.h2_bio, args.z3_bio, activation=lambda x: x)

        self.transform_z = nn.Linear(args.z1_bio + args.z2_bio + args.z3_bio, args.z1_bio + args.z2_bio + args.z3_bio)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)

        self.aux1 = nn.Linear(args.z1_bio + args.z2_bio + args.z3_bio, args.h1_bio)
        self.dropout1 = nn.Dropout(0.5)
        # self.relu1 = nn.LeakyReLU(0.1)

        self.aux2 = nn.Linear(args.z1_bio + args.z2_bio + args.z3_bio, args.h2_bio)
        self.dropout2 = nn.Dropout(0.5)
        # self.relu2 = nn.LeakyReLU(0.1)

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
        pred = torch.sigmoid(torch.matmul(z, trans_z.t()))
        return pred

    def decoder_1(self, z):
        trans_z = self.aux1(z)
        trans_z = self.dropout1(trans_z)
        trans_z = torch.sigmoid(trans_z)
        return trans_z

    def decoder_2(self, z):
        trans_z = self.aux2(z)
        trans_z = self.dropout2(trans_z)
        trans_z = torch.sigmoid(trans_z)
        return trans_z

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
        aux1 = self.decoder_1(z)
        aux2 = self.decoder_2(z)
        return reconst, aux1, aux2, x1_1, x2_2, z


class VGAE4(nn.Module):
    def __init__(self):
        super(VGAE4, self).__init__()
        self.conv1 = GraphConv(args.input_dim2_bio, args.h1_bio2, activation=F.relu)
        self.conv2 = GraphConv(args.h1_bio2, args.h2_bio2, activation=F.relu)

        self.meanGCN1 = GraphConv(args.h2_bio2, args.z1_bio2, activation=lambda x: x)
        self.logstdGCN1 = GraphConv(args.h2_bio2, args.z1_bio2, activation=lambda x: x)

        self.meanGCN2 = GraphConv(args.h2_bio2, args.z2_bio2, activation=lambda x: x)
        self.logstdGCN2 = GraphConv(args.h2_bio2, args.z2_bio2, activation=lambda x: x)

        self.meanGCN3 = GraphConv(args.h2_bio2, args.z3_bio2, activation=lambda x: x)
        self.logstdGCN3 = GraphConv(args.h2_bio2, args.z3_bio2, activation=lambda x: x)

        self.transform_z = nn.Linear(args.z1_bio2 + args.z2_bio2 + args.z3_bio2, args.z1_bio2 + args.z2_bio2 + args.z3_bio2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)

        self.aux1 = nn.Linear(args.z1_bio2 + args.z2_bio2 + args.z3_bio2, args.h1_bio2)
        self.dropout1 = nn.Dropout(0.5)
        # self.relu1 = nn.LeakyReLU(0.1)

        self.aux2 = nn.Linear(args.z1_bio2 + args.z2_bio2 + args.z3_bio2, args.h2_bio2)
        self.dropout2 = nn.Dropout(0.5)
        # self.relu2 = nn.LeakyReLU(0.1)

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
        pred = torch.sigmoid(torch.matmul(z, trans_z.t()))
        return pred

    def decoder_1(self, z):
        trans_z = self.aux1(z)
        trans_z = self.dropout1(trans_z)
        trans_z = torch.sigmoid(trans_z)
        return trans_z

    def decoder_2(self, z):
        trans_z = self.aux2(z)
        trans_z = self.dropout2(trans_z)
        trans_z = torch.sigmoid(trans_z)
        return trans_z

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
        aux1 = self.decoder_1(z)
        aux2 = self.decoder_2(z)
        return reconst, aux1, aux2, x1_1, x2_2, z



class VGAE_dot_conv_leakyrelu(nn.Module):
    def __init__(self):
        super(VGAE_dot_conv_leakyrelu, self).__init__()
        self.conv1 = GraphConv(args.input_dim_bio_sign, args.h1_bio2, activation=F.leaky_relu)
        self.conv2 = GraphConv(args.h1_bio2, args.h2_bio2, activation=F.leaky_relu)

        self.meanGCN1 = GraphConv(args.h2_bio2, args.z1_bio2, activation=lambda x: x)
        self.logstdGCN1 = GraphConv(args.h2_bio2, args.z1_bio2, activation=lambda x: x)

        self.meanGCN2 = GraphConv(args.h2_bio2, args.z2_bio2, activation=lambda x: x)
        self.logstdGCN2 = GraphConv(args.h2_bio2, args.z2_bio2, activation=lambda x: x)

        self.meanGCN3 = GraphConv(args.h2_bio2, args.z3_bio2, activation=lambda x: x)
        self.logstdGCN3 = GraphConv(args.h2_bio2, args.z3_bio2, activation=lambda x: x)

        self.transform_z = nn.Linear(args.z1_bio2 + args.z2_bio2 + args.z3_bio2, 
                                     args.z1_bio2 + args.z2_bio2 + args.z3_bio2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.1)

        self.aux1 = nn.Linear(args.z1_bio2 + args.z2_bio2+ args.z3_bio2, args.h1_bio2)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.LeakyReLU(0.1)
        # self.relu1 = nn.ReLU()

        self.aux2 = nn.Linear(args.z1_bio2 + args.z2_bio2 + args.z3_bio2, args.h2_bio2)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.LeakyReLU(0.1)
        # self.relu2 = nn.ReLU()


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

    def decoder_1(self, z):
        trans_z = self.aux1(z)
        trans_z = self.dropout1(trans_z)
        trans_z = self.relu1(trans_z)
        return trans_z

    def decoder_2(self, z):
        trans_z = self.aux2(z)
        trans_z = self.dropout2(trans_z)
        trans_z = self.relu2(trans_z)
        return trans_z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, adj, x):
        z, x1, x2 = self.encoder(adj, x)
        x1_1 = self.relu1(x1)
        x2_2 = self.relu2(x2)
        reconst = self.decoder_main(z)
        aux1 = self.decoder_1(z)
        aux2 = self.decoder_2(z)
        return reconst, aux1, aux2, x1_1, x2_2, z




