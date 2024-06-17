import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, precision_score, roc_auc_score
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import scipy.sparse as sp
from torch.optim import Adam

from layers import SinkhornDistance
from model_peryton import *
from utils import *
from get_sim import *
import args
import os
import time
import matplotlib.pyplot as plt
import random

torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# paprameters

random.seed(1)
k1 = 140  # mic
k2 = 5  # dis
A = pd.read_csv('./peryton/sign_final_ass.csv', index_col=0).to_numpy()

print("the number of microbes and diseases", A.shape)
print("the number of associations", sum(sum(A)))
x, y = A.shape
score_matrix = np.zeros([x, y])
samples = get_all_the_samples(A)

# sim_m, sim_d = get_sim_3ss_1ms_peryton(A, k1, k2)
sim_m, sim_d = np.load('./peryton/sim_m_sign.npy'), np.load('./peryton/sim_d_sign.npy')
sim_m_0 = set_digo_zero(sim_m, 0)
sim_d_0 = set_digo_zero(sim_d, 0)


# features_m, features_d = get_features_A(A)
# features_m, features_d = sparse_to_tuple(sp.coo_matrix(features_m)), sparse_to_tuple(sp.coo_matrix(features_d))

# features_d = sparse_to_tuple(sp.coo_matrix(sim_d))

sim_d_rand = np.random.randn(A.shape[1], A.shape[0])
sim_d = np.hstack((sim_d, sim_d_rand))
sim_m_rand = np.random.randn(A.shape[0], A.shape[1])
sim_m = np.hstack((sim_m, sim_m_rand))
features_m = np.vstack((sim_m, sim_d))
features_m = sparse_to_tuple(sp.coo_matrix(features_m))

# A_unsign = np.where(A==0,0,1)
row = np.hstack((np.eye(A.shape[0]), A))
row2 = np.hstack((A.T, np.eye(A.shape[1])))
matrix = np.vstack((row, row2))


w_aux1s = np.arange(0.05, 0.4, 0.05)
w_aux2s = np.arange(0.05, 0.4, 0.05)

for i in range(len(w_aux1s)):
    for j in range(len(w_aux2s)):
        w_w = 1 - w_aux1s[i] - w_aux2s[j]
        suffix = 'peryton_sign2_leakyReLU_200_sum_{:.2f}_{:.2f}_{:.2f}'.format(w_aux1s[i], w_aux2s[j], w_w)
        ##########################################################################################
        # 对微生物相似性
        adj_norm = preprocess_graph_signGCN(matrix)
        adj = matrix
        # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight = 25
        norm = 0.5
        sim_m_0 = sp.coo_matrix(matrix)
        sim_m_0.eliminate_zeros()
        adj_label = sim_m_0 + sp.eye(sim_m_0.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2])).to(device)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                        torch.FloatTensor(adj_label[1]),
                                        torch.Size(adj_label[2])).to_dense().to(device)
        features = torch.sparse.FloatTensor(torch.LongTensor(features_m[0].T),
                                        torch.FloatTensor(features_m[1]),
                                        torch.Size(features_m[2])).to(device)
        weight_mask = adj_label.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        # weight_tensor[weight_mask] = pos_weight
        weight_tensor[weight_mask] = 25
        # init model and optimizer
        model = VGAE_leakyReLU()
        model.to(device)
        print(model)
        # optimizer = Adam(model.parameters(), lr=args.learning_rate)
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        print("Optimizer", optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8, last_epoch=-1)

        sinkhorn = SinkhornDistance(eps=0.1, max_iter=200, reduction='mean', device=device)

        tra_auc, tra_ap, tra_l, v_auc, v_ap = [], [], [], [], []
        tra_acc, tra_baseloss, tra_aux1, tra_aux2 = [], [], [], []
        train_wloss, tra_kl = [], []
        tra_r2, tra_RMSE = [], []

        min_loss = 10.
        min_mse = 100.

        # suffix = 'peryton_sign2_2w'
        for epoch in range(5000):
            t = time.time()
            model.train()
            A_pred, x1_pred, x2_pred, x1, x2, z = model(adj_norm, features)

            optimizer.zero_grad()
            loss = base_loss = norm * F.mse_loss(A_pred.view(-1), adj_label.view(-1))
            # print('#######################')
            # print('base_loss:', loss.item())
            tra_baseloss.append(loss.item())

            kl_divergence = 0.5 / A_pred.size(0) * (
                    1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean() / 2
            tra_kl.append(-kl_divergence.item())
            # loss -= kl_divergence
            # loss -= kl_divergence * w_w

            loss_aux1 = norm * F.mse_loss(x1_pred.view(-1), x1.detach().view(-1))
            tra_aux1.append(loss_aux1.item())
            loss += loss_aux1 * w_aux1s[i]
            # loss += loss_aux1

            loss_aux2 = norm * F.mse_loss(x2_pred.view(-1), x2.detach().view(-1))
            tra_aux2.append(loss_aux2.item())
            loss += loss_aux2 * w_aux2s[i]
            # loss += loss_aux2

            wasser_loss = 0.5 / A_pred.size(0) * sinkhorn(z, torch.randn_like(z))[0]
            train_wloss.append(wasser_loss.item())
            loss += wasser_loss * w_w
            # loss += wasser_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            r2 = r2_score(adj_label.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1))
            RMSE = np.sqrt(mean_squared_error(adj_label.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1)))
            tra_r2.append(r2)
            tra_RMSE.append(RMSE)

            tra_l.append(loss.item())
            print('--------------------------------')
            print("Epoch:", '%04d' % (epoch + 1), "base_loss=", "{:.5f}".format(base_loss.item()),
                    "train_loss=", "{:.5f}".format(loss.item()),
                    "train_r2=", "{:.5f}".format(r2), "train_RMSE=", "{:.5f}".format(RMSE),
                    "time=", "{:.5f}".format(time.time() - t))
            if RMSE < min_mse:
                    min_mse = RMSE
                    state = {'model': model.state_dict(),
                            'epoch': epoch,
                            'min_mse': min_mse,
                            }
                    torch.save(state, './sign/opt/models/' + suffix + '.pth')

        model.eval()

        save_dic = {'train_loss': tra_l,
                'train_baseloss': tra_baseloss,
                'train_aux1': tra_aux1,
                'train_aux2': tra_aux2,
                'train_wloss': train_wloss,
                'train_kl': tra_kl,
                'train_r2': tra_r2,
                'train_RMSE': tra_RMSE,
                }
        np.save('./sign/opt/metrics/' + suffix + '.npy', save_dic)
        model_ = VGAE_full()
        model_.to(device)
        model_.load_state_dict(torch.load('./sign/opt/models/' + suffix + '.pth')['model'])
        model_.eval()
        out_m = model_(adj_norm, features)

        latent_m = out_m[-1].cpu().detach().numpy()
        np.save('./sign/opt/latent/latent_' + suffix + '.npy', latent_m)
