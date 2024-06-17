import numpy as np
import pandas as pd
import seaborn as sns
import torch
from xgboost import XGBClassifier
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import metrics

from sklearn.model_selection import KFold
from utils import *
from get_sim import *
import random

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#001e36']

random.seed(1)
# paprameters
k1 = 30
k2 = 5
D = 90  # MF dimension
A = np.load('./HMDAD/mic-dis Association.npy')
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))
x, y = A.shape
score_matrix = np.zeros([x, y])
samples = get_all_the_samples(A)

all_out = []

w_aux1s = np.arange(0.05, 0.55, 0.05)
w_aux2s = np.arange(0.05, 0.55, 0.05)

for l1 in range(len(w_aux1s)):
    for l2 in range(len(w_aux2s)):
        w_w = 1.0
        # w_w = 1 - w_aux1s[l1] - w_aux2s[l2]
        suffix = '3ss_GIP_blastn_convergeGIP_dis_drug_ms_ww1.0_{:.2f}_{:.2f}_{:.2f}'.format(w_aux1s[l1], w_aux2s[l2], w_w)
        # suffix = '3ss_1ms_all_1_w_weight_{:.2f}_{:.2f}_{:.2f}'.format(w_aux1s[l1], w_aux2s[l2], w_w)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(3, 2))
        # suffix = 'all_w_weight_0.05_0.10_0.85'
        latent_m = np.load('./opt/latent2/latent_m_mse_' + suffix + '.npy')
        latent_d = np.load('./opt/latent2/latent_d_loss_' + suffix + '.npy')
        # cross validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        iter_ = 0  # control each iterator
        sum_score = 0
        out = []
        out_results = []

        for j, (train_index, test_index) in enumerate(kf.split(samples)):
            out.append([train_index, test_index])
            iter_ = iter_ + 1

            train_samples = samples[train_index, :]
            test_samples = samples[test_index, :]

            vec_len1 = latent_m.shape[1]
            vec_len2 = latent_d.shape[1]
            train_n = train_samples.shape[0]
            train_feature = np.zeros([train_n, vec_len1 + vec_len2])
            train_label = np.zeros([train_n])

            for i in range(train_n):
                train_feature[i, 0: vec_len1] = latent_m[train_samples[i, 0], :]
                train_feature[i, vec_len1: (vec_len1 + vec_len2)] = latent_d[train_samples[i, 1], :]

                train_label[i] = train_samples[i, 2]

            test_N = test_samples.shape[0]
            test_feature = np.zeros([test_N, vec_len1 + vec_len2])
            test_label = np.zeros(test_N)

            for i in range(test_N):
                test_feature[i, 0: vec_len1] = latent_m[test_samples[i, 0], :]
                test_feature[i, vec_len1: (vec_len1 + vec_len2)] = latent_d[test_samples[i, 1], :]

                test_label[i] = test_samples[i, 2]

            data = {'x_train': train_feature,
                    'y_train': train_label,
                    'x_test': test_feature,
                    'y_test': test_label,
                    }
            # np.save('./data_for_class/mse_del_w.npy', data)

            model = xgboost.XGBClassifier()
            # model = KNeighborsClassifier()
            # model = svm = SVC(kernel='rbf', probability=True)

            model.fit(train_feature, train_label)
            pre_result = model.predict(test_feature)
            pre_test_proba = model.predict_proba(test_feature)[:, 1]

            viz = metrics.plot_roc_curve(model, test_feature, test_label,
                                         name='ROC fold {}'.format(j),
                                         color=colors[j],
                                         alpha=0.6, lw=2, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            # fpr, tpr, _ = metrics.roc_curve(test_label, pre_test_proba)
            # auc = metrics.auc(fpr, tpr)
            #
            # plt.plot(fpr, tpr,
            #          lw=1, label='ROC curve (area = %0.4f)' % auc)
            #
            # out_results.append((test_label, pre_result, pre_test_proba))

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        all_out.append([w_aux1s[l1], w_aux2s[l2], w_w, mean_auc, std_auc])

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic")
        ax.legend(loc="lower right")
        plt.show()
        plt.close(fig)
    #     plt.savefig('./opt/fig/' + suffix + '.png', dpi=300)
    #
    # np.save('./opt/out/all_w_weight_ROC.npy', all_out)

xx = list(map(lambda x: '{:.2f}'.format(x), np.unique(np.array(all_out)[:, 0])))
yy = list(map(lambda x: '{:.2f}'.format(x), np.unique(np.array(all_out)[:, 1])))
data = pd.DataFrame(np.array(all_out)[:, 3].reshape(10, -1))
data.set_axis(xx, axis=0, inplace=True)
data.set_axis(yy, axis=1, inplace=True)

sns.set_context({"figure.figsize": (8, 8)})

sns.heatmap(data=data, square=True, vmin=0.96, vmax=0.97, fmt='.2%',
            linewidth=0.5, annot=True, cmap='RdBu_r',
            cbar_kws={"format": ticker.PercentFormatter(xmax=1, decimals=2)},
            annot_kws={"fontsize": 8})
plt.xlabel('aux2')
plt.ylabel('aux1')

