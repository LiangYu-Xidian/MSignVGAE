from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

digits = load_digits()
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
X_pca = PCA(n_components=2).fit_transform(digits.data)

plt.rc('font',family='Times New Roman')

ckpt_dir = "images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, label="t-SNE")
plt.legend()
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, label="PCA")
plt.legend()

import numpy as np
import pandas as pd
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

suffix = '3ss_1ms_ww1.0_0.10_0.05_1.00'
latent_m = np.load('./phendb/opt/latent/latent_m_mse_' + suffix + '.npy')
latent_d = np.load('./phendb/opt/latent/latent_d_loss_' + suffix + '.npy')
A = pd.read_csv('./phendb/final_ass.csv', index_col=0)

A = pd.read_csv('./peryton/final_ass.csv', index_col=0)
m, d = A.shape
suffix = '3ss_1ms_ww1.0_0.50_0.15_1.00'
latent_m = np.load('./peryton/opt/latent/latent_m_mse_' + suffix + '.npy')
latent_d = np.load('./peryton/opt/latent/latent_d_loss_' + suffix + '.npy')
disease = pd.read_csv('./peryton/disease.csv',index_col=0).to_numpy()[:,0]

dis_index = np.array(list(A.columns))
mic_index = np.array(list(A.index))
mic_i = {mic_index[i]: i for i in range(mic_index.shape[0])}
dis_i = {dis_index[i]: i for i in range(dis_index.shape[0])}

item = 'Escherichia coli'
item1 = 'Fusobacteriaceae'
disease_label = A.loc[item].to_numpy().astype(int)

sim_m, sim_d = np.load('./peryton/sim_m.npy'), np.load('./peryton/sim_d.npy')

d_tsne = TSNE(n_components=2, random_state=42).fit_transform(latent_d)
# d_raw = TSNE(n_components=2, random_state=42).fit_transform(A.T)
d_raw = TSNE(n_components=2, random_state=42).fit_transform(sim_d)

m_tsne = TSNE(n_components=2, random_state=42).fit_transform(latent_m)
# m_raw = TSNE(n_components=2, random_state=42).fit_transform(A)
m_raw = TSNE(n_components=2, random_state=42).fit_transform(sim_m)

m_pca = PCA(n_components=2).fit_transform(latent_m)
m_pca_raw = PCA(n_components=2).fit_transform(sim_m)

# B = pd.read_csv('./phendb/final_ass.csv', index_col=0).to_numpy()
# raw = TSNE(n_components=2, random_state=42)
# m_raw = raw.fit_transform(B)
Ecoli_label_list = ['non E.coli', 'E.coli']
Ecoli_label_list = ['non-Fuso', 'Fuso']

plt.figure(figsize=(10, 5))
plt.subplot(121)
for i in [0, 1]:
    plt.scatter(d_tsne[disease_label == i, 0], d_tsne[disease_label == i, 1], label=Ecoli_label_list[i], c=colors[i])
plt.title('disease_latent')
plt.legend(loc="lower right")
plt.subplot(122)
for i in [0, 1]:
    plt.scatter(d_raw[disease_label == i, 0], d_raw[disease_label == i, 1], label=Ecoli_label_list[i], c=colors[i])
plt.title('disease_raw')
plt.legend(loc="lower right")

pairs = pd.read_csv('./validation/all/from common drug to predict new_drop_dup.csv', index_col=0)
temp = pairs.disease.value_counts()
topk = np.array(temp.index[:5])
i = 1
microbe_label = A[topk[i]].to_numpy().astype(int)
top1_label_list = ['non '+topk[i], topk[i]]

microbe_label = A.copy().to_numpy().T[1]
top1_label_list = ['non-alz', 'alz']

plt.figure(figsize=(10, 5))
plt.subplot(121)
for i in [0, 1]:
    plt.scatter(m_tsne[microbe_label == i, 0], m_tsne[microbe_label == i, 1], label=top1_label_list[i], c=colors[i])
plt.title('microbe_latent')
plt.legend(loc="lower right")
plt.subplot(122)
for i in [0, 1]:
    plt.scatter(m_raw[microbe_label == i, 0], m_raw[microbe_label == i, 1], label=top1_label_list[i], c=colors[i])
plt.title('microbe_raw')
plt.legend(loc="lower right")


plt.rcParams.update({"font.size":20})

pred_alz = pd.read_csv('./case study/alz_pred50_peryton.csv', index_col=0)
temp = pred_alz[pred_alz['proba'] > 0.9].to_numpy()[:, 0]
pred_index = np.array(list(map(lambda x:mic_i[x], temp)))
microbe_label = A.copy().to_numpy().T[1]
microbe_label[pred_index] = 2
label_list = ['non-alz', 'alz', 'pred_alz']
plt.figure(figsize=(10, 5))
plt.subplot(121)
for i in [0, 1, 2]:
    if i == 2:
        plt.scatter(m_tsne[microbe_label == i, 0], m_tsne[microbe_label == i, 1], label=label_list[i], c=colors[i], alpha=0.5)
        continue
    plt.scatter(m_tsne[microbe_label == i, 0], m_tsne[microbe_label == i, 1], label=label_list[i], c=colors[i])
plt.title('microbe_latent', fontsize=20)
plt.xlabel('dim_1',fontsize=14)
plt.ylabel('dim_2', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.subplot(122)
for i in [0, 1, 2]:
    if i == 2:
        plt.scatter(m_raw[microbe_label == i, 0], m_raw[microbe_label == i, 1], label=label_list[i], c=colors[i], alpha=0.5)
        continue
    plt.scatter(m_raw[microbe_label == i, 0], m_raw[microbe_label == i, 1], label=label_list[i], c=colors[i])
plt.title('microbe_raw',fontsize=20)
plt.xlabel('dim_1', fontsize=14)
plt.ylabel('dim_2', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="lower right", fontsize=14)
##########################################################
# cluster

from sklearn.cluster import KMeans

# latent_m = np.load('./phendb/opt/latent/latent_m_mse_3ss_1ms_ww1.0_0.05_0.40_1.00.npy')
# m_tsne = TSNE(n_components=2, random_state=42)
# m = m_tsne.fit_transform(latent_m)
#
# model = KMeans(n_clusters=3)
# y_pred = model.fit_predict(m)
#
# import sklearn.cluster as sc
# y_pred = sc.SpectralClustering(gamma=1, n_clusters=3).fit_predict(m)
#
# plt.figure()
# plt.scatter(m[:, 0], m[:, 1], c=y_pred, cmap='brg')
#
# from sklearn.cluster import AgglomerativeClustering
#
# y_pred = AgglomerativeClustering(n_clusters = 3, linkage = 'ward').fit_predict(latent_m)
#
# from sklearn.cluster import DBSCAN
# y_pred = DBSCAN(eps=0.4, min_samples=8).fit_predict(latent_m)
#
#
# from sklearn.mixture import GaussianMixture
# y_pred = GaussianMixture(n_components=3).fit_predict(latent_m)









































