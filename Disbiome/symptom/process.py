import numpy as np
import pandas as pd

data1 = pd.read_csv('./Disbiome/symptom/symptom-disease-HSDN.txt', sep='\t')
data = pd.read_csv('./Disbiome/symptom/disease.csv', index_col=0)

temp_name = data[['name', 'scientific name']].to_numpy()
raw2sci_name = {item[0]: item[1] for item in temp_name}

a = set(temp_name[:, 1])
b = list(a)
b.sort()
unique_dis = np.array(b)

filtered = data1[data1['MeSH Disease Term'].isin(unique_dis)]
filtered = filtered.iloc[:, [0, 1, 3]]

mesh_sym = filtered['MeSH Symptom Term'].unique()

dis_dic = {unique_dis[i]: i for i in range(len(unique_dis))}
sym_dic = {mesh_sym[i]: i for i in range(len(mesh_sym))}

score = np.zeros((unique_dis.shape[0], mesh_sym.shape[0]))

for i in range(filtered.shape[0]):
    score[dis_dic[filtered.iloc[i, 1]], sym_dic[filtered.iloc[i, 0]]] = filtered.iloc[i, 2]

from sklearn.metrics.pairwise import cosine_similarity

all_score = np.zeros((temp_name.shape[0], mesh_sym.shape[0]))

for i in range(temp_name.shape[0]):
    all_score[i] = score[dis_dic[temp_name[i, 1]]]

all_score_cos = cosine_similarity(all_score)

out = pd.DataFrame(all_score_cos, index=temp_name[:, 0], columns=temp_name[:, 0])

out.to_csv('./Disbiome/symptom/final_symptom_disease_bio.csv')
