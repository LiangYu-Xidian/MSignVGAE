import numpy as np
import pandas as pd


data = pd.read_csv('./phendb/raw_data.txt', sep='\t')
md = data.iloc[:, [1,2,-1]].drop_duplicates()
microbe = md.iloc[:, 0]
disease = md.iloc[:, 1]
microbe_u = microbe.unique()
disease_u = disease.unique()
microbe_u.sort()
disease_u.sort()
np.save('./phendb/disease_names.npy', disease_u)
np.save('./phendb/microbe_names.npy', microbe_u)

mic_dic = {microbe_u[i]:i for i in range(microbe_u.shape[0])}
dis_dic = {disease_u[i]:i for i in range(disease_u.shape[0])}
# np.save('./phendb/mic_dic.npy', mic_dic)
# np.save('./phendb/dis_dic.npy', dis_dic)

ass = np.zeros((microbe_u.shape[0], disease_u.shape[0]))
md_np = md.to_numpy()
for item in md_np:
    if item[2] == 'Decrease':
        ass[mic_dic[item[0]], dis_dic[item[1]]] = -1
    else:
        ass[mic_dic[item[0]], dis_dic[item[1]]] = 1
pd.DataFrame(ass, index=microbe_u, columns=disease_u).to_csv('./phendb/sign_final_ass.csv')

temp_dic = {item[0]:item[1] for item in ccc}

dis_dic = np.load('./phendb/dis_dic.npy', allow_pickle=True).item()
out = []
for item in dis_dic.keys():
    if item in temp_dic.keys():
        out.append([item, temp_dic[item]])
    else:
        out.append([item, ''])

out_pd = pd.DataFrame(out, columns=['name', 'meshid'])



data = pd.read_csv('./phendb/raw_data.txt', sep='\t')



















