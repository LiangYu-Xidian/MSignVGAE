import requests
from lxml import etree
import pandas as pd
import gzip
import numpy as np
import time
from tqdm import tqdm

names = np.load('./Disbiome/disease_names.npy', allow_pickle=True)
term = 'Inflammatory Bowel Disease'
url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + term
r = requests.get(url)
text = r.text
html = etree.HTML(text.encode('utf-8'))
res = html.xpath('//*[@id="maincontent"]/div/div[5]/div[1]/div[2]/p/a')
href = res[0].attrib
href = href['href']

prefix = 'https://www.ncbi.nlm.nih.gov/'
url2 = prefix + href

r2 = requests.get(url2)
text = r2.text
html = etree.HTML(text.encode('utf-8'))
res = html.xpath('//*[@id="maincontent"]/div/div[5]/div/p[8]')
meshid = res[0].text
meshid = meshid.split(':')[-1].lstrip()

out = []
temp = 319
for i, item in enumerate(names[temp:]):
    term = item.strip()
    print('i:{}  item:{} '.format(i + temp, item))
    url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + term
    r = requests.get(url)
    text = r.text
    html = etree.HTML(text.encode('utf-8'))
    res = html.xpath('//*[@id="maincontent"]/div/div[5]/div[1]/div[2]/p/a')
    if not res:
        out.append([term, 0])
        continue
    href = res[0].attrib
    href = href['href']

    prefix = 'https://www.ncbi.nlm.nih.gov/'
    url2 = prefix + href

    r2 = requests.get(url2)
    text = r2.text
    html = etree.HTML(text.encode('utf-8'))
    res = html.xpath('//*[@id="maincontent"]/div/div[5]/div/p[8]')
    meshid = res[0].text
    meshid = meshid.split(':')[-1].lstrip()
    print('meshid: ', meshid)
    out.append([term, meshid])

done = []
for item in out:
    if item[1] != 0 and item[1] != '':
        done.append(item)
###########################################
# 能直接搜到

term = 'Non-alcoholic fatty liver disease '
url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + term
r = requests.get(url)
text = r.text
html = etree.HTML(text.encode('utf-8'))
# res = html.xpath('//*[@id="maincontent"]/div/div[5]/div[1]/div[2]/p/a')
res = html.xpath('//*[@id="maincontent"]/div/div[5]/div/p[8]')
meshid = res[0].text
meshid = meshid.split(':')[-1].lstrip()

direct = []
for i, item in enumerate(names):
    term = item.strip()
    if term in done:
        continue
    print('i:{}  item:{} '.format(i, item))

    url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + term
    r = requests.get(url)
    text = r.text
    html = etree.HTML(text.encode('utf-8'))
    # res = html.xpath('//*[@id="maincontent"]/div/div[5]/div[1]/div[2]/p/a')
    res = html.xpath('//*[@id="maincontent"]/div/div[5]/div/p[8]')
    if res == []:
        continue
    meshid = res[0].text
    meshid = meshid.split(':')[-1].lstrip()
    print('meshid: ', meshid)
    direct.append([term, meshid])

done2 = []
for item in direct:
    if item[1] != 0 and item[1] != '':
        done2.append(item)

done.extend(done2)
temp = np.array(done)
done_temp = {item[0]: item[1] for item in done}
out1 = []
out2 = []
for item in names:
    temp = item.strip()
    if item in done_temp.keys():
        out1.append([item, done_temp[temp]])
    else:
        out2.append([item, ''])

all_disease = pd.read_csv('./Disbiome/disease_temp.csv', index_col=0)

for i in range(374):
    ttt = all_disease.iloc[i, 1].strip()
    if ttt in done_temp.keys():
        all_disease.iloc[i, 2] = done_temp[ttt]

# all_disease.to_csv('./Disbiome/disease_temp.csv')

#############################################################################
# get scientific name from Mesh ID

data = pd.read_csv('./Disbiome/disease_temp.csv', index_col=0)
mesh = data['mesh id']
meshids = list(mesh)
id2sname = {}
temp_out = []
for i in tqdm(range(len(meshids))):
    term = meshids[i]
    url = 'https://www.ncbi.nlm.nih.gov/mesh/?term=' + term
    r = requests.get(url)
    text = r.text
    html = etree.HTML(text.encode('utf-8'))
    res = html.xpath('//*[@id="maincontent"]/div/div[5]/div/h1')
    scientific_name = res[0].text
    id2sname[term] = scientific_name
    temp_out.append([term, scientific_name])
    time.sleep(0.5)


c = pd.DataFrame(np.array(temp_out)[:, 1])
c.columns = ['scientific name']
data['scientific name'] = c
# data.to_csv('./Disbiome/disease_temp.csv')


