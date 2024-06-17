import numpy as np
import matplotlib.pyplot as plt


data = np.load('./metrics/CPU_256_128_64_1000_aux_reconst_KL.npy', allow_pickle=True).item()
kl = np.array(data['train_kl'])
aux1 = np.array(data['train_aux1'])
aux2 = np.array(data['train_aux2'])
base = np.array(data['train_baseloss'])
loss = np.array(data['train_loss'])
wloss = np.array(data['train_wloss'])
auc = np.array(data['train_auc'])
acc = np.array(data['train_acc'])

val_auc = np.array(data['val_auc'])
val_ap = np.array(data['val_ap'])
test_auc = np.array(data['test_auc'])
test_ap = np.array(data['test_ap'])


x = range(1000)

plt.plot(x, base)
plt.plot(x, kl)
plt.plot(x, aux1)
plt.plot(x, aux2)

plt.legend(['base', 'kl', ' aux1', 'aux2'])