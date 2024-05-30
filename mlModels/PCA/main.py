from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = load_iris()
iris = load_iris()
X = pd.DataFrame(iris.data,
                 columns=iris.feature_names)
lowerdim = 2

mean = np.mean(X, axis=0)
X = X - mean
cov = np.cov(X.T, bias=1)
L, V = np.linalg.eig(cov)
inds = np.argsort(L)[::-1]
L = L[inds]
W = V[:, inds]
F = np.matmul(X, W[:,:lowerdim])
# plt.scatter_surface(F[0], F[1])
# plt.show()

#寄与率
cont = L / np.sum(L) * 100
#累積寄与率
comCont = [np.sum(cont[:i+1]) for i in range (len(L))]
# print(W[:,[0]])
# print(W[:,[1]])

tmp = np.concatenate([X,F],axis=1)
PCL = np.corrcoef(tmp.T, bias=1)[:X.shape[1], -F.shape[1]:]