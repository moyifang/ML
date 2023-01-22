import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from urllib import request

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

url='http://cs.joensuu.fi/sipu/datasets/birch3.txt'
s = request.urlopen(url).read().decode('utf8')

dfile = StringIO(s)
file_data = np.loadtxt(dfile, dtype="int", usecols=(0,1),skiprows=0)

Df = pd.DataFrame(file_data,columns = ['X','Y'])
x = file_data[:,0]
y = file_data[:,1]

fig = plt.figure(figsize=(16, 14))
ax1 = fig.add_subplot(111)
ax1.set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
ax1.scatter(x,y,c = 'r',marker = 'o',s=0.03)
plt.legend('x1')
plt.show()
##-----------------------------------------------------------KMEANS------------------------------------------------------
from sklearn.cluster import KMeans

# choose K
SSE = []
for k in range(1, 20):
    estimator = KMeans(n_clusters=k)
    estimator.fit(np.array(file_data))
    SSE.append(estimator.inertia_)
X = range(1, 20)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(file_data)
plt.figure(figsize =(20,16))
plt.scatter(x, y, c=y_pred,s=2)
plt.show()

y_pred = KMeans(n_clusters=6, random_state=9).fit_predict(file_data)
plt.figure(figsize =(20,16))
plt.scatter(x, y, c=y_pred,s=2)
plt.show()

###--------------------------------EM Algorithm--------------------------------------------
'''
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4)
gmm.fit(file_data)

# predictions from gmm
labels = gmm.predict(file_data)
Df['cluster'] = labels
plt.scatter(Df['X'], Df['Y'], c=Df['cluster'], s=0.5)
plt.show()

gmm = GaussianMixture(n_components=6)
gmm.fit(file_data)

# predictions from gmm
labels = gmm.predict(file_data)
Df['cluster'] = labels
plt.scatter(Df['X'], Df['Y'], c=Df['cluster'], s=0.5)
plt.show()'''

##-----------------------------DBSCAN-----------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors = 2)
nbrs = neigh.fit(Df[['X','Y']])
distances, indices = nbrs.kneighbors(Df[['X','Y']])
dis = np.sort(distances, axis= 0)
dis_data=dis[:,1]

plt.figure(figsize =(16, 14))
plt.plot(dis_data)
plt.title('K-distance Graph',fontsize = 20)
plt.xlabel('data sort by distance',fontsize = 14)
plt.ylabel('Epsilon', fontsize = 14)
plt.show ()

from sklearn.cluster import DBSCAN

def DBSCAN_Cluster(eps,min,data):
    y_pred = DBSCAN(eps = eps, min_samples=min, metric='euclidean').fit_predict(data)
    print(np.unique(y_pred))
    Df['label'] = y_pred

DBSCAN_Cluster(100,4,file_data)
plt.figure(figsize =(20,16))
Df_one = Df.loc[Df['label'] == -1]
Df_other = Df.loc[Df['label'] != -1]
plt.scatter(Df_one['X'], Df_one['Y'], c='gray', s=2)
plt.scatter(Df_other['X'], Df_other['Y'], c=Df_other['label'], s=2)
plt.scatter(Df_other['X'], Df_other['Y'], c=Df_other['label'], s=200)
plt.show()

DBSCAN_Cluster(7000,10,file_data)
plt.figure(figsize =(20,16))
Df_one = Df.loc[Df['label'] == -1]
Df_other = Df.loc[Df['label'] != -1]
plt.scatter(Df_one['X'], Df_one['Y'], c='gray', s=2)
plt.scatter(Df_other['X'], Df_other['Y'], c=Df_other['label'], s=0.5)
plt.show()

DBSCAN_Cluster(10000,10,file_data)
plt.figure(figsize =(20,16))
Df_one = Df.loc[Df['label'] == -1]
Df_other = Df.loc[Df['label'] != -1]
plt.scatter(Df_one['X'], Df_one['Y'], c='gray', s=2)
plt.scatter(Df_other['X'], Df_other['Y'], c=Df_other['label'], s=0.5)
plt.show()

'''
from sklearn.cluster import SpectralClustering
s = SpectralClustering(n_clusters=2,n_neighbors=10)
data=np.array(file_data)
cluster=s.fit(data)
labels = cluster.labels_
print(labels)
Df['cluster'] = labels
plt.scatter(Df['X'], Df['Y'], c=Df['cluster'], s=0.5)
plt.show()'''

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(file_data,  test_size=0.2)

from sklearn.cluster import Birch
y_pred = Birch(n_clusters = 4, threshold = 0.3, branching_factor = 20).fit_predict(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, s=2)
plt.show()

from sklearn.cluster import Birch
y_pred = Birch(n_clusters = 5, threshold = 0.2, branching_factor = 20).fit_predict(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, s=2)
plt.show()
