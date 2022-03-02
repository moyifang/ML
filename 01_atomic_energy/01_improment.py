import numpy as np

# Loading the dataset
dataset = np.load('D:/ds/ML/0224_chem/qm7x_eq.npz', allow_pickle=True)
# What is inside?
for f in dataset.files:
    print(f)

# Importing data
EAT = dataset['EAT'] # atomization energy
xyz = dataset['xyz'] # Cartesian coordinates
Z = dataset['Z'] # atomic indexes

n_molecules = len(xyz)
n_z = len(Z)
print('The number of molecules in the dataset is {:d}.'.format(n_molecules))
print('The number of atomic indexes in the dataset is {:d}.'.format(n_z))

import periodictable as per
element_obj = per.elements

import pandas as pd
ELEMENT = pd.read_csv('https://gist.githubusercontent.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/raw/1d92663004489a5b6926e944c1b3d9ec5c40900e/Periodic%2520Table%2520of%2520Elements.csv' )


'''ELE_E = ELEMENT.Electronegativity.fillna(0)
ELE_D = ELEMENT.Density.fillna(0)
ELE=[]
for i in range(len(ELE_E)):
    ELE.append(ELE_E[i] / np.linalg.norm(ELE_E[i]))
print(ELE[2])'''


density = []
for element in Z:
    temp = []
    for atom in range(len(element)):
        ele=per.elements[element[atom]].density
        #ele = ELE[element[atom]+1]
        temp.append(ele)
    density.append(temp)

print(density[1])


from tqdm import tqdm
from scipy.spatial.distance import pdist

Z = density
# Descriptor
def descriptor(R):
    nconfig = len(R)
    D = []
    for i in tqdm(range(nconfig)):
        D.append(1. / pdist(R[i]))

    return D


d = descriptor(xyz)


# Making all descriptor entries of the same size
D=[]
E=[]

max_size = np.max([len(_) for _ in d])
nconfig = len(d)
max_size_e = np.max([len(_) for _ in Z])
zconfig = len(Z)
D = np.zeros((nconfig, max_size))
E = np.zeros((zconfig, max_size_e))

Ne=[]
for i in range(nconfig):
    size = len(d[i])
    D[i, :size] = d[i]

for j in range(zconfig):
    size_z = len(Z[j])
    E[j, :size_z] = Z[j]

#Nee=[]
for i in range(len(xyz)):
    X = D[i]
    Y = E[i]
    Ne.append(np.append(D[i], E[i]))
    #Nee.append(Ne[i] / np.linalg.norm(Ne[i]))


print(Ne[1])

from sklearn import linear_model
import matplotlib.pyplot as plt

# ridge regression
reg = linear_model.Ridge(alpha=0.5)
reg.fit(Ne, EAT)


rmse = np.sqrt(np.square(EAT - reg.predict(Ne)).mean())
plt.title('RMSE: {:.3f} kcal/mol'.format(rmse))
plt.scatter(EAT, reg.predict(Ne), marker='.', color='blue')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()
