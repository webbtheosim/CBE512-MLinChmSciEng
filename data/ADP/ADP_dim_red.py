import os
import sys
import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap,TSNE

#There are 500 configurations in the .xyz trajectory file for Alanine Dipeptide.  You can visualize these configurations using VMD or Ovito.

#This array will hold 500 timeslices or 22 atoms and each of their coordinates.
txyz = np.zeros( (500,22,3) , dtype = float)

#Read in coordinates from ADPtrj.xyz file
tcount = -1 
acount = 0
with open('ADPtrj.xyz','r') as f:
    for i,line in enumerate(f):
        templine = line.split()
        if len(templine) == 1:
            tcount += 1
            acount = 0
        elif len(templine) == 4:
            txyz[tcount,acount,0] = float(templine[1])
            txyz[tcount,acount,1] = float(templine[2])
            txyz[tcount,acount,2] = float(templine[3])
            acount += 1
        else: 
            continue

#Read in the two important dihedral CVs associated with ADP from the trajectory
tdihedral = np.zeros( (500,2) , dtype = float)
tcount = -1
with open('dih.dump','r') as f:
    for i,line in enumerate(f):
        templine = line.split()
        if 'TIMESTEP' in line:
            tcount += 1
        if '5 7 9 15' in line:
            tdihedral[tcount,0] = np.cos(float(templine[-1])*(math.pi/180))
        if '7 9 15 17' in line:
            tdihedral[tcount,1] = np.cos(float(templine[-1])*(math.pi/180))

#Compute distance matrix between all atoms
#Extract upper triangle of distance matrix and flatten it into a feature vector
#td = np.zeros( (500,22,22) , dtype = float)
tdflat = np.zeros( (500,231) , dtype = float)
indices = np.triu_indices(22,1)
for i in range(len(txyz)):
    tdflat[i] = cdist(txyz[i],txyz[i],metric='euclidean')[indices]

#What of these features can you get rid of? There are certain distances that are constant (because the hydrogens are constrained) and exhibit no variance, so we can just get rid of those ones, though I don't do that here. 

#Now we can normalize all of the distances in the feature space by subtracting the mean and dividing by std of each feature.
scaler = StandardScaler()
tdflat_norm = scaler.fit_transform(tdflat)

#Then we can do a PCA and keep the two largest components
pca = PCA(n_components=2)
pca.fit(tdflat_norm)

#Apply the PCA transformation matrix to the data
tdflat_trans = pca.transform(tdflat_norm)

#Plot the two principal components and color each circle according to the value of the physical dihedral extracted from the trajectory.
fig, axs = plt.subplots(1,3)
sc1 = axs[0].scatter(tdflat_trans[:,0],tdflat_trans[:,1],c=tdihedral[:,1],vmin=-1,vmax=1)
axs[0].set_xlabel('Component 1')
axs[0].set_ylabel('Component 2')
axs[0].set_title('PCA')
plt.colorbar(sc1)

#Do the same thing using Isomap
Isomap_embedding = Isomap(n_components=2)
tdflat_trans_isomap = Isomap_embedding.fit_transform(tdflat_norm)
sc2 = axs[1].scatter(tdflat_trans_isomap[:,0],tdflat_trans_isomap[:,1],c=tdihedral[:,1],vmin=-1,vmax=1)
axs[1].set_xlabel('Component 1')
axs[1].set_title('Isomap')

#Do the same thing using t-SNE
TSNE_embedding = TSNE(n_components=2)
tdflat_trans_tsne = TSNE_embedding.fit_transform(tdflat_norm)
sc2 = axs[2].scatter(tdflat_trans_tsne[:,0],tdflat_trans_tsne[:,1],c=tdihedral[:,1],vmin=-1,vmax=1)
axs[2].set_xlabel('Component 1')
axs[2].set_title('t-SNE')

#Plot all three.  Obviously, you can use whatever dimensionality reduction technique you want.
plt.tight_layout()
plt.show()
