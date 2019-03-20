from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy import stats
from sklearn.cluster import KMeans
import random
import math
import re
import string
from nltk.corpus import stopwords
import pymongo
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from spherecluster import SphericalKMeans
import numpy
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import statistics
import scipy.cluster.hierarchy as shc
from sklearn import metrics
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing



def hopkins(X):
	d = X.shape[1]
	#d = len(vars) # columns
	n = len(X) #rows
	m = int(0.1 * n) # heuristic from article [1]
	nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
	rand_X = sample(range(0, n, 1), m)
 
	ujd = []
	wjd = []
	for j in range(0, m):
		u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
		ujd.append(u_dist[0][1])
		w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
		wjd.append(w_dist[0][1])
 
	H = sum(ujd) / (sum(ujd) + sum(wjd))
	if isnan(H):
		print ujd, wjd
		H = 0
 
	return H

df = pd.read_csv('ground_truth_gender.csv')
gender = df[["Gender"]]


#hopkins statistic
#print hopkins(df[['Avg_emojis','Avg_Punctuation']])

'''
#gender-feature charts
gender = df[["Gender", "Avg_emojis"]]
plt.figure()
sns.barplot(x='Avg_emojis',y='Gender',data=gender, palette='husl')
plt.xlabel("Emojis")
plt.title("Emojis/Gender")
plt.savefig("9.png")
'''

'''
#compute pearson correlation
df = df[['Gender', "images"]]
df_sex = pd.get_dummies(df['Gender'])
df = df.drop(df.columns[[0]], axis=1)
df_new = pd.concat([df, df_sex], axis=1)
x = df_new.values
correlation_matrix = np.corrcoef(x.T)
print correlation_matrix
'''

#Standardization
cl = np.array(df[['Avg_emojis', 'Avg_Punctuation']].astype(float))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cl)
'''
plt.scatter(scaled_data[:,0], scaled_data[:,1])
plt.xlabel('Emojis')
plt.ylabel('Punctuation')
plt.title('True position')
plt.savefig('trueposition.png')
plt.show()
'''



#Kmeans Clustering
kmeans = KMeans(n_clusters=2)  
kmeans.fit_predict(scaled_data)
gender['Clusters'] = kmeans.labels_
#print gender
#print gender[gender.Gender == 'F'].shape[0]
print gender.groupby(['Clusters', 'Gender']).size()
silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
print silhouette_avg
model = SilhouetteVisualizer(KMeans(2), title="Silhouette score of Kmeans clustering")    
model.fit(scaled_data) # Fit the training data to the visualizer
model.poof() # Draw/show/poof the data
plt.figure(figsize=(10, 8))
plt.scatter(scaled_data[:,0], scaled_data[:,1], c=kmeans.labels_, cmap='rainbow', label=color)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Kmeans Clustering - Exp.1')
#plt.savefig('kmeans1.png')
plt.show()


#Spherical k-means clustering
skm =SphericalKMeans(n_clusters=2)  
skm.fit_predict(scaled_data)
gender['Clusters'] = skm.labels_
#print gender
#print gender[gender.Gender == 'F'].shape[0]
print gender.groupby(['Clusters', 'Gender']).size()
plt.figure(figsize=(10, 8))
silhouette_avg = silhouette_score(scaled_data, skm.labels_)
print silhouette_avg
model = SilhouetteVisualizer(SphericalKMeans(2), title="Silhouette score of Kmeans clustering")    
model.fit(scaled_data) # Fit the training data to the visualizer
model.poof() # Draw/show/poof the data
plt.scatter(scaled_data[:,0], scaled_data[:,1], c=skm.labels_, cmap='rainbow')
centers = skm.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Spherical k-means Clustering')
plt.savefig('sphericalkmeans1.png')
plt.show()


#Hierarchical Clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='manhattan', linkage='single')  
cluster.fit_predict(scaled_data)
gender['Clusters'] = cluster.labels_
#print gender
#print gender[gender.Gender == 'F'].shape[0]
print gender.groupby(['Clusters', 'Gender']).size()
plt.figure(figsize=(10, 8))
plt.scatter(scaled_data[:,0], scaled_data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Hierarchical Clustering - Exp.1')
#plt.savefig('hierclustering2.png')
plt.show()






