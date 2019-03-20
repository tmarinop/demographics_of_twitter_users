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
from scipy import stats


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

def get_nth_percent(data, n):
    index = int(len(data) * n)
    return data[index]


df = pd.read_csv('ground_truth.csv')
df_tr = df
ages = df_tr[["ID","Age"]]
#print ages

#compute hopkins
#print hopkins(df_tr[['Avg_Punctuation','Avg_Slang_words', 'Avg_urls' ]])
#compute correlation coefficients of data sets
#print numpy.corrcoef(df_tr['Age'],df_tr['Sum_tweets'])

#distribution charts
plt.xlabel('Age')
plt.ylabel('Images')
plt.scatter(df_tr['Age'], df_tr['Avg_images'], alpha=0.5)
plt.show()



#plt.plot(ages['Age'])
plt.hist(df_tr['Age'],df_tr['Avg_Slang_words'])
plt.xlabel('Age')
plt.ylabel('Sum emojis')
plt.show()


#Standardzation
cl = np.array(df_tr[['Avg_Punctuation','Avg_Slang_words', 'Avg_urls']].astype(float))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cl)
#pca = PCA(n_components=2)
#d = pca.fit_transform(df_tr[['Avg_Punctuation','Avg_Slang_words', 'Avg_urls']])
#plt.scatter(d[:,0], d[:,1])  # plot points with cluster dependent colors
#plt.show()

pca = PCA(n_components=2)
d = pca.fit_transform(scaled_data)
#plt.scatter(d[:,0], d[:,1])  # plot points with cluster dependent colors
#plt.show()



#list1 = list(ages.groupby(['Clusters'])['Age'])
#print [i[1].tolist() for i in list1]




#min_max_scaler = preprocessing.MinMaxScaler()
#datapoints = min_max_scaler.fit_transform(cl)


#kmeans clustering
range_n_clusters = [4]
sse = {}
for n_clusters in range_n_clusters:
	kmeans = SphericalKMeans(n_clusters=n_clusters).fit(scaled_data)
	silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)

	
 	#model = SilhouetteVisualizer(KMeans(n_clusters), title="Silhouette score of Kmeans clustering")    
 	#model.fit(scaled_data) # Fit the training data to the visualizer
 	#model.poof() # Draw/show/poof the data
 	#print "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
 	
 	#sse[n_clusters] = kmeans.inertia_ 
	#centeroids = kmeans.cluster_centers_
	#print centeroids
	ages['Clusters'] = kmeans.labels_
	#iterations = kmeans.n_iter_
	#print iterations
	#print metrics.calinski_harabaz_score(scaled_data, kmeans.labels_)  


	print ages.groupby(['Clusters'])['Age'].mean()
	print "Std", ages.groupby(['Clusters'])['Age'].std()
	x=list(ages.groupby(['Clusters'])['Age'])
	for t in x:
		mydic = {}
		for age in t[1]:
			if age not in mydic:
				mydic[age]=0
			mydic[age]+=1
		x=[]
		y=[]
		for age in mydic:
			x.append(age)
			y.append(mydic[age])
		#plt.hist(t[1],label="Cluster {}".format(t[0]))
		#print statistics.median(x)
		#sortedages = sorted(x)
		#print 'funn',get_nth_percent(sortedages,0.10)
		#print 'funn',get_nth_percent(sortedages,0.90)
		plt.plot(x,y,label="Cluster {}".format(t[0]))
	plt.xlabel('Age')
	plt.ylabel('Number of users')
	plt.title('Spherical k-Means Clustering')

	plt.legend()
	plt.show()
	'''
	pca = PCA(n_components=2)
	d = pca.fit_transform(scaled_data)


	cluster_lists = {}
	for cluster in range(n_clusters):
		cluster_lists[cluster] = []

	for index, point in enumerate(d):
		cluster_lists[kmeans.labels_[index]].append(point)

	for key in cluster_lists:
		x, y = [], []
		for point in cluster_lists[key]:
			x.append(point[0])
			y.append(point[1])

		cluster_lists[key] = (x, y)
	plt.figure()
	for label in cluster_lists:
		tup = cluster_lists[label]
		plt.scatter(tup[0], tup[1], label="Cluster {}".format(label))

	plt.legend()
	plt.title('Spherical k-Means Clustering')

	#plt.show()

	#print len(x)
	#print [t[1] for t in x]
	#plt.figure()
#plt.figure()
#plt.plot(list(sse.keys()), list(sse.values()))
#plt.xlabel("Number of cluster")
#plt.ylabel("SSE")
#plt.show()


	


#print silhouette

#print x

#print 'Centeroids: ', centeroids
#print 'Labels: ', kmeans.labels
#print 'Iterations: ', iterations



#hierachical clustering
#plt.figure(figsize=(10, 7))  
#plt.title("Features Dendogram")  
#dend = shc.dendrogram(shc.linkage(scaled_data, method='ward'))
#plt.show() 
'''
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(scaled_data)
ages['Clusters'] = cluster.labels_
print ages.groupby(['Clusters'])['Age'].mean()

plt.figure(figsize=(10, 8))
plt.scatter(scaled_data[:,1], scaled_data[:,0], c=cluster.labels_, cmap='prism')  # plot points with cluster dependent colors
plt.legend()

plt.show()

#hierarchical clustering
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for n_clusters in range_n_clusters:
	cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(scaled_data)
	ages['Clusters'] = cluster.labels_
	print cluster.labels_
	#print ages.groupby(['Clusters'])['Age'].mean()
	#print "Std", ages.groupby(['Clusters'])['Age'].std()
	x=list(ages.groupby(['Clusters'])['Age'])
	pca = PCA(n_components=2)
	d = pca.fit_transform(df_tr[['Avg_Punctuation','Avg_Slang_words', 'Avg_urls']])
	cluster_lists = {}
	for cluster in range(n_clusters):
		cluster_lists[cluster] = []

	for index, point in enumerate(d):
		print index,point 
		#print cluster.labels_
		#cluster_lists[cluster.labels_[index]].append(point)

	for key in cluster_lists:
		x, y = [], []
		for point in cluster_lists[key]:
			x.append(point[0])
			y.append(point[1])

		cluster_lists[key] = (x, y)

	plt.figure()
	for label in cluster_lists:
		tup = cluster_lists[label]
		plt.scatter(tup[0], tup[1], label="Cluster {}".format(label))

	plt.legend()
	plt.show()



#Sphericalkmeans clustering
skm = SphericalKMeans(n_clusters=4)
skm.fit(scaled_data)

centeroids2 = skm.cluster_centers_
ages['Clusters2'] = skm.labels_
iterations2 = skm.n_iter_
print ages.groupby(['Clusters2'])['Age'].mean()


#print 'Iterations: ', skm.intertia_

range_n_clusters = [2, 3]
sse = {}
for n_clusters in range_n_clusters:
	skm = SphericalKMeans(n_clusters=n_clusters)
	skm.fit(scaled_data)
	silhouette_avg = silhouette_score(scaled_data, skm.labels_)
	#print "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg

	
 	#model = SilhouetteVisualizer(KMeans(n_clusters), title="Silhouette score of Kmeans clustering")    
 	#model.fit(scaled_data) # Fit the training data to the visualizer
 	#model.poof() # Draw/show/poof the data
 
 	#sse[n_clusters] = skm.inertia_ 
	centeroids = skm.cluster_centers_
	#print centeroids
	ages['Clusters'] = skm.labels_
	iterations = skm.n_iter_
	#print iterations
	

	print ages.groupby(['Clusters'])['Age'].mean()
	print "Std", ages.groupby(['Clusters'])['Age'].std()
	x=list(ages.groupby(['Clusters'])['Age'])
	for t in x:
		mydic = {}
		for age in t[1]:
			if age not in mydic:
				mydic[age]=0
			mydic[age]+=1
		x=[]
		y=[]
		for age in mydic:
			x.append(age)
			y.append(mydic[age])
		#plt.hist(t[1],label="Cluster {}".format(t[0]))
		#print statistics.median(x)

		plt.plot(x,y,label="Cluster {}".format(t[0]))
	plt.xlabel('Age')
	plt.ylabel('Number of users')
	plt.legend()
	plt.show()




