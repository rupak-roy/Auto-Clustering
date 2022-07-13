
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('crime_data.csv')

#drop the missing values
dataset = dataset.dropna()

#droping the pre-existing cluster values
X = dataset.iloc[:,2:].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X=sc_X.fit_transform(X)

#spliting is not required(optional for later validation purpose)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#pip install autoelbow
from autoelbow_rupakbob import autoelbow
n =autoelbow.auto_elbow_search(X)