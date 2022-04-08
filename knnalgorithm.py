import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
########################
"""
K-NN algorithm assumes the similarity between the new case/data and available cases 
and put the new case into the category that is most similar to the available categories
how knn work??????
Step-1: Select the number K of the neighbors
Step-2: Calculate the Euclidean distance of K number of neighbors
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
Step-4: Among these k neighbors, count the number of the data points in each category.
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.
"""
########################
"""
We are going to use the famous iris data set for our KNN example. 
 It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify 
 the morphologic variation of Iris flowers of three related species
The dataset consists of four attributes: sepal-width, sepal-length, petal-width and petal-length. 
These are the attributes of specific types of iris plant.
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)
#To see what the dataset actually looks like, execute the following command
dataset.shape
dataset.head()
#data preprocesing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#Feature Scaling (normalization)
"""
Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated.
The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1
The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#Training and Predictions
"""
he first step is to import the KNeighborsClassifier class from the sklearn.neighbors library. 
In the second line, this class is initialized with one parameter, i.e. n_neigbours.
This is basically the value for the K. There is no ideal value for K and it is selected after testing and evaluation,
however to start out, 5 seems to be the most commonly used value for KNN algorithm.
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))