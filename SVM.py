import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
SVM differs from the other classification algorithms in the way that it chooses the decision boundary that maximizes the distance from the nearest data 
points of all the classes.  An SVM doesn't merely find a decision boundary;  it finds the most optimal decision boundary.
"""
# Importing the Dataset
# To read data from CSV file, the simplest way is to use read_csv method of the pandas library
bankdata = pd.read_csv("bill_authentication.csv")
# Exploratory Data Analysis
# To see the rows and columns and of the data, execute
bankdata.shape
# To get a feel of how our dataset actually looks
bankdata.head()
# Data Preprocessing
"""
all the columns of the bankdata dataframe are being stored in the X variable except the "Class" column,
which is the label column. The drop() method drops this column. only the class column is being stored in the y variable. 
At this point of time X variable contains attributes while y variable contains corresponding labels.
"""
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
"""
#Once the data is divided into attributes and labels, the final preprocessing step is to divide data into training and test sets.
#the model_selection library of the Scikit-Learn library contains ''the train_test_split method''
#that allows us to seamlessly divide data into training and test sets.
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Training the Algorithm
from sklearn.svm import SVC

"""
#Scikit-Learn contains the svm library, which contains built-in classes for different SVM algorithms.
#This class takes one parameter, which is the kernel type. 
#The kernel functions return the inner product between two points in a suitable feature space.
#example linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.
#In the case of a simple SVM we simply set this parameter as "linear" since simple SVMs can only classify linearly separable data.
"""
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
# Making Predictions
# To make predictions, the predict method of the SVC class is used.
y_pred = svclassifier.predict(X_test)
"""
classification report visualizer displays the precision, recall, F1, and support scores for the model.
confusion matrix is a table that is used to define the performance of a classification algorithm.
"""
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
