# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:38:12 2021

@author: Sai Teja
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Sai Teja\Downloads\Machine Learning A-Z (Codes and Datasets)\project\placements.csv')


# Encoding Categorial Data
encoder = LabelEncoder()
columns_to_encode = ['gender','can work long time before system?','self-learning capability?','Extra-courses did','talenttests taken?','Internship?','Trained for GATE?       (yes or no)','Taken inputs from seniors or elders','interested in games','worked in teams ever?','Introvert','placed?(yes/no)']
for column in columns_to_encode:
    dataset[column] = encoder.fit_transform(dataset[column])

#Dividing Data Into Dependent and Independent Variables
X = dataset.iloc[:, 0:40].values
y = dataset.iloc[:, -1].values

#Priority Encoding For Independent Variables
encode_values = {'shell programming': 5, 'cloud computing': 20,'excellent': 20, 'system developer': 10, 'higherstudies': 5, 'stubborn': 5, 'Management': 15, 'salary': 10, 'hard worker': 30, 'machine learning': 20, 'database security': 10, 'poor': 5, 'medium': 10, 'networks': 10, 'Business process analyst': 10, 'job': 20, 'gentle': 10, 'Technical': 20, 'app development': 10, 'web technologies': 10, 'hacking': 5, 'developer': 10, 'work': 20, 'python': 20, 'data science': 15, 'testing': 10, 'smart worker': 15, 'Computer Architecture': 5, 'programming': 10, 'r programming': 10, 'parallel computing': 5, 'security': 20, 'information security': 10, 'IOT': 10, 'data engineering': 15, 'hadoop': 5, 'distro making': 5, 'game development': 10, 'Software Engineering': 15, 'system designing': 10, 'full stack': 20}
for i in range(len(X)):
    for j in range(19,40):
        if X[i][j] in encode_values:
            X[i][j]=encode_values[X[i][j]]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


 #SVM CLASSIFIER

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = classifier.predict(X_test)
    
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_svm)
print(cm)

#Accuracy For SVM
print("Accuracy for SVM Classifier is :",accuracy_score(y_test, y_pred_svm))


                   #Logistic Regression

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logistic = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_logistic)
print(cm)

#Accuracy For SVM
print("Accuracy for Logistic Regression Classifier is :",accuracy_score(y_test, y_pred_logistic))



                  #K_Nearest_Neighbours

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_KNN)
print(cm)

#Accuracy For KNN
print("Accuracy for K_Nearest-Neighbour Classifier is :",accuracy_score(y_test, y_pred_KNN))


                    #Naive Bayes

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NaiveBayes = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_NaiveBayes)
print(cm)

#Accuracy For Naive Bayes
print("Accuracy for Naive Bayes classifier is :",accuracy_score(y_test, y_pred_NaiveBayes))

                      #Decision Tree

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_DecisionTree = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_DecisionTree)
print(cm)

#Accuracy For Decision Tree
print("Accuracy for Decision Tree Classifier is :",accuracy_score(y_test, y_pred_DecisionTree))


                    #Random Forest

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RandomForest = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_RandomForest)
print(cm)

#Accuracy For Random Forest
print("Accuracy for Random Forest classifier is :",accuracy_score(y_test, y_pred_RandomForest))
