import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def TrainPredict(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    #decision tree
    d_tree = tree.DecisionTreeClassifier()
    d_tree.fit(x_train,y_train)
    y_pred_dtree = d_tree.predict(x_test)
    print("Decision Tree Accuracy Score:",accuracy_score(y_pred_dtree, y_test))

    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred_gnb = gnb.predict(x_test)
    print("Gaussian Naive Bayes Accuracy Score:", accuracy_score(y_pred_gnb,y_test))

    #K-Means
    kmeans = KMeans()
    kmeans.fit(x_train,y_train)
    y_pred_kmeans = kmeans.predict(x_test)
    print("K-Means Accuracy Score:", accuracy_score(y_pred_kmeans,y_test))

    #MLP-Neural Network
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    y_pred_mlp = mlp.predict(x_test)
    print("Neural Network (MLP) Accuracy Score:", accuracy_score(y_pred_mlp,y_test))

    #Logistic Regression
    logr = LogisticRegression()
    logr.fit(x_train,y_train)
    y_pred_logr = logr.predict(x_test)
    print("Logistic Regression Accuracy Score:", accuracy_score(y_pred_logr,y_test))

    #Random Forest
    randfor = RandomForestClassifier()
    randfor.fit(x_train, y_train)
    y_pred_randfor = randfor.predict(x_test)
    print("Random Forest Accuracy Score:",accuracy_score(y_pred_randfor,y_test))


df = pd.read_csv("ks2018-edited.csv")

print("\nRaw Data Size:", len(df))

df = df[df.state != -1]

print("Final Data Size:", len(df))

print("\n")

x= pd.DataFrame(df["usd_goal_real"])

y= df["state"]

print("Predicting the state of projects using goals(USD)...")
TrainPredict(x,y)

print("\n")

x = pd.DataFrame(df["usd_pledged_real"])

print("Predicting the state of projects using pledged money(USD)...")
TrainPredict(x,y)
