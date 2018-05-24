import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("ks2018-edited.csv")

print("\nRaw Data Size:", len(df))

df = df[df.state != -1]

print("Final Data Size:", len(df))

dict1 = {"goal": df["usd_goal_real"], "main_category": df["main_category"], "category": df["category"]}

x=pd.DataFrame(dict1)

x['main_category'] = x['main_category'].astype('category')
x['main_category'] = x['main_category'].cat.codes

x['category'] = x['category'].astype('category')
x['category'] = x['category'].cat.codes

print(x)

y=df["state"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


#Decision Tree
d_tree = tree.DecisionTreeClassifier()
d_tree.fit(x_train,y_train)
y_pred_dtree = d_tree.predict(x_test)
print("Decision Tree Accuracy Score:",accuracy_score(y_pred_dtree, y_test))

#Random Forest
randfor = RandomForestClassifier()
randfor.fit(x_train, y_train)
y_pred_randfor = randfor.predict(x_test)
print("Random Forest Accuracy Score:",accuracy_score(y_pred_randfor,y_test))

#Logistic Regression
logr = LogisticRegression()
logr.fit(x_train,y_train)
y_pred_logr = logr.predict(x_test)
print("Logistic Regression Accuracy Score:", accuracy_score(y_pred_logr,y_test))
