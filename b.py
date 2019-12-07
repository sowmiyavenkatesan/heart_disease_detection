import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math
import sklearn
df=pd.read_csv("heart_disease_dataset.csv")
sns.countplot(x="num",data=df)
sns.countplot(x="num",hue="sex",data=df)
x=df.drop('num',axis=1)
y=df['num']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
prediction=logmodel.predict(x_test)
print(prediction)
confusion_matrix(y_test,prediction)
print(accuracy_score(y_test,prediction))
