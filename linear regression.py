import numpy as np
import pandas as pd
import snss as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

#'LOGISTIC REGRESSION IS DIFFRENET FROM OLINEAR REGRESSION'

df=pd.read_csv('ratings.csv')
print(df.head())
X=df['movieId']
Y=df['rating']

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=.10)

reg=LinearRegression()

reg.fit(X_train.values.reshape(-1,1),Y_train.values.reshape(-1,1))
#
predict=reg.predict(X_test.values.reshape(-1,1))
print(predict)
print(reg.score(X_test.values.reshape(-1,1),Y_test.values.reshape(-1,1)))
#print('classification report',classification_report(Y_test.values.reshape(-1,1),predict.values.reshape(-1,1)))