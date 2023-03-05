import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import mlxtend
import snss as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  utils
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from category_encoders import BinaryEncoder
df=pd.read_csv('ratings.csv')#importing the data

# print(df.head())
#
# print(df.info())

X=df.iloc[:,1:3]#
# print('values of input features \n',X)
X=X.values# this is array

# print(X)

Y=df.iloc[:,-3]
#for y  I just need the ratingsss
# print('this is y \n ',Y)

lab=preprocessing.LabelEncoder()
y_transformed=lab.fit_transform(Y)#call fit_transform() method on our training data
# print('label encoded value of y is\n',y_transformed)

X_train, X_test, Y_train, Y_test=train_test_split(X,y_transformed,test_size=.20)#splitting the data in testing and training stes

knn=KNeighborsClassifier(n_neighbors=4)

# print(knn)

knn.fit(X_train,Y_train)# x train is used the feauters of training data and y train is used for the target
#thus knn.fit is used for  learning or stores the dataset in memory.
y_predict=knn.predict(X_test)
# print(y_predict)


# print('classification report is\n',classification_report(Y_test,y_predict))
cm = confusion_matrix(Y_test, y_predict)
# print('confsuion matrix',cm)

p,f,_=roc_curve(y_predict,Y_test,pos_label=1)
print(p+f+_)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [False, True])
# cm_display.plot()
# plt.show()
# f, ax=plt.subplots(figsize=(5,5))-