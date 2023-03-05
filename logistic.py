import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from category_encoders import BinaryEncoder
from sklearn.metrics import auc
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('iris.csv')

l= LabelEncoder()
df['variety']=l.fit_transform(df['variety'])
print(df['variety'])

X=df.drop('variety',axis=1)
Y=df['variety']

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.30)


lr=LogisticRegression()

lr.fit(xtrain,ytrain)

prediction=lr.predict(xtest)

print(lr.score(xtest,ytest))

cm=confusion_matrix(ytest,prediction)

b=metrics.f1_score(ytest,prediction,average='weighted')



def auc_roc_plot(y_test,prediction):
    fpr, tpr, thresholds = roc_curve(y_test,prediction,pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

probs=lr.predict_proba(xtest)
print(probs)
probs = probs[:, 1]
print(probs)


auc_roc_plot(ytest,probs)

cm=confusion_matrix(ytest,prediction)
fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(6,6))
plt.xlabel('prediction')
plt.ylabel('actual values')
plt.title('confusion matrix')
plt.show()