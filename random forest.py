import pandas as pd
import sklearn
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder




df=pd.read_csv('iris.csv')

l= LabelEncoder()
df['variety']=l.fit_transform(df['variety'])


X=df.drop('variety',axis=1)
Y=df['variety']

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.20)

rf=RandomForestClassifier(n_estimators=5)
# n_estimators= The required number of trees in the Random Forest. The default value is 10. We can choose any number but need to take care of the overfitting issue.
# criterion= It is a function to analyze the accuracy of the split. Here we have taken "entropy" for the information gain.
rf.fit(xtrain,ytrain)
prediction=rf.predict(xtest)


cm=confusion_matrix(ytest,prediction)


cr=classification_report(ytest,prediction)
print(cr)

f=metrics.f1_score(ytest,prediction,average='weighted')



def auc_roc_plot(ytest, prediction):
        fpr, tpr, thresholds = roc_curve(ytest, prediction, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')

        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

probs = rf.predict_proba(xtest)

probs = probs[:, 1]


auc_roc_plot(ytest, probs)

cm=confusion_matrix(ytest,prediction)
fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(6,6))
plt.xlabel('prediction')
plt.ylabel('actual values')
plt.title('confusion matrix')
plt.show()



