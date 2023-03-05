import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



df=pd.read_csv('iris.csv')

l=LabelEncoder()

df['variety']=l.fit_transform(df['variety'])

X=df.values[:,0:4]
Y=df.values[:,4]



xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=.40)

svm=SVC(probability=True)

svm.fit(xtrain,ytrain)

prediction=svm.predict(xtest)
ac=accuracy_score(ytest,prediction)
print(ac)


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


probs = svm.predict_proba(xtest)

probs = probs[:, 1]

auc_roc_plot(ytest, probs)

cm = confusion_matrix(ytest, prediction)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6))
plt.xlabel('prediction')
plt.ylabel('actual values')
plt.title('confusion matrix')
plt.show()

#finding incorrect value

# sas=pd.DataFrame({'Actual':ytest},{'Predicted':prediction})
# inv=sas[sas['Actual'] != sas['Predicted']]
# print(inv)




