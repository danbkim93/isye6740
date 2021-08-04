#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import time
import statistics
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifier
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#CV score
ridge = []
logistic = [] 
neighbors = [] 
bayes = []
support = []
neural = [] 
ada = []

#accuracy
ridge_accuracy = []
logistic_accuracy = [] 
neighbors_accuracy = [] 
bayes_accuracy = []
support_accuracy = []
neural_accuracy = [] 
ada_accuracy = []

#
ridge_f1 = []
logistic_f1 = [] 
neighbors_f1 = [] 
bayes_f1 = []
support_f1 = []
neural_f1 = [] 
ada_f1 = []


# accuracies
accuracies_avg = [] 
avg_cv_score = []
f1_avg= [] 



def run_algorithms(X_train, X_test, y_train, y_test,X, y):
    start = time.time()
    #cv
    total_scores = []
    
    total_scores_acc = [] 
    total_scores_f1 = []
    
    total_f1 = [] 
    print("Ridge regression classifier")
    print("---------------------")
    clf = RidgeClassifier()
    cv_score = cross_val_score(clf, X, y, cv=10)
    clf.fit(X_train, y_train)
    ridge_pred = clf.predict(X_test)
    print(cv_score.mean())
    total_scores.append(cv_score.mean())
    acc = scores(y_test, ridge_pred)
    f1 = f1_score(y_test, ridge_pred, average='weighted')
    
    
    ridge.append(cv_score.mean())
    ridge_f1.append(f1)
    ridge_accuracy.append(acc)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    
    clf = LogisticRegression(random_state=0)
    cv_score = cross_val_score(clf, X, y, cv=10)
    clf.fit(X_train, y_train)
    log_pred = clf.predict(X_test) 
    print("Logistic regression")
    print("---------------------")
    total_scores.append(cv_score.mean())
    print(cv_score.mean())

    acc = scores(y_test, log_pred)
    f1 = f1_score(y_test, log_pred, average='weighted')

    logistic.append(cv_score.mean())
    logistic_f1.append(f1)
    logistic_accuracy.append(acc)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    
    print("KNN classification")
    print("---------------------")
    neigh = KNeighborsClassifier(n_neighbors=28)
    cv_score_neigh = cross_val_score(neigh, X, y, cv=10)
    print(cv_score_neigh.mean())
    total_scores.append(cv_score_neigh.mean())

    neigh.fit(X_train, y_train)
    neigh.score(X_test, y_test)
    neigh_pred =neigh.predict(X_test)
    acc = scores(y_test, neigh_pred)
    f1 = f1_score(y_test, neigh_pred, average='weighted')

    neighbors.append(cv_score_neigh.mean())
    neighbors_f1.append(f1)
    neighbors_accuracy.append(acc)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    
    print("Gaussian Naive Bayes") 
    print("---------------------")
    gnb = GaussianNB()
    cv_score_Gnb = cross_val_score(gnb, X, y, cv=10)
    print(cv_score_Gnb.mean())
    total_scores.append(cv_score_Gnb.mean())
    bayes.append(cv_score_Gnb.mean())

    gnb_pred = gnb.fit(X_train, y_train).predict(X_test)
    acc = scores(y_test, gnb_pred)
    f1 = f1_score(y_test, gnb_pred, average='weighted')

    gnb = GaussianNB()
    bayes_accuracy.append(acc)
    bayes_f1.append(f1)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    
    print("Support Vector Machine")
    print("---------------------")
    clf = svm.SVC()
    cv_score_svm = cross_val_score(clf, X, y, cv=10)
    print(cv_score_svm.mean())
    total_scores.append(cv_score_svm.mean())
    support.append(cv_score_svm.mean())

    clf.fit(X_train, y_train)
    svm_pred = clf.predict(X_test)
    acc = scores(y_test, svm_pred)
    f1 = f1_score(y_test, svm_pred, average='weighted')

    support_f1.append(f1)
    support_accuracy.append(acc)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    print("Neural Net")
    print("---------------------")
    clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=5000)
    cv_score_neural = cross_val_score(clf, X, y, cv=10)
    
    print(cv_score_neural.mean())
    total_scores.append(cv_score_neural.mean())
    
    neural.append(cv_score_neural.mean())

    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    acc = scores(y_test, clf_pred)
    f1 = f1_score(y_test, clf_pred, average='weighted')

    neural_accuracy.append(acc)
    neural_f1.append(f1)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    print("AdaBoost")
    print("---------------------")
    clf = AdaBoostClassifier(n_estimators=105, random_state=0)
    cv_score_ada = cross_val_score(clf, X, y, cv=10)
    
    print(cv_score_ada.mean())
    total_scores.append(cv_score_neural.mean())
    ada.append(cv_score_neural.mean())


    clf.fit(X_train, y_train)
    ada_pred = clf.predict(X_test)
    acc = scores(y_test, ada_pred)
    f1 = f1_score(y_test, ada_pred, average='weighted')

    ada_accuracy.append(acc)
    ada_f1.append(f1)
    total_scores_acc.append(acc)
    total_scores_f1.append(f1)
    
    print("Average of CV score = ", statistics.mean(total_scores))
    print("Average of F1 score = " , statistics.mean(total_scores_f1))
    print("Average of Accuracy score = " , statistics.mean(total_scores_acc))
    avg_cv_score.append(statistics.mean(total_scores))
    accuracies_avg.append(statistics.mean(total_scores_acc))
    f1_avg.append(statistics.mean(total_scores_f1))
    print("Standard deviation of CV score = " , statistics.stdev(total_scores))
    end = time.time()
    print("Time elapsed: " , end - start)


# In[ ]:





# In[2]:


data = pd.read_csv("pima-indians-diabetes.csv")

def scores(test,pred):
    result = accuracy_score(test, pred)
    print("accuracy = " ,result )
    print(classification_report(test, pred))
    return result

dic = dict({"6":"Pregnancies",
 "148": "Glucose",
 "72": "BloodPressure",
 "35":"SkinThickness",
 "0":"Insulin",
 "33.6": "BMI",
 "0.627": "DiabetesPedigreeFunction",
 "50": "Age",
 "1": "Class"})


# In[3]:


data = data.rename(columns = dic)
data
y = data.iloc[:, 8]
X = data.iloc[: , :8]


# In[4]:


random.seed(10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


# original dataset 
random.seed(10)
run_algorithms(X_train, X_test, y_train, y_test,X, y )


# In[6]:


## PCA 
random.seed(10)
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2) 
run_algorithms(X_train_pca, X_test_pca, y_train, y_test,X_pca, y )


# In[7]:


## PCA  3 components
random.seed(10)
pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2) 
run_algorithms(X_train_pca, X_test_pca, y_train, y_test,X_pca, y )


# In[8]:


## PCA  4 components
random.seed(10)
pca = PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2) 
run_algorithms(X_train_pca, X_test_pca, y_train, y_test,X_pca, y )


# In[9]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[10]:


random.seed(10)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
run_algorithms(X_train_scaled, X_test_scaled, y_train, y_test,X_scaled, y )


# In[11]:



## PCA  2 components
random.seed(10)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
X_train_pca_scaled, X_test_pca_scaled, y_train, y_test = train_test_split(X_pca_scaled, y, test_size=0.2) 
run_algorithms(X_train_pca_scaled, X_test_pca_scaled, y_train, y_test,X_pca_scaled, y )


# In[12]:


## PCA  3 components
random.seed(10)
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
X_train_pca_scaled, X_test_pca_scaled, y_train, y_test = train_test_split(X_pca_scaled, y, test_size=0.2) 
run_algorithms(X_train_pca_scaled, X_test_pca_scaled, y_train, y_test,X_pca_scaled, y )


# In[ ]:





# In[13]:


## PCA  4 components
random.seed(10)
pca = PCA(n_components=4)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
X_train_pca_scaled, X_test_pca_scaled, y_train, y_test = train_test_split(X_pca_scaled, y, test_size=0.2) 
run_algorithms(X_train_pca_scaled, X_test_pca_scaled, y_train, y_test,X_pca_scaled, y )


# In[14]:


X_reduced  = X.drop(["SkinThickness", "BloodPressure"],axis=1)
X_pregnancies_dropped = X.drop(["Pregnancies"],axis=1)
X_all_dropped = X.drop(["SkinThickness", "BloodPressure","Age"],axis=1)


# In[15]:


random.seed(10)
scaler = preprocessing.StandardScaler().fit(X_reduced)
X_reduced_scaled = scaler.transform(X_reduced)
X_train_reduced_scaled, X_test_reduced_scaled, y_train, y_test = train_test_split(X_reduced_scaled, y, test_size=0.2)
run_algorithms(X_train_reduced_scaled, X_test_reduced_scaled, y_train, y_test,X_reduced_scaled, y )


# In[16]:


random.seed(10)
scaler = preprocessing.StandardScaler().fit(X_pregnancies_dropped)
X_pregnancies_dropped_scaled = scaler.transform(X_pregnancies_dropped)
X_train_pregnancies_dropped_scaled, X_test_pregnancies_dropped_scaled, y_train, y_test = train_test_split(X_pregnancies_dropped_scaled, y, test_size=0.2)
run_algorithms(X_train_pregnancies_dropped_scaled, X_test_pregnancies_dropped_scaled, y_train, y_test,X_pregnancies_dropped_scaled, y )


# In[17]:


random.seed(10)
scaler = preprocessing.StandardScaler().fit(X_all_dropped)
X_all_dropped_scaled = scaler.transform(X_all_dropped)
X_train_all_dropped_scaled, X_test_all_dropped_scaled, y_train, y_test = train_test_split(X_all_dropped, y, test_size=0.2)
run_algorithms(X_train_all_dropped_scaled, X_test_all_dropped_scaled, y_train, y_test,X_all_dropped_scaled, y )


# In[ ]:





# In[18]:



random.seed(10)
scaler = preprocessing.StandardScaler().fit(X_all_dropped)
X_all_dropped_scaled = scaler.transform(X_all_dropped)

pca = PCA(n_components=2)
pca.fit(X_all_dropped_scaled)

X_all_dropped_scaled_pca = pca.transform(X_all_dropped_scaled)
X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test = train_test_split(X_all_dropped_scaled_pca, y, test_size=0.2) 
run_algorithms(X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test,X_all_dropped_scaled_pca, y )


# In[19]:



random.seed(10)
pca = PCA(n_components=3)
pca.fit(X_all_dropped_scaled)
X_all_dropped_scaled_pca = pca.transform(X_all_dropped_scaled)
X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test = train_test_split(X_all_dropped_scaled_pca, y, test_size=0.2) 
run_algorithms(X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test,X_all_dropped_scaled_pca, y )


# In[20]:



random.seed(10)
pca = PCA(n_components=4)
pca.fit(X_all_dropped_scaled)
X_all_dropped_scaled_pca = pca.transform(X_all_dropped_scaled)
X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test = train_test_split(X_all_dropped_scaled_pca, y, test_size=0.2) 
run_algorithms(X_train_all_dropped_scaled_pca, X_test_all_dropped_scaled_pca, y_train, y_test,X_all_dropped_scaled_pca, y )


# In[ ]:





# In[21]:


ridge
logistic  
neighbors  
bayes
support 
neural  
ada


# In[34]:


plt.figure(figsize=(20, 10))
plt.plot(ridge)
plt.plot(logistic)
plt.plot(neighbors) 
plt.plot(bayes)
plt.plot(support)
plt.plot(neural)
plt.plot(ada)
plt.legend(["Ridge", "Logistic", "K-nearest-neighbors", "Naive Bayes", "SVM", "Neural Network",
           "AdaBoost"])
plt.ylabel("CV score")
print("ridge maximum is {} at = {}".format(max(ridge), np.argmax(ridge)))
print("logistic maximum is {} at = {}".format(max(logistic), np.argmax(logistic)))
print("knearest neighbor maximum is {} at = {}".format(max(neighbors), np.argmax(neighbors)))
print("bayes maximum is {} at = {}".format(max(bayes), np.argmax(bayes)))
print("svm maximum is {} at = {}".format(max(support), np.argmax(support)))
print("neural maximum is {} at = {}".format(max(neural), np.argmax(neural)))
print("ada maximum is {} at = {}".format(max(ada), np.argmax(ada)))


# In[40]:


plt.figure(figsize=(20, 10))
plt.plot(ridge)
plt.plot(logistic)
plt.plot(support)
plt.legend(["Ridge", "Logistic", "SVM"])
plt.ylabel("CV score")


# In[37]:


plt.figure(figsize=(20, 10))
plt.plot(ridge_accuracy)
plt.plot(logistic_accuracy)
plt.plot(neighbors_accuracy) 
plt.plot(bayes_accuracy)
plt.plot(support_accuracy)
plt.plot(neural_accuracy)
plt.plot(ada_accuracy)
plt.legend(["Ridge", "Logistic", "K-nearest-neighbors", "Naive Bayes", "SVM", "Neural Network",
           "AdaBoost"])
plt.ylabel("Accuracy")

print("ridge maximum is {} at = {}".format(max(ridge_accuracy), np.argmax(ridge_accuracy)))
print("logistic maximum is {} at = {}".format(max(logistic_accuracy), np.argmax(logistic_accuracy)))
print("knearest neighbor maximum is {} at = {}".format(max(neighbors_accuracy), np.argmax(neighbors_accuracy)))
print("bayes maximum is {} at = {}".format(max(bayes_accuracy), np.argmax(bayes_accuracy)))
print("svm maximum is {} at = {}".format(max(support_accuracy), np.argmax(support_accuracy)))
print("neural maximum is {} at = {}".format(max(neural_accuracy), np.argmax(neural_accuracy)))
print("ada maximum is {} at = {}".format(max(ada_accuracy), np.argmax(ada_accuracy)))


# In[42]:


plt.figure(figsize=(20, 10))
plt.plot(ridge_accuracy)
plt.plot(logistic_accuracy)
plt.plot(support_accuracy)
plt.plot(ada_accuracy)
plt.legend(["Ridge", "Logistic",  "SVM","AdaBoost"])
plt.ylabel("Accuracy")


# In[38]:


plt.figure(figsize=(20, 10))
plt.plot(ridge_f1)
plt.plot(logistic_f1)
plt.plot(neighbors_f1) 
plt.plot(bayes_f1)
plt.plot(support_f1)
plt.plot(neural_f1)
plt.plot(ada_f1)
plt.legend(["Ridge", "Logistic", "K-nearest-neighbors", "Naive Bayes", "SVM", "Neural Network",
           "AdaBoost"])
plt.ylabel("F1-score")

print("ridge maximum is {} at = {}".format(max(ridge_f1), np.argmax(ridge_f1)))
print("logistic maximum is {} at = {}".format(max(logistic_f1), np.argmax(logistic_f1)))
print("knearest neighbor maximum is {} at = {}".format(max(neighbors_f1), np.argmax(neighbors_f1)))
print("bayes maximum is {} at = {}".format(max(bayes_f1), np.argmax(bayes_f1)))
print("svm maximum is {} at = {}".format(max(support_f1), np.argmax(support_f1)))
print("neural maximum is {} at = {}".format(max(neural_f1), np.argmax(neural_f1)))
print("ada maximum is {} at = {}".format(max(ada_f1), np.argmax(ada_f1)))


# In[43]:


plt.figure(figsize=(20, 10))
plt.plot(ridge_f1)
plt.plot(logistic_f1)
plt.plot(support_f1)
plt.plot(ada_f1)
plt.legend(["Ridge", "Logistic",  "SVM","AdaBoost"])
plt.ylabel("Accuracy")


# In[39]:


print(
statistics.mean(ridge),
statistics.mean(logistic),
statistics.mean(neighbors),
statistics.mean(bayes),
statistics.mean(support) ,
statistics.mean(neural)  ,
statistics.mean(ada))


# In[26]:


print(
statistics.stdev(ridge),
statistics.stdev(logistic),
statistics.stdev(neighbors),
statistics.stdev(bayes),
statistics.stdev(support) ,
statistics.stdev(neural)  ,
statistics.stdev(ada))


# In[27]:





# In[ ]:





# In[29]:


print("mean  ={}, std  = {}".format(statistics.mean(ridge),statistics.stdev(ridge)))
print("mean  ={}, std  = {}".format(statistics.mean(logistic),statistics.stdev(logistic)))
print("mean  ={}, std  = {}".format(statistics.mean(support),statistics.stdev(support)))


# In[ ]:




