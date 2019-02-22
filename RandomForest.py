# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:07:07 2018

@author: youko
"""

'''this file is to use random forest as to predict the people who are at risk
of having AMI
'''
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
##import our datasets 
myData=pd.read_csv('deleted_numerical_data_full.csv')
del myData['Unnamed: 0']
##set attributes and target
attributes=list(myData.columns.values)
attributes.remove('AMI_FLAG')
attributes.remove('ID')
target=['AMI_FLAG']
##split the whole data set in to Active and InActive
Active=myData[myData['AMI_FLAG']==1]
InActive=myData[myData['AMI_FLAG']==0]

#trA, tsA = train_test_split(Active, test_size=0.382)
#trIn, tsIn = train_test_split(InActive, test_size=2726*1.62/97274)
##combine the test dataframe
#frames = [tsA, tsIn]
#test_set = pd.concat(frames)

##split Active and InActive into test dataset and training set
##=======================================================================
##do this modeling for the whole sample as to get the full prediction
X_test=pd.DataFrame(myData, columns=attributes)
Y_test=pd.DataFrame(myData, columns=target)
remain_In=InActive
predict_value=[]
RF_score=[]
proba=[]
##considering the happen rate of AMI in common people, we set a threshold
##of 0.7 when classifying. 0.7 was decided by trial and error.
threshold=0.7
for i in range(0,35,1):
    trIn=remain_In
    #trA,remain_A=train_test_split(Active,test_size=0.382)
    trIn,remain_In=train_test_split(trIn,test_size=1-(2726/len(InActive)))
    frames = [Active, trIn]
    training_set = pd.concat(frames)
    X_train=pd.DataFrame(training_set,columns=attributes)
    Y_train=pd.DataFrame(training_set,columns=target)
    #X_test=pd.DataFrame(myData, columns=attributes)
    #Y_test=pd.DataFrame(myData, columns=target)
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,Y_train)
    predicted_proba = clf.predict_proba(X_test)
    pred = (predicted_proba [:,1] >= threshold).astype('int')
    predict_value.append(pred)
    #RF_score.append(average_precision_score(Y_test, clf.predict(X_test)))
#predicted_proba = clf.predict_proba(X_test)
#predicted = (predicted_proba [:,1] >= threshold).astype('int')
##apply the majority voting method    
df = pd.DataFrame(predict_value)
p=df.sum(axis=0)  
p=p.to_frame()
p.index=myData.index
p.columns = ['votes']
##
sorted_list=p.sort_values('votes',ascending=False)
for i in range(0,35,1):
    df
def label (row):
   if row['votes'] >27:
      return 1
   else:
      return 0
   return 'Other'
p['Label']  = p.apply (lambda row: label (row),axis=1)
#prob=clf.predict_proba(X_test)
matrix =confusion_matrix(Y_test,p['Label'])
print(matrix)
AUC_score=metrics.roc_auc_score(Y_test,p['Label'])
print(AUC_score)

##make the AUC and ROC plot
fpr, tpr, threshold = metrics.roc_curve(Y_test, p['Label'])
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
plt.savefig('AUC_ROC_RF.png')

##give the high risk group list
profile=pd.read_csv('TAMU_FINAL_DATASET_2018.csv')
##combine the dataset together
sorted_list['ID']=profile['ID']
sorted_list['SEX']=profile['SEX_CD']
sorted_list['AGE']=profile['AGE']
sorted_list.groupby(['votes']).count()
#got the high risk group
high_risk_group=sorted_list.loc[sorted_list['votes'] == 35]
#got the risk group
risk_group=sorted_list.loc[sorted_list['votes'] < 35]

filename='high_risk_group.csv'
high_risk_group.to_csv(filename)
filename1='risk_group.csv'
risk_group.to_csv(filename1)
#print("Accuracy:",metrics.accuracy_score(Y_test, p['Label']))
#print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, p['Label']))  
#print('Mean Squared Error:', metrics.mean_squared_error(Y_test, p['Label']))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, p['Label']))) 

    # trA, tsA = train_test_split(Active, test_size=0.382)
    # trIn, tsIn = train_test_split(InActive, test_size=2726*1.62/97274)
##combine the test dataframe
     #frames = [tsA, tsIn]
    # test_set = pd.concat(frames)

##get the number of trees
#def get_ntrees():
   # accuracy=0
    #ntrees=0
    #for i in range(50,2000,50):
     #  clf=RandomForestClassifier(n_estimators=1000)
     #  clf.fit(X_train,Y_train)
      # Y_pred=clf.predict(X_test)
      # new_accuracy=metrics.accuracy_score(Y_test, Y_pred)
    #if new_accuracy > accuracy:
     #   accuracy=new_accuracy
     #   ntrees=i
   # ntrees=[accuracy,ntrees]
   # return ntrees

##get the n_estimator
#ntrees=get_ntrees()
##further split the trIn in to subsamples, together with fixed trA

#X_test=pd.DataFrame(test_set, columns=attributes)
#Y_test=pd.DataFrame(test_set, columns=target)


##======================================================================
##try to do the random forest with the whole training set 
#frames = [trA, trIn]
#training_set = pd.concat(frames)
#X_train=pd.DataFrame(training_set,columns=attributes)
#Y_train=pd.DataFrame(training_set,columns=target)
#clf=RandomForestClassifier(n_estimators=200)
#clf.fit(X_train,Y_train)
#pred=clf.predict(X_test)
#average_precision_score(Y_test, pred)          
##======================================================================


##====================================
'''
remain_trIn,chosen_trIn=train_test_split(trIn, test_size=len(trA)/len(trIn))
frames = [trA, chosen_trIn]
subsample = pd.concat(frames)
X_train = pd.DataFrame(subsample, columns=attributes)
Y_train=pd.DataFrame(subsample, columns=target)
clf=RandomForestClassifier(n_estimators=200)
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
matrix =confusion_matrix(Y_test,pred)
print(matrix)
metrics.roc_auc_score(Y_test,pred)
'''

##==========================================
'''
predict_value=[]
remain_trIn=trIn
RF_score=[]
for i in range(0,9,1):
    trIn=remain_trIn
    remain_trIn,chosen_trIn=train_test_split(trIn, test_size=len(trA)/len(remain_trIn))
##bind the chosen_trIn and trA to create a subsample
    frames = [trA, chosen_trIn]
    subsample = pd.concat(frames)
##get X_train and Y_train
    
    X_train = pd.DataFrame(subsample, columns=attributes)
    Y_train=pd.DataFrame(subsample, columns=target)  
##build the random forest tree model
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,Y_train)
##build random forests based on sub-samples
    predict_value.append(clf.predict(X_test))
    RF_score.append(average_precision_score(Y_test, clf.predict(X_test)))
    
##apply the majority voting scheme for the prediction results generated by subsamples
df = pd.DataFrame(predict_value)
p=df.sum(axis=0)  
p=p.to_frame()
p.index=test_set.index
p.columns = ['votes']

def label (row):
   if row['votes'] >24:
      return 1
   else:
      return 0
   return 'Other'
p['Label']  = p.apply (lambda row: label (row),axis=1)

##get the measurement of this model
average_precision_score(Y_test, p['Label'])
matrix =confusion_matrix(Y_test,p['Label'])

print(matrix)

##======================================================================
print("Accuracy:",metrics.accuracy_score(Y_test, p['Label']))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, p['Label']))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, p['Label']))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, p['Label'])))
''' 
