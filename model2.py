from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv 
import pickle

import numpy as np
import pandas as pd

#get the file with the variables
data=pd.read_csv("Myfile.csv", sep=",")
data=pd.DataFrame(data,columns=data.columns[2:7])


x=data.drop(columns=['KSS'])
y=data['KSS']

#create the model 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)
predictions=clf.predict(x_test)
score=accuracy_score(y_test, predictions)
print(score)
confusion_matrix=confusion_matrix(y_test, predictions,labels=[1,2,3])
print(confusion_matrix)

#save the model
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
