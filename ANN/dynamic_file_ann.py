''' Required Libraries
Theano,
Tesorflow,
Keras
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
try:
    filename=input("Enter the filename: ")
    dataset = pd.read_csv(filename)
    
    
except FileNotFoundError:
    exit(1)
    
#start=int(input('Enter the column no. of First independent variable: '))
#end=int(input('Enter the column no. of Last independent variable: '))
target=int(input('Enter the column no. of Target(Output): '))
print('Enter the indexes of irrelevant variables: ')
nonIV=list(map(int,input().split(' ')))
#X = dataset.iloc[:, start:end+1].values
y = dataset.iloc[:, target].values

X=dataset.drop(dataset.columns[nonIV],axis=1)
X=X.drop(dataset.columns[[target]],axis=1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
print('Enter the column names of non-numerical IVs: ')
nonNums=list(map(str,input().split(' ')))

for nonNum in nonNums:
    labelencoder_X_1 = LabelEncoder()
    X[nonNum] = labelencoder_X_1.fit_transform(X[nonNum])
print('Enter the indexes of categorical features: ')
categoricalFeatures=list(map(str,input().split(' ')))
for catFeat in categoricalFeatures:
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
testSize=int(input('Enter the test size from 0.0 to 0.5(Suggested 0.2 to 0.3): '))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier=KerasClassifier(build_fn=build_classifier,batch_size = 10, epochs = 100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
mean=accuracies.mean()
variance=accuracies.std()

