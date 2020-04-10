#!/usr/bin/env python
# coding: utf-8

# ### Preprocessing

# In[ ]:


import numpy as np
import pandas as pd 
import tensorflow as tf 
import keras from keras.models
import Sequential from keras.layers
import Dense from keras.layers 
import Dropout from keras
import regularizers from keras.optimizers
import SGD from keras.utils 
import to_categorical from numpy
import array 
seed=7;
np.random.seed(seed)
train=pd.read_csv(r"C:\Users\Acer\Desktop\mlworks\training1.csv") 
df2=train.values
np.random.shuffle(df2)
x_train=df2[0:,0:5]
y_train=df2[0:,5:6]
y_train=y_train.astype('int32')
test=pd.read_csv(r"C:\Users\Acer\Desktop\mlworks\testing2.csv") 
df4=df3.values
np.random.shuffle(df4)
x_test=df4[0:,0:5]
y_test=df4[0:,5:6]
y_test=y_test.astype('int32')
val=pd.read_csv(r"C:\Users\Acer\Desktop\mlworks\validate1.csv") 
df6=df5.values
np.random.shuffle(df6)
x_val=df6[0:,0:5]
y_val=df6[0:,5:6]
y_val=y_val.astype('int32')
#trainx=train.values 
#testx=test.values
#valx=val.values
#x_train=trainx[:,:5] 
#print(x_train.shape)
#x_test=testx[:,:5] 
#print(x_test.shape)
#x_val=valx[:,:5]
#print(x_val.shape) 
xtrain=x_train.reshape(108,50,50,5,1) 
xtest=x_test.reshape(27,50,50,5,1)
xval=x_val.reshape(54,50,50,5,1) 
ytrain=[] for i in range(4):
    for j in range (1,28):      
        ytrain.append(j) 
ytest=[] 
for i in range(1):    
    for j in range (1,28):      
        ytest.append(j)        
yval=[] 
for i in range(2):   
    for j in range (1,28):      
        yval.append(j) 
from sklearn import preprocessing 
lb = preprocessing.LabelBinarizer() 
ytrain = lb.fit_transform(ytrain) 
ytest = lb.fit_transform(ytest) 
yval = lb.fit_transform(yval) 



# ### CNN Architecture

# In[ ]:


from keras.layers import Conv3D, MaxPool3D, Flatten, Dense 
from keras.layers import Dropout, Input, BatchNormalization 
from sklearn.metrics import confusion_matrix, accuracy_score 
from keras.losses import categorical_crossentropy 
from keras.optimizers import Adadelta 
from keras.models import Model 
from keras.layers import Conv3D, MaxPooling3D 
from keras.optimizers import SGD 
import tflearn
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
def baseline_model():
    model = Sequential() 
    model.add(Conv3D(16, kernel_size=(2, 2, 1), activation='relu', input_shape=(50,50,5,1))) 
    model.add(Conv3D(32, (2, 2, 1), activation='relu')) 
    #model.add(MaxPooling3D(pool_size=(2, 2, 2))) 
    #model.add(Conv3D(32, (2, 2, 1), activation='relu')) 
    #model.add(Conv3D(64, (2, 2, 1), activation='relu')) 
    model.add(MaxPooling3D(pool_size=(2, 2, 2))) 
    model.add(Conv3D(64, (2, 2, 1), activation='relu')) 
    model.add(Conv3D(64, (2, 2, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization()) 
    model.add(Flatten()) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(0.4)) 
    model.add(Dense(128, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(27, activation='softmax')) 
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0009), metrics=['accuracy'])
    model.summary()
    return model
#model.fit(xtrain, ytrain, batch_size=128, epochs=100, verbose=1, validation_data=(xval, yval)) 
#accuracy = model.evaluate(xtest, ytest, verbose=1) 
#print(accuracy[1]*100) 
#pred = model.predict(xval)
#pred = np.argmax(pred, axis=1) 
#print(pred)
estimator = KerasClassifier(build_fn=baseline_model, epochs=6, batch_size=5, verbose=1)


kfold = KFold(n_splits=20, shuffle=True, random_state=seed)
estimator.fit(x_train, y_train)

results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

predictions = estimator.predict(x_test)
print(predictions)

#predt=model.predict_proba(x_test)
#print(predt)

#for i in range(2600,2606):
   # XpredictInputData = np.array([x_train[i]])
    #ynew = model.predict(XpredictInputData)
    #ynew=ynew.astype('int32')
   # print("X=%s, Predicted=%s" % (x_train[i], ynew))
list_class=['1a','2a','3a','4a','5a','6a','7a','8a','9a','1b','2b','3b','4b','5b','6b','7b','8b','9b','1c','2c','3c','4c','5c','6c','7c','8c','9c']
for i in range(0,20):
    print(" prediction " + predictions[i] , " class "+list_class[predictions[i]-1])

