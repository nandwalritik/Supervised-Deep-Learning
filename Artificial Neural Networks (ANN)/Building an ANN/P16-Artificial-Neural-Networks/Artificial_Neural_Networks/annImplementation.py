# Data Prepocessing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[: , 13].values
 
# Encoding cotegorical data
 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
 
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X  = onehotencoder.fit_transform(X).toarray()

X  = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train  = sc.fit_transform(X_train)
X_test   = sc.transform(X_test)

# Making the ANN


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
classifier = Sequential()

classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation = 'relu',input_dim = 11))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.add(Dropout(p = 0.1))
# Output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform',activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train,Y_train,batch_size = 10 , epochs = 100)

# predicting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5 )

# Making New Single Prediction Not From the Dataset
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,y_pred)

# Improving ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation = 'relu'))
    classifier.add(Dense(units = 1 , kernel_initializer = 'uniform',activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10 , epochs=100)
#accuracies = cross_val_score(estimator = classifier , X = X_train , y = Y_train , cv = 10 , n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()

# Dropout Reglarization(completed)

# Tuning the ANN
from sklearn.model_selection import GridSearchCV
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam' , 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid= parameters,
                           scoring   = 'accuracy',
                           cv        = 10
                           )

grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy   = grid_search.best_score_

