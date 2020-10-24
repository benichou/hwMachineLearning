import pandas as pd
import sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn import metrics 
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


data = load_wine()

# describe the metadata, feature names and target names
wineDataSetsMetadata = dir(data)
featureNames = data['feature_names']
targetNames = data['target_names']

X = pd.DataFrame(data.data, columns=featureNames)
Y = data.target


xTrain, xTest, yTrain, yTest = train_test_split(X, 
                                                Y, 
                                                shuffle=True, 
                                                test_size=0.2, 
                                                random_state=1234)


#Default Neural Network model without any tuning - base metric
MLPmodelDefault = MLPClassifier()
MLPmodelDefault.fit(xTrain, yTrain)

MLPmodelDefault.get_params()

# train
yPredTrainMLPDefault = MLPmodelDefault.predict(xTrain)
print(f'Train Error - Default Network: {1- metrics.accuracy_score(yTrain, yPredTrainMLPDefault)}' )  
# test 
yPredTestMLPDefault = MLPmodelDefault.predict(xTest)
print(f'Test Error - Default Network: {1- metrics.accuracy_score(yTest, yPredTestMLPDefault)}' )

#Parameter tuning with GridSearchCV 

#######################
### Neural Network A (auto)
#######################
estimatorMLP = MLPClassifier(batch_size='auto', warm_start=True, max_iter=1000)
parametersMLP = {
    'hidden_layer_sizes': (10,120,10),
    'activation': ('identity', 'logistic', 'tanh', 'relu'),
    'alpha': (0.000001, 0.00001, 0.0001),
    'solver': ('lbfgs', 'sgd', 'adam'),
    'verbose': [False]
                   }
# with GridSearch
gridSearchMLPA = GridSearchCV(
    estimator=estimatorMLP,
    param_grid=parametersMLP,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
estimatorMLP.get_params()

MLPA=gridSearchMLPA.fit(xTrain, yTrain)
print(f'We have chosen the following hyperparameters with GridSearchCV {MLPA.best_params_}')

# train
yPredTrainMLPA =MLPA.predict(xTrain)
print(f'Train Error - Neural Network A: {1- metrics.accuracy_score(yTrain, yPredTrainMLPA)}' )  
# test
yPredTestMLPA =MLPA.predict(xTest)
print(f'Test Error - Neural Network A: {1- metrics.accuracy_score(yTest, yPredTestMLPA)}' )

#######################
### Neural Network (Solver=sgd)
#######################

estimatorMLP = MLPClassifier(batch_size='auto', warm_start=True, solver='sgd', max_iter=1000)
parametersMLP = {
    'hidden_layer_sizes': (10,120,10),
    'activation': ('identity', 'logistic', 'tanh', 'relu'),
    'alpha': (0.000001, 0.00001, 0.0001),
    'learning_rate': ('constant', 'invscaling', 'adaptive'),
    'momentum': (0.1,0.9,0.1),
                   }
# with GridSearch
gridSearchMLPB = GridSearchCV(
    estimator=estimatorMLP,
    param_grid=parametersMLP,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)

# fit
MLPB=gridSearchMLPB.fit(xTrain, yTrain)

# best hyperparameters
print(f'We have chosen the following hyperparameters {MLPB.best_params_}')

# train
yPredTrainMLPB =MLPB.predict(xTrain)
print(f'Train Error - Neural Network B: {1- metrics.accuracy_score(yTrain, yPredTrainMLPB)}' )  
# test
yPredTestMLPB =MLPB.predict(xTest)
print(f'Test Error - Neural Network B: {1- metrics.accuracy_score(yTest, yPredTestMLPB)}' )  


#######################
### Neural Network (Solver=adam)
#######################
estimatorMLP = MLPClassifier(batch_size='auto', warm_start=True, solver='adam', max_iter=1000)
parametersMLP = {
    'hidden_layer_sizes': (10,120,10),
    'activation': ('identity', 'logistic', 'tanh', 'relu'),
    'alpha': (0.000001, 0.00001, 0.0001),
    'beta_1': (0.1,0.9,0.1),
    'beta_2': (0.1,0.9,0.1),
                   }
# with GridSearch
gridSearchMLPC = GridSearchCV(
    estimator=estimatorMLP,
    param_grid=parametersMLP,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)

MLPC=gridSearchMLPC.fit(xTrain, yTrain)

# best hyperparameters
print(f'We have chosen the following hyperparameters {MLPC.best_params_}')

# train
yPredTrainMLPC =MLPC.predict(xTrain)
print(f'Train Error - Neural Network B: {1- metrics.accuracy_score(yTrain, yPredTrainMLPC)}' )  
# test
yPredTestMLPC =MLPC.predict(xTest)
print(f'Test Error - Neural Network B: {1- metrics.accuracy_score(yTest, yPredTestMLPC)}' )  






# Summary

# baseline 
# train

print(f'Train Error - Default Network: {1- metrics.accuracy_score(yTrain, yPredTrainMLPDefault)}' )  
print(f'Test Error - Default Network: {1- metrics.accuracy_score(yTest, yPredTestMLPDefault)}' )

# neural network A

print(f'Train Error - Neural Network A: {1- metrics.accuracy_score(yTrain, yPredTrainMLPA)}' )  
print(f'Test Error - Neural Network A: {1- metrics.accuracy_score(yTest, yPredTestMLPA)}' )

# neural network B

print(f'Train Error - Neural Network B: {1- metrics.accuracy_score(yTrain, yPredTrainMLPB)}' )  
print(f'Test Error - Neural Network B: {1- metrics.accuracy_score(yTest, yPredTestMLPB)}' ) 

# neural network C

print(f'Train Error - Neural Network    C: {1- metrics.accuracy_score(yTrain, yPredTrainMLPC)}' )  
print(f'Test Error - Neural Network C: {1- metrics.accuracy_score(yTest, yPredTestMLPC)}' )  









