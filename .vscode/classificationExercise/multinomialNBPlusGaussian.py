from sklearn.datasets import load_wine
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier # for feature importance

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the data
data = load_wine()

# describe the metadata, feature names and target names
wineDataSetsMetadata = dir(data)
featureNames = data['feature_names']
targetNames = data['target_names']

df = pd.DataFrame(X, columns=featureNames)
df.columns
df.describe()

# get the features and 
X = data.data
Y = data.target




xTrain, xTest, yTrain, yTest = train_test_split(X, 
                                                Y, 
                                                shuffle=True, 
                                                test_size=0.2, 
                                                random_state=1234)

# fit the multinomialNaive Bayes

clf = MultinomialNB()
multiNbClf = clf.fit(xTrain, yTrain)

## train
# predict on the train and get the train error for multinomialNB

predictionsOnTrain = multiNbClf.predict(xTrain)

# get the train accuracy and train error for multinomialNaive Bayes

print(f'Accuracy {metrics.accuracy_score(yTrain, predictionsOnTrain)}')
print(f'Train error {1- metrics.accuracy_score(yTrain, predictionsOnTrain)}') # train error at 9%
# let us get the confusion matrix for train/predictions on train without visualization

print(metrics.confusion_matrix(yTrain, predictionsOnTrain))

## test

# predict on the test and get the test error for multinomialNB

predictionsOnTest = multiNbClf.predict(xTest)

# get the train accuracy and train error for multinomialNaive Bayes

print(f'Accuracy {metrics.accuracy_score(yTest, predictionsOnTest)}')
print(f'Test Error {1- metrics.accuracy_score(yTest, predictionsOnTest)}') # test error at 16%
# let us get the confusion matrix for train/predictions on train without visualization

print(metrics.confusion_matrix(yTest, predictionsOnTest))
# The test error is at 



# fit the gaussian Naive Bayes
clf = GaussianNB()
gaussNbClf = clf.fit(xTrain, yTrain)


## train
# predict on the train and get the train error for multinomialNB

predictionsOnTrain = gaussNbClf.predict(xTrain)

# get the train accuracy and train error for multinomialNaive Bayes

print(f'Accuracy {metrics.accuracy_score(yTrain, predictionsOnTrain)}')
print(f'Train error {1- metrics.accuracy_score(yTrain, predictionsOnTrain)}') # train error at 0%
# let us get the confusion matrix for train/predictions on train without visualization

print(metrics.confusion_matrix(yTrain, predictionsOnTrain))

## test

# predict on the test and get the test error for multinomialNB

predictionsOnTest = gaussNbClf.predict(xTest)

# get the train accuracy and train error for multinomialNaive Bayes

print(f'Accuracy {metrics.accuracy_score(yTest, predictionsOnTest)}')
print(f'Test Error {1- metrics.accuracy_score(yTest, predictionsOnTest)}') # test error at 11%
# let us get the confusion matrix for train/predictions on train without visualization

print(metrics.confusion_matrix(yTest, predictionsOnTest))



## Feature Importance

#before feature importance, let us see the Pearson's correlation between features. If we find a combination of features 
# that have a very high Pearsons correlation, we will drop one of the features in that combination

plt.figure(figsize=(18, 15))
sns.heatmap(df.corr(), annot=True, fmt="0.1f")
plt.show()

# let us drop the Total_phenols variable before deriving the right feature importance

df.drop('total_phenols', axis=1, inplace=True)

xTrain, xTest, yTrain, yTest = train_test_split(df, 
                                                Y,
                                                stratify=Y)

# deriving the feature importance on the data

model_feature_importance = RandomForestClassifier(n_estimators=1000).fit(xTrain,yTrain).feature_importances_
feature_scores = pd.DataFrame({'score':model_feature_importance}, index=df.columns).sort_values('score')
print(feature_scores)
sns.barplot(feature_scores['score'], feature_scores.index)
plt.show()






