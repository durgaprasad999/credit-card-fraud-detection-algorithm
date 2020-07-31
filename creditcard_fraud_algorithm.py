#import scipy.stats as stats 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv('creditcard.csv')

#overview of data
data.info()

#print the columns
print(data.columns)

#print the shape
print(data.shape)

#describe the data
print(data.describe())

#divide the data to small piece
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)

#plot the histogram of each parameter 
data.hist(figsize=(20,20))
plt.show()

#determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

print('fraud cases: {}'.format(fraud))
print('valid cases: {}'.format(valid))

#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#get all the columns from the dataframe
columns = data.columns.tolist()

#filter the columns to remove data we dont wanted
columns = [c for c in columns if c not in ["Class"]]

#store the variable we will be predicting on
target = "Class"

X = data[columns]
Y = data[target]

#print thr shape of X and Y
print(X.shape)
print(Y.shape)

#define a random state
state = 1

#define a outlier detection method
classifiers = {
    'Isolation Forest': IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = state),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}

#fill the model
n_outliers = len(fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    #reshape the prediction values 0 for valid, 1 for fraud 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #run classification matrix
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))