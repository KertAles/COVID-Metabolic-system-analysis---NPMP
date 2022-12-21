import requests, pandas, csv
import sklearn.metrics
import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
#url = 'https://raw.githubusercontent.com/CompBioLj/COVID_GEMs_and_MEMs/main/flux_samples/Gimme_Lung_sample_I.csv'
#res = requests.get(url, allow_redirects=True)
#with open('Gimme_Lung_sample_I.csv','wb') as file:
#    file.write(res.content)
#url = 'https://raw.githubusercontent.com/CompBioLj/COVID_GEMs_and_MEMs/main/flux_samples/Gimme_Lung_sample_H.csv'
#res = requests.get(url, allow_redirects=True)
#with open('Gimme_Lung_sample_H.csv','wb') as file:
#    file.write(res.content)

from sklearn.model_selection import train_test_split


infected_samples = pandas.read_csv('Gimme_Lung_sample_I.csv', sep = ';')
healthy_samples = pandas.read_csv('Gimme_Lung_sample_H.csv', sep = ';')
reactions = sorted(list(set(healthy_samples.columns) & set(infected_samples.columns)))
infected_samples = infected_samples[reactions]
healthy_samples = healthy_samples[reactions]


Y = np.concatenate((np.zeros(1000),  np.ones(1000)), axis=0)
X = np.concatenate((infected_samples, healthy_samples), axis=0)

#train, test = train_test_split(X, test_size=0.2)

train_X = X[200:1800, :]
train_Y = Y[200:1800]

test_X = np.concatenate((X[0:200, :], X[1800:, :]))
test_Y = np.concatenate((Y[0:200], Y[1800:]))


model = GaussianNB()
model = tree.DecisionTreeClassifier()
model.fit(train_X, train_Y)

test_pred = model.predict(test_X)

acc = sklearn.metrics.accuracy_score(test_Y, test_pred)
f1 = sklearn.metrics.f1_score(test_Y, test_pred)

print('blah')
