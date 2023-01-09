import requests, pandas, csv
import sklearn.metrics
import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
from utils import split_data, read_data

if __name__ == '__main__' :

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


    # ------------ Primer split data, read data usage
    model_cell_data = {}
    for model in ['Gimme', 'iMAT', 'init', 'Tinit']:
        for cell in ['293T', 'A549', 'CALU', 'Lung', 'NHBE']:
            model_cell_data[model + '_' + cell] = split_data(read_data(model=model, cells=cell))

    # Primer
    model_cell_data['Gimme_293T']['X'][0] # Gimme_293T train set X
    model_cell_data['Gimme_293T']['X'][1] # Gimme_293T train set X
    model_cell_data['Gimme_293T']['Y'][0] # Gimme_293T train set Y
    model_cell_data['Gimme_293T']['Y'][1] # Gimme_293T train set Y

