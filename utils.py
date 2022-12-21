import pandas as pd
import numpy as np

def read_data(path='./data/', model='Gimme', cells='293T') :
    # Construct base path
    base_path = path + model + '_' + cells + '_sample_' 
    # Read healthy and infected cells data
    data_H = pd.read_csv(base_path + 'H.csv', sep = ';')
    data_I = pd.read_csv(base_path + 'I.csv', sep = ';')
    # Extract common columns
    common_reactions = sorted(list(set(data_H.columns) & set(data_I.columns)))
    data_H = data_H[common_reactions]
    data_I = data_I[common_reactions]
    return (data_H, data_I)


def split_data(data, ratio=0.2) :
    # Data oblike (data_H, data_I)
    X = np.concatenate((data[0], data[1]), axis=0)
    Y = np.concatenate((np.zeros(1000),  np.ones(1000)), axis=0)
    start = int(ratio * 1000)
    end = 2000 - int(ratio * 1000)
    train_set_X = X[start:end, :]
    train_set_Y = Y[start:end]
    test_set_X = np.concatenate((X[0:start, :], X[end:, :]))
    test_set_Y = np.concatenate((Y[0:start], Y[end:]))
    return {"X":[train_set_X, test_set_X], "Y":[train_set_Y, test_set_Y]}

def crossevaluation(data, i=0, ratio=0.2) :
    # TODO:: Za na koncu

    raise NotImplementedError


#for model in ['Gimme', ....]
#    for cell in ['293T'] :

