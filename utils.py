import pandas as pd
import numpy as np

def read_data(path='./data/', model='Gimme', cell='293T', shuffle=False, shuffle_seed=42) :
    # Construct base path
    base_path = path + model + '_' + cell + '_sample_'
    # Read healthy and infected cells data
    data_H = pd.read_csv(base_path + 'H.csv', sep = ';')
    data_I = pd.read_csv(base_path + 'I.csv', sep = ';')

    if shuffle :
        data_H = data_H.sample(frac=1, random_state=shuffle_seed)
        data_I = data_I.sample(frac=1, random_state=shuffle_seed)

    # Extract common columns
    common_reactions = sorted(list(set(data_H.columns) & set(data_I.columns)))
    data_H = data_H[common_reactions]
    data_I = data_I[common_reactions]
    return (data_H, data_I), common_reactions

def split_data(data, ratio=0.2) :
    # Data oblike (data_H, data_I)
    X = np.concatenate((data[0], data[1]), axis=0)
    Y = np.concatenate((np.zeros(1000),  np.ones(1000)), axis=0)

    start = int(ratio * 1000)
    end = 2000 - int(ratio * 1000)
    train_X = X[start:end, :]
    train_Y = Y[start:end]
    test_X = np.concatenate((X[0:start, :], X[end:, :]))
    test_Y = np.concatenate((Y[0:start], Y[end:]))

    return train_X, train_Y, test_X, test_Y

def read_split_data(path='./data/', model='Gimme', cell='293T', shuffle=False, shuffle_seed=42, ratio=0.2) :
    data, reactions = read_data(path, model, cell, shuffle, shuffle_seed)
    train_X, train_Y, test_X, test_Y = split_data(data, ratio)

    return train_X, train_Y, test_X, test_Y, reactions

def read_subsystem() :
    subsystem_dict = {}
    subs_data = pd.read_csv("data/Human-GEM_subsystems.csv", sep = ';')
    for i in range(len(subs_data["subsystem"])):
        subsystem = subs_data["subsystem"][i]
        reaction = subs_data["rxn"][i]

        if subsystem in subsystem_dict.keys():
            subsystem_dict[subsystem].append(reaction)
        else:
            subsystem_dict[subsystem]  = [reaction]
    #returns : {'subsys1' : [], 'subsys2': [] .... }
    return subsystem_dict

def read_subsystem_data(subsystem_dict, common_reactions, subsystem='neki') :
    #Done :: preberi iz txt vse podsisteme in pripadajoče reakcije
    #Done :: relevant_reaction = common_reaction & subsys_reactions
    #Done :: extract relevant reactions
    subsystem_reactions = subsystem_dict[subsystem]
    relevant_reactions = set(subsystem_reactions).intersection(set(common_reactions))
    return relevant_reactions


def crossevalidation(data, i=0, ratio=0.2) :
    # TODO:: Za na koncu

    raise NotImplementedError


#for model in ['Gimme', ....]
#    for cell in ['293T'] :

