import pandas as pd
def read_data(path='./data/', model='Gimme', cell='293T') :
    base_path = path + model + '_' + cell + '_'
    data_H = pd.read()
    #TODO :: read healthy and infected cells data
    #TODO :: extract common columns


    #return data_H, data_I


def split_data(data, ratio=0.2) :

    #TODO :: make train/test set 0.8/0.2
    # Lahko samo fiksno zadnjih 20% - 200 infected 200 healthy
    print('blah')

def crossevaluation(data, i=0, ratio=0.2) :
    # TODO:: Za na koncu

    print('blah')


#for model in ['Gimme', ....]
#    for cell in ['293T'] :