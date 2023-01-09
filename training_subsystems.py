from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import json

from utils import read_split_data, read_subsystem, read_subsystem_data
import pickle

def evaluate_model(model, test_X, test_Y) :
    #prediction = model.predict(test_X)
    #acc = sklearn.metrics.accuracy_score(test_Y, prediction)
    acc = model.score(test_X, test_Y)
    return acc
# 'iMAT', 'init', 'Tinit'

if __name__ == '__main__' :
    results = {}
    n_nb_reactions = 0
    n_dt_reactions = 0
    n_reactions = 0
    subsystem_dict = read_subsystem()
    for model in ['Tinit']:
        for cell in ['Lung']:
                
            model_name = model + '_' + cell
            print('Training on model ' + model_name)
 
            train_X, train_Y, test_X, test_Y, reactions = read_split_data(model=model, cell=cell, shuffle=True, ratio=0.5)
            for subsystem in subsystem_dict.keys():

                model_name_subs = model + '_' + cell + '_' + subsystem

                reactions = read_subsystem_data(subsystem_dict,reactions, subsystem)
                print(len(reactions))                         
                n_reactions += len(reactions)
                model_gauss = GaussianNB()
                model_tree = tree.DecisionTreeClassifier()

                model_gauss.fit(train_X, train_Y)
                model_tree.fit(train_X, train_Y)

                acc_nb = evaluate_model(model_gauss, test_X, test_Y)
                acc_dt = evaluate_model(model_tree, test_X, test_Y)

                results[model_name_subs] = {'NB': acc_nb,
                                    'DT': acc_dt}

                print('Accuracy --- NB : ' + str(acc_nb) + '  DT : ' + str(acc_dt) + ' Sub: ' +model_name_subs)

                dt_reactions = []
                nb_reactions = []

                for i, reaction in enumerate(reactions):
                    model_gauss.fit(train_X[:, i].reshape(-1, 1), train_Y)
                    model_tree.fit(train_X[:, i].reshape(-1, 1), train_Y)

                    if abs(model_gauss.var_[0]) > 1e-6 :
                        acc_nb = evaluate_model(model_gauss, test_X[:, i].reshape(-1, 1), test_Y)
                        acc_dt = evaluate_model(model_tree, test_X[:, i].reshape(-1, 1), test_Y)
                        #print('Accuracy for reaction ' + reaction + ' --- NB : ' + str(acc_nb) + '  DT : ' + str(acc_dt))
                        if acc_nb == 1.0 :
                            nb_reactions.append(reaction)
                        if acc_dt == 1.0 :
                            dt_reactions.append(reaction)
                if len(reactions) != 0:
                    print(len(nb_reactions)/len(reactions))
                    print(len(dt_reactions)/len(reactions))  
  
        #tree.plot_tree(model_tree)
        #plt.show()

            
            # write subsystem results to json file
        with open("results/subsystem_results.json", "w") as f:
            json.dump(results, f)

            n_nb_reactions += len(nb_reactions)
            n_dt_reactions += len(dt_reactions)

            f = open('./models/' + model_name + '_NB' + '.pickle', 'wb')
            pickle.dump(model_gauss, f)
            f.close()

            f = open('./models/' + model_name + '_DT' + '.pickle', 'wb')
            pickle.dump(model_tree, f)
            f.close()

    n_nb_reactions /= 20
    n_dt_reactions /= 20
    n_reactions /= 20
    print('Average number of common reactions : ' + str(n_reactions))
    print('Average changed reactions --- NB : ' + str(n_nb_reactions) + '  DT : ' + str(n_dt_reactions))
    #print(results)

    print('Done.')