import sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

from utils import read_split_data, read_subsystem, read_subsystem_data
import pickle
from rand_functions import *


def evaluate_model(model, test_X, test_Y):
    # prediction = model.predict(test_X)
    # acc = sklearn.metrics.accuracy_score(test_Y, prediction)
    acc = model.score(test_X, test_Y)
    return acc


subsystem_dict = read_subsystem()
common_reaction_dict = {'Gimme': {'293T': {}, 'A549': {}, 'CALU': {}, 'Lung': {}, 'NHBE': {}},
                        'iMAT': {'293T': {}, 'A549': {}, 'CALU': {}, 'Lung': {}, 'NHBE': {}},
                        'init': {'293T': {}, 'A549': {}, 'CALU': {}, 'Lung': {}, 'NHBE': {}},
                        'Tinit': {'293T': {}, 'A549': {}, 'CALU': {}, 'Lung': {}, 'NHBE': {}}}
common_reactions_per_model = {'Gimme': [], 'iMAT': [], 'init': [], 'Tinit': []}
common_reactions_per_model_for_box_plot = {'Gimme': [], 'iMAT': [], 'init': [], 'Tinit': []}
common_reactions_per_cell = {'Gimme': {'293T': [], 'A549': [], 'CALU': [], 'Lung': [], 'NHBE': []},
                             'iMAT': {'293T': [], 'A549': [], 'CALU': [], 'Lung': [], 'NHBE': []},
                             'init': {'293T': [], 'A549': [], 'CALU': [], 'Lung': [], 'NHBE': []},
                             'Tinit': {'293T': [], 'A549': [], 'CALU': [], 'Lung': [], 'NHBE': []}}
common_reactions_per_cell_for_box_plot = {'293T': [], 'A549': [], 'CALU': [], 'Lung': [], 'NHBE': []}
results = {}
n_nb_reactions = 0
n_dt_reactions = 0
n_reactions = 0

# ['Gimme', 'iMAT', 'init', 'Tinit']
# ['293T', 'A549', 'CALU', 'Lung', 'NHBE']
for model in ['Gimme', 'iMAT', 'init', 'Tinit']:
    for cell in ['293T', 'A549', 'CALU', 'Lung', 'NHBE']:
        model_name = model + '_' + cell

        print('Training on model ' + model_name)

        train_X, train_Y, test_X, test_Y, reactions = read_split_data(model=model, cell=cell, shuffle=True, ratio=0.5)
        n_reactions += len(reactions)

        model_gauss = GaussianNB()
        model_tree = tree.DecisionTreeClassifier()

        model_gauss.fit(train_X, train_Y)
        model_tree.fit(train_X, train_Y)

        acc_nb = evaluate_model(model_gauss, test_X, test_Y)
        acc_dt = evaluate_model(model_tree, test_X, test_Y)

        results[model_name] = {'NB': acc_nb,
                               'DT': acc_dt}
        print('Accuracy --- NB : ' + str(acc_nb) + '  DT : ' + str(acc_dt))

        for subsystem in subsystem_dict.keys():
            react_for_subsystem = read_subsystem_data(subsystem_dict, reactions, subsystem)

            dt_reactions = []
            nb_reactions = []
            for i, reaction in enumerate(react_for_subsystem):
                model_gauss.fit(train_X[:, i].reshape(-1, 1), train_Y)
                model_tree.fit(train_X[:, i].reshape(-1, 1), train_Y)
                if abs(model_gauss.var_[0]) > 1e-6:
                    acc_nb = evaluate_model(model_gauss, test_X[:, i].reshape(-1, 1), test_Y)
                    acc_dt = evaluate_model(model_tree, test_X[:, i].reshape(-1, 1), test_Y)
                    # print('Accuracy for reaction ' + reaction + ' --- NB : ' + str(acc_nb) + '  DT : ' + str(acc_dt))
                    if acc_nb == 1.0:
                        nb_reactions.append(reaction)
                    if acc_dt == 1.0:
                        dt_reactions.append(reaction)
            if len(react_for_subsystem) > 4:
                common_reaction_dict[model][cell][subsystem] = (react_for_subsystem, nb_reactions)
                common_reactions_per_cell[model][cell].append((len(nb_reactions) / len(react_for_subsystem), subsystem))
                common_reactions_per_model[model].append((len(nb_reactions) / len(react_for_subsystem), subsystem))
                # No name data
                common_reactions_per_cell_for_box_plot[cell].append((len(nb_reactions) / len(react_for_subsystem)))
                common_reactions_per_model_for_box_plot[model].append(len(nb_reactions) / len(react_for_subsystem))

        # tree.plot_tree(model_tree)
        # plt.show()

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
# print(results)

print('Done.')

# Save data to .csv file
save_csv(common_reactions_per_model, "reactions_per_model")
save_csv(common_reactions_per_model, "reactions_per_cell")

# Save the data
save_data(common_reactions_per_model_for_box_plot, "reactions_per_model_data")
save_data(common_reactions_per_cell_for_box_plot, "reactions_per_cell_data")
save_data(common_reactions_per_model, "reactions_per_model")
save_data(common_reactions_per_cell, "reactions_per_cell")


