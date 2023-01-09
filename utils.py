import pandas as pd
import matplotlib.pyplot as plt
import json
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
    common_reactions = sorted(list(set(data_H) & set(data_I)))
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

def read_split_data_subs(path='./data/', model='Gimme', cell='293T', subsystem=[], shuffle=False, shuffle_seed=42, ratio=0.2) :
    data, reactions = read_data(path, model, cell, subsystem, shuffle, shuffle_seed)
    train_X, train_Y, test_X, test_Y = split_data(data, ratio)

    return train_X, train_Y, test_X, test_Y, reactions

def read_split_data(path='./data/', model='Gimme', cell='293T', shuffle=False, shuffle_seed=42, ratio=0.2) :
    data, reactions = read_data(path, model, cell, shuffle, shuffle_seed)
    #print(data, reactions)
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


def get_number_of_subsystem_reactions(model, cell):
    subsystem_dict = read_subsystem()
    subsys_reaction_count = {}
    data, common_reactions = read_data(model=model, cell=cell)
    for subsystem in subsystem_dict.keys():
        subsys_reaction_count[subsystem] = len(read_subsystem_data(subsystem_dict, common_reactions, subsystem))

    return subsys_reaction_count


def save_csv(data, file_name):
    print('test')
    with open(f'results/{file_name}.csv', 'w') as f:
        for key in data:
            f.write(f'{key}\n')
            for values in data[key]:
                f.write(';'.join((str(values[0]), values[1])) + '\n')
    return


# Save data to json file
def save_data(data, name_of_file):
    with open(f'results/{name_of_file}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(data))


def draw_boxplot(data, file_name, figsize=(6, 4)):
    data_array = []
    for keys in data:
        data_array.append(data[keys])

    ticks = data.keys()

    def set_box_color(bp, file_name):
        # colors for model or cell
        colors = []
        if file_name == 'reactions_per_model_data.txt':
            # colors: royalblue, limegreen, peachpuff1, burnt orange
            colors = ['#CCF381', '#CCF381', '#CCF381', '#CCF381']
        elif file_name == 'reactions_per_cell_data.txt':
            # colors: royalblue, limegreen, peachpuff1, burnt orange, scarlet
            colors = ['#EE4E34', '#EE4E34', '#EE4E34', '#EE4E34', '#EE4E34']
        elif file_name == 'all_comb.txt':
            # only one color, 20 combinations
            colors = ['#234E70']

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        plt.setp(bp['medians'], color='black')
        # plt.setp(bp['boxes'], color=color)
        # plt.setp(bp['whiskers'], color=color)
        # plt.setp(bp['caps'], color=color)
        return

    plt.figure(figsize=figsize, dpi=200)

    boxplot = plt.boxplot(data_array, positions=np.array(range(len(data.keys()))) * 2.0, patch_artist=True, sym='',
                          widths=1.5)
    set_box_color(boxplot, file_name)

    # legend
    # plt.plot([], c='#cc00ff', label=f'Percentage range of significantly changed reactions per {file.name.split("_")[2]}')
    # plt.plot([], c='#2C7BB6', label='Oranges')
    # plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(0, 1)
    plt.xlabel(f"{file_name.split('_')[2].split('.')[0]}")
    plt.ylabel("Percentage of significantly changed reactions")
    plt.tight_layout()
    # plt.title(f"{file.name.split('_')[2]}")
    plt.savefig(f'{file_name.split(".")[0]}.png')
    # show plot
    plt.show()


def writeout_results():
    with open('results/reactions_per_model_data.txt', 'r') as file:
        data1 = json.load(file)
        draw_boxplot(data1, file.name)

    with open('results/reactions_per_cell_data.txt', 'r') as file:
        data2 = json.load(file)
        draw_boxplot(data2, file.name)

    with open('results/reactions_per_cell.txt', 'r') as file:
        data3 = json.load(file)
        sorted_cells = sorted(data3['iMAT']['Lung'], reverse=True)
        table_data = ''

        # subsys = read_subsystem()
        # subsys_size = {}
        # for sub in subsys :
        #    subsys_size[sub] = len(subsys[sub])
        subsys_size = get_number_of_subsystem_reactions('iMAT', 'Lung')

        for i, cell in enumerate(sorted_cells):
            table_data += cell[1] + ' & ' + str(round(cell[0] * 100, 2)) + '\%' + ' & ' + str(subsys_size[cell[1]])
            table_data += ' \\\\ \n \\hline \n'
            if i == 15:
                break

        print(table_data)

        print(sum([cell[0] for cell in sorted_cells]) / len(sorted_cells))

        sorted_cells = sorted(sorted_cells, key=lambda x: subsys_size[x[1]], reverse=True)

        table_data = ''

        for i, cell in enumerate(sorted_cells):
            table_data += cell[1] + ' & ' + str(round(cell[0] * 100, 2)) + '\%' + ' & ' + str(subsys_size[cell[1]])
            table_data += ' \\\\ \n \\hline \n'
            if i == 9:
                break

        print(table_data)


if __name__ == '__main__' :
    writeout_results()