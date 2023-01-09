import matplotlib.pyplot as plt
import json
import numpy as np


def save_csv(data, file_name):
    print('test')
    with open(f'{file_name}.csv', 'w') as f:
        for key in data:
            f.write(f'{key}\n')
            for values in data[key]:
                f.write(';'.join((str(values[0]), values[1])) + '\n')
    return


# Save data to json file
def save_data(data, name_of_file):
    with open(f'{name_of_file}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(data))


def draw_boxplot(data, file_name):
    data_array = []
    for keys in data:
        data_array.append(data[keys])

    ticks = data.keys()

    def set_box_color(bp, file_name):
        # colors for model or cell
        colors = []
        if file_name == 'reactions_per_model_data.txt':
            # colors: royalblue, limegreen, peachpuff1, burnt orange
            colors = ['#234E70', '#CCF381', '#EEA47FFF', '#EE4E34']
        elif file_name == 'reactions_per_cell_data.txt':
            # colors: royalblue, limegreen, peachpuff1, burnt orange, scarlet
            colors = ['#234E70', '#CCF381', '#EEA47FFF', '#EE4E34', '#B85042']
        elif file_name == 'all_comb.txt':
            # only one color, 20 combinations
            colors = ['#EEA47FFF']
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        plt.setp(bp['medians'], color='black')
         # plt.setp(bp['boxes'], color=color)
         # plt.setp(bp['whiskers'], color=color)
         # plt.setp(bp['caps'], color=color)
        return

    plt.figure(figsize=(8,5))
 
    boxplot = plt.boxplot(data_array, positions=np.array(range(len(data.keys()))) * 2.0, patch_artist=True, sym='', widths=1.5)
    set_box_color(boxplot, file_name)

    # legend
    # plt.plot([], c='#cc00ff', label=f'Percentage range of significantly changed reactions per {file.name.split("_")[2]}')
    # plt.plot([], c='#2C7BB6', label='Oranges')
    # plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(0, 1)
    plt.xlabel(f"{file.name.split('_')[2]}")
    plt.ylabel("Percentage of significantly changed reactions")
    plt.tight_layout()
    # plt.title(f"{file.name.split('_')[2]}")
    plt.savefig(f'{file.name.split(".")[0]}.png')
    # show plot
    plt.show()

if __name__ == '__main__':
    with open('reactions_per_model_data.txt', 'r') as file:
        data1 = json.load(file)
        draw_boxplot(data1, file.name)

    with open('reactions_per_cell_data.txt', 'r') as file:
        data2 = json.load(file)
        draw_boxplot(data2, file.name)
