import matplotlib.pyplot as plt
import json
import numpy as np


def save_csv(data):
    pass


def save_data(data, name_of_file):
    with open(f'{name_of_file}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(data))


def draw_boxplot(data):
    data_array = []
    for keys in data:
        data_array.append(data[keys])

    ticks = data.keys()

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    boxplot = plt.boxplot(data_array, positions=np.array(range(len(data.keys()))) * 2.0, sym='', widths=0.6)
    set_box_color(boxplot, '#cc00ff')

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


with open('reactions_per_model_data.txt', 'r') as file:
    data1 = json.load(file)
    draw_boxplot(data1)

with open('reactions_per_cell_data.txt', 'r') as file:
    data2 = json.load(file)
    draw_boxplot(data2)
