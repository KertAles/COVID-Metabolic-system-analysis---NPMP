# Python program to read
# json file

import matplotlib.pyplot as plt
import json

# Opening JSON file
f = open('subsystem_results.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
statistics_data = []
for i in data.values():
	statistics_data.append(i)

dt_data = []
nb_data = []
for elt in statistics_data:
    #plt.boxplot([elt['DT']])
    dt_data.append([elt['DT']])
    nb_data.append(elt['NB'])
#bp = ax.bxp(positions=(len(statistics_data)))
#x = [i for i in range(len(dt_data))]
#print(len(dt_data))
#print(len(x))
plt.boxplot(dt_data)
plt.scatter([i for i in range(len(dt_data))],dt_data,s=0.8)
#plt.boxplot([1])
#plt.boxplot(nb_data,whis=[2,98])

plt.show()
# Closing file
f.close()
