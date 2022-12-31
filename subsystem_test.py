from utils import read_subsystem, read_subsystem_data, read_data
subsystem_dict = read_subsystem()
common_reaction_dict = {'Gimme':{'293T':{},'A549':{},'CALU':{}, 'Lung':{}, 'NHBE':{}}, 
                        'iMAT':{'293T':{},'A549':{},'CALU':{}, 'Lung':{}, 'NHBE':{}},
                        'init':{'293T':{},'A549':{},'CALU':{}, 'Lung':{}, 'NHBE':{}},
                         'Tinit':{'293T':{},'A549':{},'CALU':{}, 'Lung':{}, 'NHBE':{}}}

for model in ['Gimme', 'iMAT', 'init', 'Tinit']:
    for cell in ['293T', 'A549', 'CALU', 'Lung', 'NHBE']:
        data, common_reactions = read_data(model = model, cell = cell)
        for subsystem in subsystem_dict.keys():
            common_reaction_dict[model][cell][subsystem] = read_subsystem_data(subsystem_dict,common_reactions, subsystem)                         
