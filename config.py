# Example of config file

max_sep = 50
thresh_window = 20000           # number of instances (minutes) to include in the budget counting
budget = 0.01         # labeling budget, approx amount of instances labeled/max number of instances labeled? 
total_runs = 5
ml = 'knn'
ML_id = 'KNN'
k_value = 3
dataset_name = 'synthetic'
iml_strategies = ['uncertainty', 'error', 'state change', 'time', 'random']