import numpy as np
import random

def load_dataset(dataset_name, total_runs):
    # The synthetic dataset is included as on example of how to use the code
    # If you want to use other datasets, you can include them here
    # dataset should be a list, where each item i a list with the format [sensorvalue1 sensorvalue2 ... classlabel]
    if dataset_name == "synthetic":
        classes = list(range(5))
        complete_shuffle = False
        datasets = load_synthetic_data(total_runs, complete_shuffle, classes)
    else:
        print("Please choose valid dataset.")
    return datasets, classes


def load_synthetic_data(total_runs, complete_shuffle, classes):
    #parameters below can be adjusted depending of application to be simulated
    mean_time_parameter = 20    #mean value for the amount of data instances in a row for the same class
    stdev_time_parameter = 5    #the standard deviation value for the mean_time_parameter

    dataset_orig = load_data_to_nested_list("", "synthetic_dataset.txt")
    n_dataset = len(dataset_orig)
    range_dataset = list(range(n_dataset))
    datasets = []
    if complete_shuffle:
        for _ in range(total_runs):
            shuffled_range = range_dataset
            random.shuffle(shuffled_range)
            shuffled_dataset = [dataset_orig[i] for i in shuffled_range]
            datasets.append(shuffled_dataset)
        return datasets

    mean_time = (n_dataset / len(classes)) / mean_time_parameter
    stdev_time = mean_time / stdev_time_parameter
    dataset_sorted = sort_dataset(classes, dataset_orig)

    for _ in range(total_runs):
        new_dataset = get_arranged_dataset(classes, dataset_orig, dataset_sorted, mean_time, stdev_time)
        datasets.append(new_dataset)
    return datasets

def sort_dataset(classes, dataset_orig):
    dataset_sorted = []
    for _ in classes:
        dataset_sorted.append([])
    for d in dataset_orig:
        dataset_sorted[int(d[-1])].append(d)
    return dataset_sorted

def get_arranged_dataset(classes, dataset_orig, dataset_sorted, mean_time, stdev_time):
    new_dataset = []
    counters = [0] * len(classes)
    while len(new_dataset) < len(dataset_orig):
        rand_class = np.random.randint(len(classes))
        while counters[rand_class] >= len(dataset_sorted[rand_class]):
            rand_class = np.random.randint(len(classes))
        rand_time = np.random.normal(mean_time, stdev_time)
        for i in list(range(int(rand_time))):
            if i + counters[rand_class] >= len(dataset_sorted[rand_class]):
                break
            else:
                new_dataset.append(
                    dataset_sorted[rand_class][i + counters[rand_class]]
                )
        counters[rand_class] = counters[rand_class] + int(rand_time)
    return new_dataset

def load_data_to_nested_list(file_dir, filename):
    file = open(file_dir + filename, "r")
    val = file.read()
    nested_list = []
    val = val[1:-2]
    for x in val.split("],["):
        nested_list.append([float(x_i) for x_i in x.split(",") if x_i])
    return nested_list