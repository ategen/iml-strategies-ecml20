import numpy as np
from sklearn.metrics import confusion_matrix
import IML_data_stream as learn
import dataset_loading
from config import (
    max_sep,
    thresh_window,
    budget,
    total_runs,
    ml,
    ML_id,
    k_value,
    dataset_name,
    iml_strategies,
)


def main():
    datasets, classes = dataset_loading.load_dataset(dataset_name, total_runs)

    conf_m_av = initialize_conf_m(classes, iml_strategies)
    queries_av = np.zeros(len(iml_strategies))
    accuracies_av = []
    for _ in iml_strategies:
        accuracies_av.append([])
    for dataset in datasets:
        for j, strategy in enumerate(iml_strategies):
            acc, pred, labels, queries = learn.learning_sklearn(
                dataset, classes, max_sep, thresh_window, budget, strategy, ml, k_value
            )

            accuracies_av[j], queries_av[j] = average_add(
                accuracies_av[j], acc, queries_av[j], queries
            )
            conf_m_av[j] = construct_conf_m(labels, pred, classes, conf_m_av, j)

    for j, strategy in enumerate(iml_strategies):
        accuracies_av[j], queries_av[j] = average_sum(
            accuracies_av[j], total_runs, queries_av[j]
        )

    conf_m_av = finalize_conf_m(classes, conf_m_av, iml_strategies, total_runs)

    save_data(
        budget,
        total_runs,
        thresh_window,
        ML_id,
        iml_strategies,
        accuracies_av,
        queries_av,
        conf_m_av,
    )


def initialize_conf_m(classes, iml_strategies):
    conf_m_av = []
    for _ in iml_strategies:
        conf_m_av.append([])
    return conf_m_av


def construct_conf_m(labels, pred, classes, conf_m_av, j):
    conf_m = construct_conf_m_multi(labels, pred, classes)
    return conf_m_add(conf_m_av[j], conf_m)


def construct_conf_m_multi(labels, pred, classes):
    conf_m = confusion_matrix(labels, pred)
    add_labels = []
    for c in classes:
        if c not in labels:
            add_labels.append(c)
    if add_labels != []:
        conf_m_tmp = []
        tmp_counter_row = 0
        for j_row in classes:
            if j_row in add_labels:
                conf_row_tmp = list(np.zeros(len(classes)))
            else:
                conf_row_tmp = []
                tmp_counter_col = 0
                for j_col in classes:
                    if j_col in add_labels:
                        conf_row_tmp.append(0)
                    else:
                        conf_row_tmp.append(conf_m[tmp_counter_row][tmp_counter_col])
                        tmp_counter_col = tmp_counter_col + 1
                tmp_counter_row = tmp_counter_row + 1
            conf_m_tmp.append(conf_row_tmp)
        conf_m = conf_m_tmp
    return conf_m


def finalize_conf_m(classes, conf_m_av, iml_strategies, total_runs):
    for j, _ in enumerate(iml_strategies):
        conf_m_av[j] = conf_m_sum(conf_m_av[j], total_runs)
    return conf_m_av


def conf_m_add(conf_m_av, conf_m):
    if conf_m_av == []:
        return conf_m

    for i, c_row in enumerate(conf_m):
        for j, c in enumerate(c_row):
            conf_m_av[i][j] = conf_m_av[i][j] + c
    return conf_m_av


def conf_m_sum(conf_m_av, total_runs):
    for i, c_row in enumerate(conf_m_av):
        for j, _ in enumerate(c_row):
            conf_m_av[i][j] = conf_m_av[i][j] / total_runs
    return conf_m_av


def average_add(accuracies_av, accuracies, query_counter_av, query_counter):
    if accuracies_av == []:
        return accuracies, query_counter

    if len(accuracies) > len(accuracies_av):
        accuracies = accuracies[0 : len(accuracies_av)]
    elif len(accuracies) < len(accuracies_av):
        accuracies_av = accuracies_av[0 : len(accuracies)]
    accuracies_av = [sum(x) for x in zip(accuracies_av, accuracies)]
    query_counter_av = query_counter_av + query_counter
    return accuracies_av, query_counter_av


def average_sum(accuracies, total_runs, query_counter):
    accuracies[:] = [i / total_runs for i in accuracies]
    query_counter = query_counter / total_runs
    return accuracies, query_counter


def save_data(
    budget,
    total_runs,
    thresh_window,
    ML_id,
    iml_strategies,
    accuracies_av,
    queries_av,
    conf_m_av,
):
    save_name_id = "{0}budget_{2}w_{1}".format(budget * 100, total_runs, thresh_window)
    save_file("iml", "strategies", "list", iml_strategies)
    save_file("accum_acc", ML_id, save_name_id, accuracies_av)
    save_file("queries", ML_id, save_name_id, queries_av)
    save_file("conf_m", ML_id, save_name_id, conf_m_av)
    return


def save_file(script_name, value_name, runs, value):
    file_name = "{0}_{1}_{2}runs.txt".format(script_name, value_name, str(runs))
    with open(file_name, "w") as f:
        for i in value:
            f.write("%s," % i)
    f.close()


main()