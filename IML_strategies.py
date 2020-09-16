import random
import numpy as np
from collections import defaultdict


def query_main(
    query_option,
    thresh_budget,
    thresh_counter,
    ML,
    k_value,
    ml_model,
    predict,
    data_instance,
    data_stream,
    result,
    prev_label,
    last_query_counter,
    query_last,
):
    query_user = True  # no difference for most strategies to regular "query", only for 'state change'
    if query_option == "default":
        query = True
    elif query_option == "uncertainty":
        query = query_label_budget(thresh_budget, thresh_counter)
        if predict and query:
            if ML == "knn":
                # query based on uncertinty, here two thirds majority voting
                query = query_knn_TMV(
                    k_value, data_instance[:-1], data_stream, ml_model
                )
            else:
                print("Invalid  ML argument: {0}".format(ML))
    elif query_option == "error":
        # provide label when the last prediction was wrong (and there is enough budget)
        accuracy_d = getAccuracy([data_instance[-1]], [result])
        query = query_label_accuracy(accuracy_d, thresh_budget, thresh_counter)
    elif query_option == "state change":
        # provide label when there is a change in the environment (and there is enough budget)
        query, query_user = label_state_change(
            prev_label, data_instance[-1], thresh_budget, thresh_counter, query_last
        )
    elif query_option == "time":
        # provide label at certain frequency (based on the budget), regardless of everything else
        query, last_query_counter = label_time(last_query_counter, thresh_budget)
    elif query_option == "random":
        # provide label radomly, given a certain budget
        query = label_random(thresh_budget, thresh_counter)
    else:
        print("Invalid query option argument: {0}".format(query_option))

    return query, last_query_counter, query_user


def query_label_budget(thresh_budget, thresh_counter):
    b = np.count_nonzero(thresh_counter) / len(thresh_counter)
    return b < thresh_budget


def query_knn_TMV(k_value, d_point, data_stream, ml_model):
    threshold_knn = (2 / 3)
    k_neigh, k_neigh_index = ml_model.kneighbors(np.array(d_point).reshape(1, -1))
    k_neigh_index = k_neigh_index[0]
    counter = defaultdict(int)
    for k in k_neigh_index:
        k_neigh = data_stream[k][-1]
        counter[k_neigh] = counter[k_neigh] + 1
    max_label = max(counter.items(), key=lambda x: x[1])
    
    return (counter[max_label] / k_value) > threshold_knn


def label_state_change(prev_label, label, thresh_budget, thresh_counter, query_last):
    if prev_label != label:
        query = True
        query_user = True
    else:
        query_user = False
        query = query_last
    if query:
        query = query_label_budget(thresh_budget, thresh_counter)
        if not query:
            query_user = False
    return query, query_user


def label_time(last_query_counter, thresh_budget):
    if last_query_counter >= (1 / thresh_budget):
        last_query_counter = last_query_counter - (1 / thresh_budget)
        query = True
    else:
        query = False
    last_query_counter = last_query_counter + 1
    return query, last_query_counter


def label_random(thresh_budget, thresh_counter):
    b = random.random()
    return thresh_budget >= b


def query_label_accuracy(result, thresh_budget, thresh_counter):
    if result:
        return False
    return query_label_budget(thresh_budget, thresh_counter)


def getAccuracy(testSet, predictions):
    correct = 0
    count_null = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct / (float(len(testSet)) - count_null)) * 100.0
