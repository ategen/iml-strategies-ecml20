import numpy as np
from sklearn.neighbors import KNeighborsClassifier  ####### IMPORT RELEVANT CLASSIFIER
import IML_strategies as iml


def learning_sklearn(
    dataset, classes, max_sep, thresh_window, thresh_budget, strategy, ML, ml_parameter
):
    accuracies_accu = []
    labels = []
    data_stream = []
    predictions = []
    tot_count = 0
    pred_sum = 0
    prior_counter = np.zeros(np.shape(classes))
    last_query_counter = 1
    class_counter = {}
    thresh_counter = np.zeros(
        thresh_window
    )  # array that keeps track of  how many of the instances within the last window time frame has been queried (0/1)
    query_counter = 0
    prev_label = -1
    last_query = -1
    predict = False

    ml_model = initialize_ml_model(ML, ml_parameter)

    for d in dataset:
        data_instance = [item for item in d]

        ######## DO PREDICTIONS AND GET ACCURACY ###########
        if prev_label < 0:  # first data point in stream
            query = True
            query_user = True
        else:
            (
                predict,
                result,
                predictions,
                labels,
                accuracies_accu,
                tot_count,
                pred_sum,
            ) = test_model(
                ML,
                ml_parameter,
                ml_model,
                data_stream,
                prior_counter,
                data_instance,
                last_query,
                predictions,
                tot_count,
                accuracies_accu,
                pred_sum,
                d,
                labels,
            )

            query, last_query_counter, query_user = iml.query_main(
                strategy,
                thresh_budget,
                thresh_counter,
                ML,
                ml_parameter,
                ml_model,
                predict,
                data_instance,
                data_stream,
                result,
                prev_label,
                last_query_counter,
                query,
            )
        ######## QUERY FOR LABEL AND UPDATE MODEL #########
        if query:
            (
                query_counter,
                data_stream,
                class_counter,
                prior_counter,
                last_query,
                ml_model,
                thresh_tmp,
            ) = update_model(
                query_user,
                query_counter,
                last_query,
                prior_counter,
                class_counter,
                data_stream,
                data_instance,
                max_sep,
                d,
                ml_model,
            )
        else:
            thresh_tmp = 0

        # shift all values in thresh_counter (whether or not there has been a query)
        thresh_counter[1:] = thresh_counter[:-1]
        thresh_counter[0] = thresh_tmp

        prev_label = d[-1]

    return accuracies_accu, predictions, labels, query_counter


def initialize_ml_model(ML, ml_parameter):
    if ML == "knn":
        k_value = ml_parameter
        ml_model = KNeighborsClassifier(n_neighbors=k_value)
    else:
        print("Please choose valid ML strategy")
    return ml_model


def test_model(
    ML,
    ml_parameter,
    ml_model,
    data_stream,
    prior_counter,
    data_instance,
    last_query,
    predictions,
    tot_count,
    accuracies_accu,
    pred_sum,
    d,
    labels,
):
    if ML == "knn":
        predict = predict_boolean_KNN(
            ml_parameter, len(data_stream), prior_counter
        )  # check if at least k data points and if at least two classes
    if predict:
        # predict based on current model
        result = ml_model.predict(np.array(data_instance[:-1]).reshape(1, -1))
        result = result[0]
    else:  # not enough data points
        result = last_query

    predictions.append(result)
    # test accuracy
    accuracy, tot_count, pred_sum = get_accuracy_add_on(
        tot_count, pred_sum, d[-1], result
    )
    accuracies_accu.append(accuracy)
    labels.append(d[-1])
    return predict, result, predictions, labels, accuracies_accu, tot_count, pred_sum


def update_model(
    query_user,
    query_counter,
    last_query,
    prior_counter,
    class_counter,
    data_stream,
    data_instance,
    max_sep,
    d,
    ml_model,
):
    if query_user:
        query_counter = query_counter + 1
        thresh_tmp = 1
    else:
        data_instance[-1] = last_query
        thresh_tmp = 0

    data_stream, class_counter = append_stream(
        data_stream, class_counter, data_instance, max_sep
    )
    prior_counter[int(data_instance[-1])] = prior_counter[int(data_instance[-1])] + 1
    if query_user:
        last_query = d[-1]

    prior_counter_bool = [i > 1 for i in prior_counter]
    if sum(prior_counter_bool) > 1:  # total number of class labels > 1
        # update model with new value
        X, Y = update_model_sklearn(data_stream)
        ml_model.fit(X, Y)

    return (
        query_counter,
        data_stream,
        class_counter,
        prior_counter,
        last_query,
        ml_model,
        thresh_tmp,
    )


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def update_model_sklearn(data_stream):
    X = []
    Y = []
    for d in data_stream:
        X.append(d[:-1])
        Y.append(d[-1])
    return X, Y


def append_stream(data_stream, class_counter, data_instance, max_sep):
    data_stream.append(data_instance)
    if int(data_instance[-1]) not in class_counter:
        class_counter[int(data_instance[-1])] = 1
    elif class_counter[int(data_instance[-1])] >= max_sep:
        for i, d in enumerate(data_stream):
            if int(d[-1]) == int(data_instance[-1]):
                index = i
                break
        data_stream.pop(index)
    else:
        class_counter[int(data_instance[-1])] = (
            class_counter[int(data_instance[-1])] + 1
        )

    return data_stream, class_counter


def get_accuracy(testSet, predictions):
    correct = 0
    count_null = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct / (float(len(testSet)) - count_null)) * 100.0


def get_accuracy_add_on(tot_count, pred_sum, test, pred):
    if test == pred:
        pred_sum = pred_sum + 1

    tot_count = tot_count + 1
    accuracy = (pred_sum / tot_count) * 100

    return accuracy, tot_count, pred_sum


def predict_boolean(prior_counter):
    prior_counter_bool = [i > 1 for i in prior_counter]
    return sum(prior_counter_bool) > 1  # class labels


def predict_boolean_KNN(k, len_data_stream, prior_counter):
    if predict_boolean(prior_counter):
        return len_data_stream < k
    return False
