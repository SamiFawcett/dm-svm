import pandas as pd
import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler
import sys


def binary_class_assignment(original_response_column, class_threshold):
    new_response_column = []
    for value in original_response_column:
        if (value <= class_threshold):
            new_response_column.append(1)
        elif (value > class_threshold):
            new_response_column.append(-1)
    return np.array(new_response_column)


def p_range(point_view, lower_bound, upper_bound):
    p_subset = []
    for i in range(lower_bound, upper_bound + 1):
        p_subset.append(point_view[i])

    return np.array(p_subset)


def parse(filename):
    headers = list(pd.read_csv(filename, nrows=0).columns)
    headers.pop(len(headers) - 1)
    headers.pop(0)
    data = pd.read_csv(filename, usecols=headers).to_numpy()
    unscaled_column_view = np.transpose(data)

    raw_response_column = unscaled_column_view[0]
    response_column = binary_class_assignment(raw_response_column, 50)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    column_view = list(np.transpose(scaled_data))
    column_view.pop(0)
    column_view.insert(0, response_column)

    return [scaled_data, column_view, response_column, headers]


def augmented_kernel(data, kernel_type, spread):
    k = []
    for point_one in data:
        k_row = []
        for point_two in data:
            if (kernel_type == 'linear'):
                k_row.append(np.dot(point_one, point_two) + 1)
            if (kernel_type == 'gaussian'):
                numer = -1 * np.power(LA.norm(point_one - point_two), 2)
                denom = 2 * spread
                k_row.append(np.exp(numer / denom) + 1)
        k.append(k_row)

    return np.array(k)


def update(alpha, response_column, aug_kern, k_idx):

    suma = 0
    for i in range(0, len(alpha)):
        suma += alpha[i] * response_column[i] * aug_kern[i, k_idx]

    return suma


def svm_dual(data, kernel_type, kernel_param, regularization_constant,
             convergence_threshold, response_column, max_iter):
    aug_k = augmented_kernel(data, kernel_type, kernel_param)
    t = 0
    n = len(data)
    converged = False
    #form = formulation
    alpha_prev = np.zeros(n)
    alpha_next = np.zeros(n)

    step = []
    for k in range(0, n):
        step.append(1 / aug_k[k, k])

    while not converged:
        randomized = np.random.choice(len(data), len(data), False)
        for j in randomized:
            alpha_j = alpha_prev[j] + (
                step[j] * (1 - response_column[j] *
                           update(alpha_prev, response_column, aug_k, j)))

            if (alpha_j < 0):
                alpha_j = 0
            if (alpha_j > convergence_threshold):
                alpha_j = convergence_threshold

            alpha_prev[j] = alpha_j
        alpha_next = alpha_prev
        t = t + 1

        if (LA.norm(np.array(alpha_next) - np.array(alpha_prev)) <=
                convergence_threshold):
            converged = True
        if (t == max_iter):
            converged = True

    return alpha_next


if __name__ == "__main__":
    arguments = sys.argv
    filename = arguments[1]
    regularization_constant = float(arguments[2])
    convergence_threshold = float(arguments[3])
    max_iter = int(arguments[4])
    kernel_type = arguments[5]
    kernel_param = float(arguments[6])

    print("C:", regularization_constant, "EPI:", convergence_threshold,
          "KERNEL TYPE:", kernel_type, "SPREAD:", kernel_param)

    #example input: energydata_complete.csv 0.1 0.001 5000 linear 0.01

    parsed_data = parse(filename)

    #parsed results
    point_view = parsed_data[0]
    column_view = parsed_data[1]
    response_column = parsed_data[2]
    column_headers = parsed_data[3]

    #data partition
    training_data = p_range(point_view, 0, 999)
    training_response = p_range(response_column, 0, 999)
    validation_data = p_range(point_view, 1000, 1399)
    testing_data = p_range(point_view, 1400, 2399)
    testing_response = p_range(response_column, 1400, 2399)

    alphas = svm_dual(training_data, kernel_type, kernel_param,
                      regularization_constant, convergence_threshold,
                      training_response, max_iter)

    if (kernel_type == 'linear'):
        response_prediction = []
        weights = list(np.zeros(len(testing_data[0])))
        aug_k = augmented_kernel(testing_data, kernel_type, kernel_param)
        for i in range(0, len(testing_data)):
            suma = 0
            for k in range(0, len(alphas)):
                if (alphas[k] > 0):
                    weights += alphas[k] * testing_response[k] * testing_data[i]
                    suma += alphas[k] * testing_response[k] * aug_k[k, i]

            if (suma > 0):
                response_prediction.append(1)
            else:
                response_prediction.append(-1)
        print("weights:", weights)
        print("bias:", weights[0])
        accuracy = []
        for i in range(0, len(testing_response)):
            if (response_prediction[i] == testing_response[i]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        print("accuracy: ", sum(accuracy) / len(accuracy))

    if (kernel_type == 'gaussian'):
        response_prediction = []
        weights = list(np.zeros(len(testing_data[0])))
        aug_k = augmented_kernel(testing_data, kernel_type, kernel_param)
        for i in range(0, len(testing_data)):
            suma = 0
            for k in range(0, len(alphas)):
                if (alphas[k] > 0):
                    weights += alphas[k] * testing_response[k] * testing_data[i]
                    suma += alphas[k] * testing_response[k] * aug_k[k, i]

            if (suma > 0):
                response_prediction.append(1)
            else:
                response_prediction.append(-1)
        print("weights:", weights)
        print("bias:", weights[0])
        accuracy = []
        for i in range(0, len(testing_response)):
            if (response_prediction[i] == testing_response[i]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        print("accuracy: ", sum(accuracy) / len(accuracy))
