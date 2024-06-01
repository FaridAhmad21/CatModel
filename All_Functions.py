import copy

import numpy as np
import Sigmoid_formula as sf


def initialize_with_zero(dim):
    w = np.zeros((dim, 1))
    b = 0.0

    return w, b


def propagate(w, b, X, Y):

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sf.sigmoid(np.dot(w.T, X) + b)

    cost = np.sum(Y * (np.log(A)) + (- np.log(1 - A) * (1 - Y))) / m

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m

    cost = np.squeeze(np.array(cost))



    grads = {"dw": dw,
             "db": db}

    return grads, cost

    # m = X.shape[1]
    #
    # # FORWARD PROPAGATION (FROM X TO COST)
    # ### START CODE HERE ### (≈ 2 lines of code)
    # A = sf.sigmoid(np.dot(w.T, X) + b)  # compute activation
    # cost = (-1. / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis=1)  # compute cost
    # ### END CODE HERE ###
    #
    # # BACKWARD PROPAGATION (TO FIND GRAD)
    # ### START CODE HERE ### (≈ 2 lines of code)
    # dw = (1. / m) * np.dot(X, ((A - Y).T))
    # db = (1. / m) * np.sum(A - Y, axis=1)
    # ### END CODE HERE ###
    #
    # assert (dw.shape == w.shape)
    # assert (db.dtype == float)
    # cost = np.squeeze(cost)
    # assert (cost.shape == ())
    #
    # grads = {"dw": dw,
    #          "db": db}
    #
    # return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_Prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sf.sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] >= 0.5:
            Y_Prediction[0, i] = 1
        else:
            Y_Prediction[0, i] = 0

    assert (Y_Prediction.shape == (1, m))

    return Y_Prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zero(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_Prediction_test = predict(w, b, X_test)
    Y_Prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_Prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_Prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_Prediction_test,
         "Y_prediction_train": Y_Prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
