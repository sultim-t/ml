import pandas as pd
import numpy as np
from scipy import sparse
from math import sqrt

import utils

MOVIE_ID_MIN = 1
MOVIE_ID_MAX = 17770
CUSTOMER_ID_MIN = 1
CUSTOMER_ID_MAX = 2649429
USER_COUNT = 480189
RATING_MIN = 5
RATING_MAX = 5

FACTOR_COUNT = 4  # K
START_W = 1 / 10000000
GRADIENT_MAX_ITER = 5


def predict(instID, instances, weights, V):
    featureCount = instances.shape[1]
    assert(weights.shape[0] == featureCount)
    assert(V.shape[1] == FACTOR_COUNT)

    # look getMSEGradient for comments
    s_predict = 0
    for f in range(FACTOR_COUNT):
        a = V[:, f].multiply(instances[instID, :].transpose())
        s1 = a.sum()
        s2 = a.multiply(a).sum()
        s_predict += s1 * s1 - s2

    return instances[instID, :].dot(weights[:, 0])[0, 0] + 0.5 * s_predict


def calcR2TrainTest(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    aTrain = bTrain = 0
    aTest = bTest = 0
    y_avg = np.average(results)

    for i in range(instanceCount):
        p = predict(i, instances, weights, V)
        y = results[i]

        if testRangeTuple[0] <= i <= testRangeTuple[1]:
            aTest += pow(y - p, 2)
            bTest += pow(y - y_avg, 2)
        else:
            aTrain += pow(y - p, 2)
            bTrain += pow(y - y_avg, 2)

    return 1 - aTrain / bTrain, 1 - aTest / bTest


def calcMSETrainTest(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    sumTrain = 0
    sumTest = 0
    for i in range(instanceCount):
        p = predict(i, instances, weights, V)
        y = results[i]

        if testRangeTuple[0] <= i <= testRangeTuple[1]:
            sumTest += pow(y - p, 2)
        else:
            sumTrain += pow(y - p, 2)

    return 1 / instanceCount * sumTrain, 1 / instanceCount * sumTest


def calcRMSETrainTest(instances, weights, V, results, testRangeTuple):
    tr, ts = calcMSETrainTest(instances, weights, V, results, testRangeTuple)
    return sqrt(tr), sqrt(ts)


# get MSE gradient for FM model
# returns a tuple (weights gradient, V gradient)
def getMSEGradient(instanceID, instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    w_R = sparse.lil_matrix((featureCount, 1))
    V_R = sparse.lil_matrix((featureCount, FACTOR_COUNT))

    sumsForFactor = []

    # get derivative over each variable in weights and V
    for derivativeId in range(weights.shape[0] + V.shape[0] + V.shape[1]):
        s = v_i = v_f = 0
        # if the variable is from weights
        isW = derivativeId < weights.shape[0]
        if not isW:
            # decode i and f indices of V
            v_id = derivativeId - weights.shape[0]
            v_i = v_id // V.shape[1]
            v_f = v_id % V.shape[1]

        # for only one
        for inst in range(instanceID, instanceID + 1):
            # get result
            y = results[inst, 0]

            # calculate predicted result, using factorization
            # machine model equation
            s_predict = 0
            for f in range(FACTOR_COUNT):
                # calculating using Lemma 3.1
                # per element multiplication, i.e. V[_, f] * instances[inst, _]
                a = V[:, f].multiply(instances[inst, :].transpose())
                s1 = a.sum()
                # sum of squared elements
                s2 = a.multiply(a).sum()

                s_predict += s1 * s1 - s2
                # also, cache sums for calculating model's partial derivative
                sumsForFactor.append(s1)
            y_p = instances[inst, :].dot(weights[:, 0])[0, 0] + 0.5 * s_predict

            # calculate partial derivative for FM model (equation 4)
            if isW:
                # if the variable is from weights
                y_deriv = weights[derivativeId, 0]
            else:
                # calculate derivative for a variable from V
                y_deriv = instances[inst, v_i] * sumsForFactor[v_f] - V[v_i, v_f] * instances[inst, v_i] * instances[inst, v_i]

            s += (y - y_p) * (-1) * y_deriv

            sumsForFactor.clear()

        # only one is used
        # deriv = 2 / instanceCount * s
        deriv = 2 * s

        if isW:
            w_R[derivativeId, 0] = deriv
        else:
            V_R[v_i, v_f] = deriv

    print("getMSEGradient done.")
    return w_R.tocsr(), V_R.tocsr()


def gradientDescent(allInstances, testRangeTuple, results):
    instanceCount, featureCount = allInstances.shape

    # (1 x feature count)
    weights_k = weights_prev = sparse.csr_matrix(np.full((featureCount, 1), START_W))

    # (feature count x factor count)
    V_k = V_prev = sparse.csr_matrix(np.full((featureCount, FACTOR_COUNT), START_W))

    for i in range(1, GRADIENT_MAX_ITER + 1):
        lambda_i = 1 / i
        randId = utils.getRandomExcept(instanceCount, testRangeTuple)

        w_grad, V_grad = getMSEGradient(randId, allInstances, weights_k, V_k, results)

        weights_k = weights_prev - lambda_i * w_grad
        V_k = V_prev - lambda_i * V_grad

    r2_train, r2_test = calcR2TrainTest(allInstances, weights_k, V_k, results, testRangeTuple)
    rmse_train, rmse_test = calcRMSETrainTest(allInstances, weights_k, V_k, results, testRangeTuple)

    return weights_k, V_k, r2_test, rmse_test, r2_train, rmse_train

def main():
    # X is (number_of_ratings x (number_of_users + number_of_movie_ids))
    X, ratings = utils.getReadyData()
    instanceCount, featureCount = X.shape

    foldCount = 5
    folds = []
    step = instanceCount // foldCount
    for i in range(foldCount - 1):
        folds.append((i * step, (i + 1) * step - 1))
    folds.append(((foldCount - 1) * step, instanceCount - 1))

    resultTable = pd.DataFrame(
        columns=[],
        index=["R^2", "RMSE", "R^2-train", "RMSE-train"]  # +
              # ["w" + str(i) for i in range(featureCount)] +
              # ["V" + str(i) + str(j) for i in range(featureCount) for j in range(FACTOR_COUNT)]
    )

    for i in range(foldCount):
        weights, V, r2, rmse, r2_tr, rmse_tr = gradientDescent(X, folds[i], ratings)

        #V = V.reshape((-1, 1))

        resultTable.insert(i, "T" + str(i + 1), np.array([r2, rmse, r2_tr, rmse_tr]))
            #np.concatenate((np.array([r2, rmse, r2_tr, rmse_tr]), weights, V)))

        print("Fold #" + str(i) + "R^2-train: " + str(r2_tr))
        print("Fold #" + str(i) + "RMSE-train: " + str(rmse_tr))

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("out.csv")


main()