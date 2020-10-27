import random

import numpy as np
from scipy import sparse

import utils

MOVIE_ID_MIN = 1
MOVIE_ID_MAX = 17770
CUSTOMER_ID_MIN = 1
CUSTOMER_ID_MAX = 2649429
USER_COUNT = 480189
RATING_MIN = 5
RATING_MAX = 5

FACTOR_COUNT = 8  # K


def predict(instance, weights, V):
    featureCount = instance.shape[0]
    assert(weights.shape[0] == featureCount)
    assert(V.shape[1] == FACTOR_COUNT)

    # calculating using Lemma 3.1
    # https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    s = 0

    for f in range(FACTOR_COUNT):
        s1 = s2 = 0
        for i in range(featureCount):
            p = V[i][f] * instance[i]

            s1 += p
            s2 += p * p
        s += s1 * s1 - s2

    return np.dot(instance, weights) + 0.5 * s


def calcR2(instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    a = b = 0
    y_avg = np.average(results)

    for i in range(instanceCount):
        p = predict(instances[i], weights, V)
        y = results[i]
        a += pow(y - p, 2)
        b += pow(y - y_avg, 2)

    return 1 - a / b


def calcMSE(instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    s = 0
    for i in range(instanceCount):
        p = predict(instances[i], weights, V)
        y = results[i]
        s += pow(y - p, 2)

    return 1 / instanceCount * s


# get MSE gradient for FM model
# returns a tuple (weights gradient, V gradient)
def getMSEGradient(instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    w_R = np.ndarray(shape=(1, featureCount))
    V_R = np.ndarray(shape=(featureCount, FACTOR_COUNT))

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

        for inst in range(instanceCount):
            # get result
            y = results[inst]

            # calculate predicted result, using factorization
            # machine model equation
            instance = instances[inst]
            s_predict = 0
            for f in range(FACTOR_COUNT):
                # calculating using Lemma 3.1
                s1 = s2 = 0
                for i in range(featureCount):
                    p = V[i][f] * instance[i]
                    s1 += p
                    s2 += p * p
                s_predict += s1 * s1 - s2
                # also, cache sums for calculating model's partial derivative
                sumsForFactor.append(s1)
            y_p = np.dot(instance, weights) + 0.5 * s_predict
            y_deriv = 0

            # calculate partial derivative for FM model (equation 4)
            if isW:
                # if the variable is from weights
                y_deriv = weights[derivativeId]
            else:
                # calculate derivative for a variable from V
                y_deriv = instance[v_i] * sumsForFactor[v_f] - V[v_i][v_f] * instance[v_i] * instance[v_i]

            s += (y - y_p) * (-1) * y_deriv

            sumsForFactor.clear()

        deriv = 2 / instanceCount * s

        if isW:
            w_R[derivativeId] = deriv
        else:
            V_R[v_i][v_f] = deriv

    return w_R, V_R


def gradientDescent(allInstances, trainIndices, testIndices, results):
    maxIterations = 100
    _, featureCount = allInstances.shape

    # (1 x feature count)
    weights_k = weights_prev = np.full(featureCount, 1)

    # (feature count x factor count)
    V_k = V_prev = sparse.csr_matrix(np.full((featureCount, FACTOR_COUNT)))

    for i in range(1, maxIterations + 1):
        lambda_i = 1 / i

        randId = trainIndices[random.randint(0, len(trainIndices) - 1)]
        instanceRand = np.ndarray([allInstances[randId]])
        resultRand = np.ndarray([results[randId]])

        w_grad, V_grad = getMSEGradient(instanceRand, weights_k, V_k, resultRand)

        weights_k   = weights_prev  - lambda_i * w_grad
        V_k         = V_prev        - lambda_i * V_grad

    for i in testIndices:
        instance = allInstances[i]


def main():
    # X is (number_of_ratings x (number_of_users + number_of_movie_ids))
    X, ratings = utils.getReadyData()

    folds = [()]

    gradientDescent()


main()