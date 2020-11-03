import time
import datetime
import pandas as pd
import numpy as np
from scipy import sparse
from math import sqrt

import utils


FACTOR_COUNT = 2
GRADIENT_MAX_ITER = 5
MINIBATCH_SIZE = 1000
ONES = sparse.csr_matrix((1, 1))


def getB1(instances, V):
    # these 2 sums from lemma 3.1
    # (sum of [V_i,f * x_i])^2
    A0 = instances * V
    # point-wise multiplication, here it's getting square of each element
    A1 = A0.multiply(A0)

    # sum of [(V_i,f)^2 * (x_i)^2]
    sqV = V.multiply(V)
    sqX = instances  # there are only 0 and 1, no need for squaring
    A2 = sqX * sqV

    # (first sum) - (second sum)
    # A0, A1 and A2 shape: (instance count, factor count)
    B0 = A1 - A2
    # sum rows
    B1 = B0 * ONES

    return A0, B1


def calcR2TrainTest(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)
    ta = testRangeTuple[0]
    tb = testRangeTuple[1]

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    predicted = W1 + B1 * 0.5
    assert(predicted.shape[0] == instanceCount)

    yTest_avg = results[ta:tb, 0].mean()

    D0Test = results[ta:tb, 0] - predicted[ta:tb, 0]
    aTest = D0Test.multiply(D0Test).sum()
    D1Test = results[ta:tb, 0] - sparse.csr_matrix([yTest_avg for _ in range(tb - ta)]).transpose()
    bTest = D1Test.multiply(D1Test).sum()

    # [ta, tb][train]
    if ta == 0:
        yTrain_avg = results[(tb + 1):, 0].mean()

        D0Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        aTrain = D0Train.multiply(D0Train).sum()

        D1Train = results[(tb + 1):, 0] - sparse.csr_matrix([yTrain_avg for _ in range(instanceCount - tb - 1)]).transpose()
        bTrain = D1Train.multiply(D1Train).sum()

    # [train][ta, tb]
    elif tb == instanceCount - 1:
        yTrain_avg = results[:(ta - 1), 0].mean()

        D0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        aTrain = D0Train.multiply(D0Train).sum()

        D1Train = results[:(ta - 1), 0] - sparse.csr_matrix([yTrain_avg for _ in range(ta - 1)]).transpose()
        bTrain = D1Train.multiply(D1Train).sum()

    # [train][ta, tb][train]
    else:
        yTrain_avg = (results[(tb + 1):, 0].mean() + results[:(ta - 1), 0].mean()) * 0.5

        D0_0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        D0_1Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        aTrain = D0_0Train.multiply(D0_0Train).sum() + D0_1Train.multiply(D0_1Train).sum()

        D1_0Train = results[:(ta - 1), 0] - sparse.csr_matrix([yTrain_avg for _ in range(ta - 1)]).transpose()
        D1_1Train = results[(tb + 1):, 0] - sparse.csr_matrix([yTrain_avg for _ in range(instanceCount - tb - 1)]).transpose()
        bTrain = D1_0Train.multiply(D1_0Train).sum() + D1_1Train.multiply(D1_1Train).sum()

    return 1 - aTrain / bTrain, 1 - aTest / bTest


def calcMSETrainTest(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)
    ta = testRangeTuple[0]
    tb = testRangeTuple[1]

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    predicted = W1 + B1 * 0.5
    assert(predicted.shape[0] == instanceCount)

    D0Test = results[ta:tb, 0] - predicted[ta:tb, 0]
    sumTest = D0Test.multiply(D0Test).sum()

    # [ta, tb][train]
    if ta == 0:
        D0Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        sumTrain = D0Train.multiply(D0Train).sum()

    # [train][ta, tb]
    elif tb == instanceCount - 1:
        D0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        sumTrain = D0Train.multiply(D0Train).sum()

    # [train][ta, tb][train]
    else:
        D0_0Train = results[:(ta - 1), 0] - predicted[:(ta - 1), 0]
        D0_1Train = results[(tb + 1):, 0] - predicted[(tb + 1):, 0]
        sumTrain = D0_0Train.multiply(D0_0Train).sum() + D0_1Train.multiply(D0_1Train).sum()

    return 1 / instanceCount * sumTrain, 1 / instanceCount * sumTest


def calcRMSETrainTest(instances, weights, V, results, testRangeTuple):
    tr, ts = calcMSETrainTest(instances, weights, V, results, testRangeTuple)
    return sqrt(tr), sqrt(ts)


# get MSE gradient for FM model
# returns a tuple (weights gradient, V gradient)
def getMSEGradient(instanceIDs, instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)
    assert(len(instanceIDs) == MINIBATCH_SIZE)

    stime = time.time()

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    predicted = W1 + B1 * 0.5
    assert(predicted.shape[0] == instanceCount)

    # transposed for +=
    w_R = sparse.csr_matrix((1, featureCount))
    for a in instanceIDs:
        w_R += (results[a, 0] - predicted[a, 0]) * (instances[a, :])
    w_R = 2.0 / len(instanceIDs) * w_R
    w_R = w_R.transpose()

    II = instances.multiply(instances).transpose()

    V_R = sparse.lil_matrix((featureCount, FACTOR_COUNT))
    for v_f in range(FACTOR_COUNT):
        for a in instanceIDs:
            #stime = time.time()
            s = instances[a, :].transpose() * A0[a, v_f] - V[:, v_f].multiply(II[:, a])
            V_R[:, v_f] = (results[a, 0] - predicted[a, 0]) * s
            #print(time.time() - stime)

    V_R = 2.0 / len(instanceIDs) * V_R

    print("Done in %s sec." % str(time.time() - stime))
    return w_R.tocsr(), V_R.tocsr()


def gradientDescent(allInstances, testRangeTuple, results):
    instanceCount, featureCount = allInstances.shape

    # (1 x feature count)
    weights_k = weights_prev = sparse.rand(featureCount, 1, 0.001).tocsr()

    # (feature count x factor count)
    V_k = V_prev = sparse.rand(featureCount, FACTOR_COUNT, 0.001).tocsr()

    for i in range(1, GRADIENT_MAX_ITER + 1):
        print("    Gradient descent iteration #%i... " % i, end='')
        lambda_i = 1 / i

        randIds = [utils.getRandomExcept(instanceCount, testRangeTuple) for _ in range(MINIBATCH_SIZE)]

        w_grad, V_grad = getMSEGradient(randIds, allInstances, weights_k, V_k, results)

        weights_k = weights_prev - lambda_i * w_grad
        V_k = V_prev - lambda_i * V_grad

        weights_prev = weights_k
        V_prev = V_k

    r2_train, r2_test = calcR2TrainTest(allInstances, weights_k, V_k, results, testRangeTuple)
    rmse_train, rmse_test = calcRMSETrainTest(allInstances, weights_k, V_k, results, testRangeTuple)

    return weights_k, V_k, r2_test, rmse_test, r2_train, rmse_train


def main():
    print("Start time:         %s" % str(datetime.datetime.now()))

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
        index=["R^2", "RMSE", "R^2-train", "RMSE-train"]
    )

    global ONES
    ONES = sparse.csr_matrix([1 for _ in range(FACTOR_COUNT)]).transpose()

    print("Loading done:       %s" % str(datetime.datetime.now()))

    for i in range(foldCount):
        print("Fold #%i started:    %s" % (i + 1, str(datetime.datetime.now())))

        weights, V, r2, rmse, r2_tr, rmse_tr = gradientDescent(X, folds[i], ratings)

        resultTable.insert(i, "T%i" % (i + 1), np.array([r2, rmse, r2_tr, rmse_tr]))

        print("    Fold #%i R^2-train  : %f" % (i + 1, r2_tr))
        print("    Fold #%i RMSE-train : %f" % (i + 1, rmse_tr))
        print("    Fold #%i R^2-test   : %f" % (i + 1, r2))
        print("    Fold #%i RMSE-test  : %f\n" % (i + 1, rmse))

    print("Folds done:         %s" % str(datetime.datetime.now()))

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("out.csv")

    print("Data saved:         %s" % str(datetime.datetime.now()))


main()
