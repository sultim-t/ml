import time
import pandas as pd
import numpy as np
from scipy import sparse
from math import sqrt

import utils


FACTOR_COUNT = 2  # K
GRADIENT_MAX_ITER = 10
MINIBATCH_SIZE = 1000


def getB1(instances, V):
    # these 2 sums from lemma 3.1
    # (sum of [V_i,f * x_i])^2
    A0 = instances * V
    # point-wise multiplication, here it's getting square of each element
    A1 = A0.multiply(A0)

    # sum of [(V_i,f)^2 * (x_i)^2]
    sqV = V.multiply(V)
    #sqX = instances.multiply(instances)
    sqX = instances  # there are only 0 and 1, no need for squaring
    A2 = sqX * sqV

    # (first sum) - (second sum)
    # A0, A1 and A2 shape: (instance count, factor count)
    B1 = A1 - A2

    return A0, B1


def calcR2TrainTest(instances, weights, V, results, testRangeTuple):
    #print("Calculating R2 for a train and test set...")
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    aTrain = bTrain = 0
    aTest = bTest = 0
    yTest_avg = results[testRangeTuple[0]:testRangeTuple[1], 0].mean()
    if testRangeTuple[0] == 0:
        yTrain_avg = results[testRangeTuple[1]:, 0].mean()
    elif testRangeTuple[1] == instanceCount:
        yTrain_avg = results[:testRangeTuple[0], 0].mean()
    else:
        yTrain_avg = (results[testRangeTuple[1]:, 0].mean() + results[:testRangeTuple[0], 0].mean()) * 0.5

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    for i in range(instanceCount):
        p = W1[i, 0] + B1[i, :].sum() * 0.5

        y = results[i, 0]

        if testRangeTuple[0] <= i <= testRangeTuple[1]:
            aTest += pow(y - p, 2)
            bTest += pow(y - yTest_avg, 2)
        else:
            aTrain += pow(y - p, 2)
            bTrain += pow(y - yTrain_avg, 2)

    r2Train = 1 - aTrain / bTrain
    r2Test = 1 - aTest / bTest
    #print("    R^2 train: %f, R^2 test: %f" % (r2Train, r2Test))
    return r2Train, r2Test


def calcMSETrainTest(instances, weights, V, results, testRangeTuple):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)

    sumTrain = 0
    sumTest = 0

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    for i in range(instanceCount):
        p = W1[i, 0] + B1[i, :].sum() * 0.5
        y = results[i, 0]

        if testRangeTuple[0] <= i <= testRangeTuple[1]:
            sumTest += pow(y - p, 2)
        else:
            sumTrain += pow(y - p, 2)

    return 1 / instanceCount * sumTrain, 1 / instanceCount * sumTest


def calcRMSETrainTest(instances, weights, V, results, testRangeTuple):
    #print("Calculating RMSE for a train and test set...")
    tr, ts = calcMSETrainTest(instances, weights, V, results, testRangeTuple)
    rmseTrain = sqrt(tr)
    rmseTest = sqrt(ts)
    #print("    RMSE train: %f, RMSE test: %f" % (rmseTrain, rmseTest))
    return rmseTrain, rmseTest


# get MSE gradient for FM model
# returns a tuple (weights gradient, V gradient)
def getMSEGradient(instanceIDs, instances, weights, V, results):
    instanceCount, featureCount = instances.shape
    assert(weights.shape[0] == featureCount == V.shape[0])
    assert(V.shape[1] == FACTOR_COUNT)
    assert(results.shape[0] == instanceCount)
    assert(len(instanceIDs) == MINIBATCH_SIZE)

    #w_R = sparse.csr_matrix((featureCount, 1))
    V_R = sparse.lil_matrix((featureCount, FACTOR_COUNT))

    derivCount = weights.shape[0] + V.shape[0] + V.shape[1]
    print("Partial derivative count: %i" % derivCount)

    A0, B1 = getB1(instances, V)
    W1 = instances * weights

    print_iter = 0

    stime = time.time()
    predicted = W1 + B1.sum(1) * 0.5

    print("   predicted ", end='')
    print(time.time() - stime)
    stime = time.time()

    #for w_i in range(featureCount):
        #s_wt = (np.multiply(results - predicted, ((-1) * instances[:, w_i]).toarray())).sum()
    #reversed dims for +=
    w_R = sparse.csr_matrix((1,featureCount))
    for a in instanceIDs:
        w_R += (results[a, 0] - predicted[a, 0]) * (instances[a, :])
    w_R = 2.0 / len(instanceIDs) * w_R
    w_R = w_R.transpose()

    print("    w_R ", end='')
    print(time.time() - stime)
    stime = time.time()

    II = instances.multiply(instances).transpose()

    print("    II ", end='')
    print(time.time() - stime)
    stime = time.time()

    #for v_f in range(FACTOR_COUNT):
    #    for v_i in range(featureCount):
    #        ys_deriv = instances[:, v_i].multiply(A0[:, v_f]) - V[v_i, v_f] * II[:, v_i]  # * instances[:, v_i].multiply(instances[:, v_i])
    #        s_v = np.multiply((results - predicted), ((-1) * ys_deriv).toarray()).sum()
    #        V_R[v_i, v_f] = 2.0 / instanceCount * s_v

    for v_f in range(FACTOR_COUNT):
        for a in instanceIDs:
            #stime = time.time()
            s = instances[a, :].transpose() * A0[a, v_f] - V[:, v_f].multiply(II[:, a])
            V_R[:, v_f] = (results[a, 0] - predicted[a, 0]) * s
            #print(time.time() - stime)


    V_R = 2.0 / len(instanceIDs) * V_R

    print("    VFVI ", end='')
    print(time.time() - stime)

    print("    Partial derivatives for V done.")
    print("getMSEGradient done.")
    return w_R.tocsr(), V_R.tocsr()


def gradientDescent(allInstances, testRangeTuple, results):
    instanceCount, featureCount = allInstances.shape

    # (1 x feature count)
    weights_k = weights_prev = sparse.rand(featureCount, 1, 0.001).tocsr()

    # (feature count x factor count)
    V_k = V_prev = sparse.rand(featureCount, FACTOR_COUNT, 0.001).tocsr()

    for i in range(1, GRADIENT_MAX_ITER + 1):
        print("Gradient descent iteration #%i..." % i)
        lambda_i = 0.4

        randIds = [utils.getRandomExcept(instanceCount, testRangeTuple) for _ in range(MINIBATCH_SIZE)]

        w_grad, V_grad = getMSEGradient(randIds, allInstances, weights_k, V_k, results)

        weights_k = weights_prev - lambda_i * w_grad
        V_k = V_prev - lambda_i * V_grad

    r2_train, r2_test = calcR2TrainTest(allInstances, weights_k, V_k, results, testRangeTuple)
    rmse_train, rmse_test = calcRMSETrainTest(allInstances, weights_k, V_k, results, testRangeTuple)

    return weights_k, V_k, r2_test, rmse_test, r2_train, rmse_train

def main():
    # X is (number_of_ratings x (number_of_users + number_of_movie_ids))
    X, ratings = utils.getReadyData()
    instanceCount, featureCount = X.shape

    print("Data loaded.")

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

    print("Folds created.")

    for i in range(foldCount):
        print("Processing fold #%i..." % (i + 1))

        weights, V, r2, rmse, r2_tr, rmse_tr = gradientDescent(X, folds[i], ratings)

        #V = V.reshape((-1, 1))

        resultTable.insert(i, "T%i" % (i + 1), np.array([r2, rmse, r2_tr, rmse_tr]))
            #np.concatenate((np.array([r2, rmse, r2_tr, rmse_tr]), weights, V)))

        print("Fold #%i  R^2-train: %f" % (i + 1, r2_tr))
        print("Fold #%i RMSE-train: %f\n" % (i + 1, rmse_tr))

        #with open("T%i_w.npy" % i, 'wb') as f:
        #    sparse.save_npz(f, weights)
        #with open("T%i_V.npy" % i, 'wb') as f:
        #    sparse.save_npz(f, V)

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("out.csv")


main()