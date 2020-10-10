import pandas as pd
import numpy as np
import utils
import random

# метод градиентного спуска
# возвращает (weights, R^2, RMSE, R^2-train, RMSE-train)
def gradientDescent(trainSet, testSet):
    maxIterations = 100
    useStochastic = False
    #gradEpsilon = 0.01

    instances, results, featureCount = utils.formatDataSet(trainSet)

    # начальное значение весов
    w0 = np.full(featureCount, 1 / featureCount)
    wk = wk_prev = w0
    k = 1

    while k < maxIterations:
        lambda_k = 1 / k

        # стохастический граиентный спуск
        if useStochastic:
            instRand = instances[random.randint(0, len(instances) - 1)]
            inst = np.array([instRand])
            # только одно
            gradient = np.array([
                utils.partialDerivativeMSE(j, inst, wk_prev, results)
                for j in range(0, featureCount)])
        else:
            gradient = np.array([
                utils.partialDerivativeMSE(j, instances, wk_prev, results)
                for j in range(0, featureCount)])

        wk = wk_prev - lambda_k * gradient

        #if lambda_k * np.linalg.norm(gradient) < gradEpsilon:
        #    break

        wk_prev = wk
        k += 1

    r2_train = utils.calcR2(instances, wk, results)
    rmse_train = utils.calcRMSE(instances, wk, results)

    test_instances, test_results, _ = utils.formatDataSet(testSet)
    r2 = utils.calcR2(test_instances, wk, test_results)
    rmse = utils.calcRMSE(test_instances, wk, test_results)

    print("Gradient descent done for one data set.")

    return wk, r2, rmse, r2_train, rmse_train


def crossValidate():
    # содержит последнюю колонку "Results"
    featureNames = utils.getFeatureNames()
    linFeatureNames = utils.getLinFeatureNames()

    tests = [
        "Dataset/Testing/TestSet/Test_Case_1.csv",
        "Dataset/Testing/TestSet/Test_Case_2.csv",
        "Dataset/Testing/TestSet/Test_Case_3.csv",
        "Dataset/Testing/TestSet/Test_Case_4.csv",
        "Dataset/Testing/TestSet/Test_Case_5.csv"
        #"Dataset/Training/Features_Variant_5.csv",
    ]
    testCount = len(tests)

    allSets = [
        pd.read_csv(file, names=featureNames + ["Answer"])
        for file in tests
    ]

    for data in allSets:
        # нормировка
        for a in featureNames:
            d = data[a].max() - data[a].min()
            if d != 0:
                data[a] = (data[a] - data[a].min()) / d
        # удалим столбцы с лин зависимостью -- дни недели, avg
        for lfn in linFeatureNames:
            del data[lfn]

    featureCount = len(featureNames) - len(linFeatureNames)
    resultTable = pd.DataFrame(
        columns=[],
        # index=["R^2", "RMSE", "R^2-train", "RMSE-train"] + ["f" + str(i) for i in range(featureCount)]
        index=["R^2", "RMSE", "R^2-train", "RMSE-train"] + [f for f in featureNames if f not in linFeatureNames]
    )

    for i in range(testCount):
        # i-ый
        testSet = allSets[i]
        # concat csv кроме i-ого
        trainSet = pd.concat([
            allSets[j] for j in range(testCount) if j != i
        ])

        weights, r2, rmse, r2_t, rmse_t = gradientDescent(trainSet, testSet)
        resultTable.insert(i, "T" + str(i + 1), np.concatenate((np.array([r2, rmse, r2_t, rmse_t]), weights[1:])))

    e = resultTable.mean(axis=1)
    std = resultTable.std(axis=1)
    resultTable.insert(5, "E", e)
    resultTable.insert(6, "STD", std)

    print(resultTable.head())

    resultTable.to_csv("out.csv")


crossValidate()
