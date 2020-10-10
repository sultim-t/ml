import numpy as np
from math import sqrt
import random

def getFeatureNames():
    # page features
    page = [
        "Page Likes",
        "Page Category",
        "Page Checkin",
        "Page Talking About",  # number of people who are active on this page
    ]

    # comments features
    # before base time
    essential = [
        "Comments Total",
        "Comments Last 24h",
        "Comments 48h-24h",
        "Comments First 24h",
        "Comments Diff C2C3",
    ]

    # min, max, average, median, standard derivation for essentials
    ds = ["Min", "Max", "Avg", "Med", "Std"]
    derived = [c + " @" + d for c in essential for d in ds]

    ps = [
        "Base time",  # [0..71]
        "Post Length",
        "Post Share Count",
        "Post Promotion Status",  # [0,1]
        "Target Hours"  # [0..23], predict for this amount of hours
    ]

    weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    # [0,1]
    days = ["Post published @" + w for w in weekdays]
    days += ["Base date/time @" + w for w in weekdays]

    return page + derived + essential + ps + days


def getLinFeatureNames():
    return [
        "Post published @Sat",
        "Base date/time @Sat",
        "Comments Total @Avg",
        "Comments Last 24h @Avg",
        "Comments 48h-24h @Avg",
        "Comments First 24h @Avg",
        "Comments Diff C2C3 @Avg"
    ]


def getFeatureAbbr():
    return [i for i in range(0, 54)]


def formatDataSet(data):
    instances = data.to_numpy()
    # последний столбец -- результаты, и удалим его из instances
    featureCount = len(data.columns) - 1
    results = instances[:, featureCount]
    instances = np.delete(instances, featureCount, axis=1)
    # добавим столбец с единицами для умножения матриц на вектор
    instances = np.insert(instances, 0, 1, axis=1)
    featureCount += 1
    return instances, results, featureCount


# линейная, instance[0] всегда 1 - для weights[0]
def predict(instance, weights):
    assert(len(instance) == len(weights))
    return np.dot(instance, weights)


# частная производная по j-ой переменной весов
def partialDerivativeMSE(j, instances, weights, results, useL2Rglrz = False):
    instanceCount, featureCount = instances.shape
    s = 0
    pdL2 = 0
    alphaRglrz = 0  # TODO

    for i in range(instanceCount):
        inst = instances[i]
        result = results[i]
        d = result - predict(inst, weights)
        s += d * (-inst[j])

    pdQ = 2 / instanceCount * s

    # частная производная L2 по j-ой переменной = 2*w[j];
    # не регуляризуем w0
    if useL2Rglrz and j != 0:
        pdL2 = 2 * weights[j]

    if useL2Rglrz:
        return pdQ + alphaRglrz * pdL2
    else:
        return pdQ


def calcR2(instances, weights, results):
    instanceCount, featureCount = instances.shape
    a = b = 0
    y_avg = np.average(results)

    for i in range(instanceCount):
        p = predict(instances[i], weights)
        y = results[i]
        a += pow(y - p, 2)
        b += pow(y - y_avg, 2)

    return 1 - a / b


def calcMSE(instances, weights, results):
    instanceCount, featureCount = instances.shape
    s = 0

    for i in range(instanceCount):
        p = predict(instances[i], weights)
        y = results[i]
        s += pow(y - p, 2)

    return 1 / instanceCount * s


def calcRMSE(instances, weights, results):
    return sqrt(calcMSE(instances, weights, results))
