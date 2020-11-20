import datetime
import numpy as np
import utils
import random

AP_MAX_ITERATIONS = 5
# to control cluster size
AP_S_KK = -20
TEST_CHECKIN_COUNT = 100


def affinityPropagation(S):
    N = utils.NODE_COUNT
    assert(S.shape[0] == S.shape[1] == N)

    # set all diagonal elements S[k,k] to AP_S_KK
    S.flat[::(N + 1)] = AP_S_KK

    R = np.zeros((N, N))
    A = np.zeros((N, N))

    Ids = np.arange(N)

    for _ in range(AP_MAX_ITERATIONS):
        print("  Iteration start:  %s" % str(datetime.datetime.now()))
        AS = A + S

        # find max without consideration of j=k
        M0 = AS.max(axis=1)
        amax = AS.argmax(axis=1)

        # now set all max values to -inf
        AS[Ids, amax] = -np.inf
        # and find max again
        M1 = AS.max(axis=1)

        R = S - M0
        # if j=k, this will overwrite them
        # with other max
        R[Ids, amax] = S[Ids, amax] - M1

        #print("  R:                %s" % str(datetime.datetime.now()))

        # availability
        M2 = np.maximum(R, 0)

        # ignore A[k,k] for now, it'll be rewritten later
        S2 = M2.sum(axis=0)

        for k in range(N):
            A[:, k] = R[k, k] + S2[k] - M2[:, k] - M2[k, k]

        A = np.minimum(A, 0)
        #print("  A:                %s" % str(datetime.datetime.now()))

        # set diagonal elements
        # A[k,k]: sum over all j, but without j=k (i.e. diagonal)
        A.flat[::(N + 1)] = M2.sum(axis=0) - M2.diagonal()
        #print("  A diagonal:       %s" % str(datetime.datetime.now()))

    # argmax on each row
    return (A + R).argmax(axis=1)


# find top recommendations for each cluster
def findRecommendations(user2Cluster, clusters, users, locations):
    # dict (clusters) of dicts (count for each location)
    rcmd = {i: {} for i in clusters}

    for user, loc in zip(users, locations):
        c = user2Cluster[user]

        cr = rcmd[c]
        if loc in cr:
            # increment if location is in cluster dict
            cr[loc] += 1
        else:
            # init value for a new location for this cluster dict
            cr[loc] = 1

    # top 10 locations for each cluster
    result = {}

    for clusterId, clusterDict in rcmd.items():
        # list of locationIds: firstly,
        # sort clusterDict by values and then get keys
        srt = [k for k, v in
               sorted(
                   clusterDict.items(),
                   key=lambda i: i[1],
                   reverse=True)]

        result[clusterId] = srt[:10]

    return result


def testCheckins(checkins, recommends):
    count = 0

    for user, cluster, loc in checkins:
        locTopList = recommends[cluster]
        if locTopList is not None:
            # check every top location of user's cluster
            for i in locTopList:
                if i == loc:
                    count += 1
                    break
    # return accuracy
    return count / TEST_CHECKIN_COUNT


def main():
    print("Start time:         %s" % str(datetime.datetime.now()))

    graph = utils.getGraph()
    print("Loading done:       %s" % str(datetime.datetime.now()))

    user2Cluster = affinityPropagation(graph.toarray())
    clusters = np.unique(user2Cluster)
    clusterCount = clusters.shape[0]

    print("Cluster count:      %d" % clusterCount)

    users, locations = utils.getCheckins()

    # find top 10 location for each cluster
    recommends = findRecommendations(user2Cluster, clusters, users, locations)

    # genereate test checkins
    checkins = [random.randint(0, len(users))
                for _ in range(TEST_CHECKIN_COUNT)]
    checkins = [(users[checkinId],
                 user2Cluster[users[checkinId]],
                 locations[checkinId])
                for checkinId in checkins]

    accuracy = testCheckins(checkins, recommends)
    print("Accuracy [0,1]:     %f%%" % (accuracy * 100))

    print("Done:               %s" % str(datetime.datetime.now()))


main()
