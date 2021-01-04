from pyclustering.utils import euclidean_distance_square
from clustering import Clustering
from distance import min_distance, max_distance
from normalization import normalized_X
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import timeit

from Node import Node

numberOfCluster = 20
numberOfLevels = 1


def AugGMM(cluster, X1, indexMap, K, C1):

    
    l = 1
    
    for k in range(K - 2):
        LLmin = []
        LLmax = []
        for node1 in cluster.root.children:
            # print("children ", node1.elements)
            minmax = 10000000
            minmin = 10000000
            for e in C1:
                id = cluster.documentMap[tuple(e)][l]
                distmax, distmin = cluster.dismatrix[l][id][node1.id]
                #cluster.dismatrixitem[l][indexMap[tuple(e)]][node1.id]

                if minmax > distmax[0]:
                    minmax = distmax[0]
                if minmin > distmin[0]:
                    minmin = distmin[0]

            LLmin.append(minmin)
            LLmax.append(minmax)

        maxofMin = max(LLmin)
        selecteditem = []
        i = 0
        for it in LLmax:
            if it > maxofMin:
                selecteditem = selecteditem + cluster.root.children[i].elements
            i = i + 1

        L = []
        for i in selecteditem:
            min = 10000000
            for j in C1:
                dist = euclidean_distance_square(i, j)
                if min > dist:
                    min = dist
            L.append(min)

        # print(maxOfmins)
        index_max = np.argmax(L)
        # print(selecteditem[index_max])

        C1.append(selecteditem[index_max])
        X1.remove(selecteditem[index_max])
        id = cluster.documentMap[tuple(selecteditem[index_max])][1]
        node = cluster.root.children[id - 1]
        node.elements.remove(selecteditem[index_max])

    print("Aug-GMM result:", C1)
    return C1


def GMM(X, K, C):
    for k in range(K - 2):
        L = []
        for i in X:
            min = 10000000
            for j in C:
                dist = euclidean_distance_square(i, j)
                if min > dist:
                    min = dist
            L.append(min)

        # print(maxOfmins)
        index_max = np.argmax(L)
        # print(L[index_max])
        # print(X[index_max])

        C.append(X[index_max])

        X.remove(X[index_max])
        # print("C:" , C)
        # print("X:", X)

    print("final C:", C)
    return C


def checkResult(augGmmResult, gmmResult):
    if sorted(augGmmResult) == sorted(gmmResult):
        print("array equal")
    else:
        print("array not equal")
        for i in gmmResult:
            if i not in augGmmResult:
                print(i, " not in Aug GMM")

        for i in augGmmResult:
            if i not in gmmResult:
                print(i, " not in GMM")


def main():
    # Xin, Y = make_blobs(n_samples=1000, centers=10, cluster_std=0.60, random_state=0)
    # np.random.seed(46)
    # Xin = np.random.randint(10000, size=(20, 2))
    # print(Xin)

    Xin = normalized_X

    X = Xin.tolist()
    # print(X)
    K = 15

    a = []
    b = []
    C = []

    maxd = 0

    for i in X:
        for j in X:
            if (i[0] == j[0] and i[1] == j[1]) == False:
                dis = euclidean_distance_square(i, j)
                if maxd < dis:
                    maxd = dis
                    a = i
                    b = j

    # print(a,b,max)

    C.append(a)
    C.append(b)

    X.remove(a)
    X.remove(b)

    start = timeit.default_timer()

    gmmResult = GMM(X, K, C)

    stop = timeit.default_timer()

    print('Time for gmm: ', stop - start)

    X1 = Xin.tolist()

    a1 = []
    b1 = []
    C1 = []

    maxd = 0

    for i in X1:
        for j in X1:
            if (i[0] == j[0] and i[1] == j[1]) == False:
                dis = euclidean_distance_square(i, j)
                if maxd < dis:
                    maxd = dis
                    a1 = i
                    b1 = j

    # print(a,b,max)

    C1.append(a1)
    C1.append(b1)

    X1.remove(a1)
    X1.remove(b1)

    indexMap = {}
    index = 0
    for e in X:
        indexMap[tuple(e)] = index
        index = index + 1

    cluster = Clustering(X1)
    cluster.buildTree(cluster.root)
    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrix(numberOfCluster, numberOfLevels)
    #cluster.createDistanceMatrixforelements(numberOfCluster, numberOfLevels)

    start = timeit.default_timer()

    augGmmResult = AugGMM(cluster, X1, indexMap, K, C1)

    stop = timeit.default_timer()
    print('Time for aug-gmm: ', stop - start)

    checkResult(augGmmResult, gmmResult)


main()
