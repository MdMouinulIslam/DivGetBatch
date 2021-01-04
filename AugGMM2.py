from pyclustering.utils import euclidean_distance_square
# from sklearn.datasets.samples_generator import make_blobs
from clustering2 import Clustering
from distance import min_distance, max_distance
import numpy as np
import timeit
from normalization import normalized_X


from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import timeit

from Node import Node

numberOfCluster = 50
numberOfLevels = 1

gmmtimeStep1  = 0.0
augtimeStep1 = 0.0

def AugGMM(cluster, X, indexMap, K):
    global augtimeStep1
    l = 1
    maxdis = 0
    selectedNode1 = None
    selectedNode2 = None

    start = timeit.default_timer()

    for node1 in cluster.root.children:
        for node2 in cluster.root.children:
            distmax,distmin = cluster.dismatrix[l][node1.id][node2.id]
            if maxdis < distmax:
                maxdis = distmax
                selectedNode1 = node1
                selectedNode2 = node2

    XX = selectedNode1.elements + selectedNode2.elements
    C = []
    CCluster = []
    maxd = 0
    for i in XX:
        for j in XX:
            if i != j:
                dis = euclidean_distance_square(i, j)
                if maxd < dis:
                    maxd = dis
                    a = i
                    b = j

    # print(a,b,max)
    selectedNode1.elements.remove(a)
    selectedNode2.elements.remove(b)
    C.append(a)
    C.append(b)
    X.remove(a)
    X.remove(b)
    CCluster.append(selectedNode1)
    CCluster.append(selectedNode2)

    stop = timeit.default_timer()
    augtimeStep1 = stop - start
    #print('augGMM Time for step 1: ', stop - start)

    #print(a)
    #print(b)
    # print(C)

    for k in range(K - 2):
        l = 0
        clusters  = cluster.root.children
        while(l < numberOfLevels):
            l = l + 1
            LLmin = []
            LLmax = []
            for node1 in clusters:
                #print("children ", node1.elements)
                minmax = 10000000
                minmin = 10000000
                for e in C:
                    id = cluster.documentMap[tuple(e)][l]
                    distmax,distmin = cluster.dismatrix[l][id][node1.id] #cluster.dismatrix[l][indexMap[tuple(e)]][node1.id]
                    if minmax > distmax:
                        minmax = distmax
                    if minmin > distmin:
                        minmin = distmin

                LLmin.append(minmin)
                LLmax.append(minmax)


            maxofMin = max(LLmin)
            i = 0
            clusterTocheck = []
            for it in LLmax:
                if it > maxofMin:
                    if len(clusters[i].children) == 0:
                        clusterTocheck.append(clusters[i])
                    else:
                        clusterTocheck.extend(clusters[i].children)
                i = i + 1
            clusters = clusterTocheck
        selecteditem = []
        i = 0
        # for it in LLmax:
        #     if it > maxofMin:
        #         selecteditem = selecteditem + cluster.root.children[i].elements
        #     i = i + 1
        for c in clusters:
            selecteditem = selecteditem + c.elements
        L = []
        if len(selecteditem) == 0:
            selecteditem = X
            print("please reduce number of cluster value")
        for i in selecteditem:
            min = 10000000

            for j in C:
                dist = euclidean_distance_square(i, j)
                if min > dist:
                    min = dist
            L.append(min)

        # print(maxOfmins)
        index_max = np.argmax(L)
        #print(selecteditem[index_max])


        for l in range(1,numberOfLevels+1):
            id = cluster.documentMap[tuple(selecteditem[index_max])][l]
            node = cluster.levelMatrix[l][id]
            node.elements.remove(selecteditem[index_max])

        C.append(selecteditem[index_max])
        #X.remove(selecteditem[index_max])


    print("Aug-GMM result:", C)
    return C


def GMM(X, K):
    global gmmtimeStep1
    a = []
    b = []
    C = []
    maxd = 0


    start = timeit.default_timer()

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

    stop = timeit.default_timer()
    gmmtimeStep1 = stop - start
    #print('GMM Time for step 1: ', stop - start)

    #print(a)
    #print(b)
    # print(X)
    # print(C)

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
        #print(L[index_max])
        #print(X[index_max])

        C.append(X[index_max])

        X.remove(X[index_max])
        # print("C:" , C)
        # print("X:", X)

    print("final C:", C)
    return C


def GMM(X, K):
    global gmmtimeStep1
    a = []
    b = []
    C = []
    maxd = 0


    start = timeit.default_timer()

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

    stop = timeit.default_timer()
    gmmtimeStep1 = stop - start


    for k in range(K - 2):
        L = []
        for i in X:
            min = 10000000
            for j in C:
                dist = euclidean_distance_square(i, j)
                if min > dist:
                    min = dist
            L.append(min)

        index_max = np.argmax(L)
        C.append(X[index_max])
        X.remove(X[index_max])


    print("final C:", C)
    return C

def checkResult(augGmmResult, gmmResult):
    if sorted(augGmmResult) == sorted(gmmResult):
        print("array equal")
    else:
        print("array not equal")
        for i in gmmResult:
            if i not in augGmmResult:
                print(i," not in Aug GMM")

        for i in augGmmResult:
            if i not in gmmResult:
                print(i, " not in GMM")


def run(numberofsample,numberofCluster, numberofLevel,K):
    Xin, Y = make_blobs(n_samples=numberofsample, centers=50, cluster_std=0.80, random_state=0)
    #import matplotlib.pyplot as plt
    #plt.scatter(Xin[:, 0], Xin[:, 1], s=50);
    #plt.show()


    #np.random.seed(46)
    #Xin = np.random.randint(10000, size=(20, 2))
    #print(Xin)
    #Xin = normalized_X
    X = Xin.tolist()
    #print(X)

    start = timeit.default_timer()

    gmmResult = GMM(X, K)

    stop = timeit.default_timer()

    print('Time for gmm: ', stop - start - gmmtimeStep1)


    X = Xin.tolist()
    indexMap = {}
    index = 0
    for e in X:
        indexMap[tuple(e)] = index
        index = index + 1

    cluster = Clustering(X, numberofCluster, numberofLevel)
    cluster.buildTree(cluster.root)
    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrix(numberofCluster, numberofLevel)
    #cluster.createDistanceMatrix(numberOfCluster, numberOfLevels)

    start = timeit.default_timer()

    augGmmResult = AugGMM(cluster, X, indexMap, K)

    stop = timeit.default_timer()
    print('Time for aug-gmm: ', stop - start  - augtimeStep1)

    checkResult(augGmmResult, gmmResult)



run(10000,200,1,15)
