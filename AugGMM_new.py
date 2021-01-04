from pyclustering.utils import euclidean_distance_square
from clustering_final import Clustering
from distance import min_distance, max_distance
#from normalization import normalized_X
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import timeit
from Node import Node


# numberOfCluster = 40
# numberOfLevels = 1



def AugGMM(cluster, X, K,indexMap,L):

    global Auggmm_step1_time

    l = 1
    maxdis = 0
    selectedNode1 = None
    selectedNode2 = None

    start = timeit.default_timer()

    for node1 in cluster.root.children:
        for node2 in cluster.root.children:
            distmin , distmax = cluster.dismatrix[l][node1.id][node2.id]
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
    a= [ 0.99067444,  4.44921468]
    b = [ 1.05237385 , 4.31595483]

    # selectedNode1.elements.remove(a)
    # selectedNode2.elements.remove(b)
    C.append(a)
    C.append(b)
    X.remove(a)
    X.remove(b)
    CCluster.append(selectedNode1)
    CCluster.append(selectedNode2)

    stop = timeit.default_timer()



    Auggmm_step1_time = stop - start
    print('Time for AugGMM step 1: ', Auggmm_step1_time)

    # print(a)
    # print(b)
    # print(C)

    for k in range(K - 2):

        RemainItems = getNext(cluster,C,indexMap,L)
        V = []
        for i in RemainItems:
            min = 10000000
            for j in C:
                dist = euclidean_distance_square(i, j)
                if min > dist:
                    min = dist
            V.append(min)

        # print(maxOfmins)
        index_max = np.argmax(V)
        # print(RemainItems[index_max])

        C.append(RemainItems[index_max])
        X.remove(RemainItems[index_max])
        id = cluster.documentMap[tuple(RemainItems[index_max])][1]
        node = cluster.root.children[id - 1]
        node.elements.remove(RemainItems[index_max])

    print("Aug-GMM result:", C)
    return C


def getNext(cluster,C,indexMap,L):
    LLmin = []
    LLmax = []
    clusterArray = cluster.root.children
    for l in range(1,L+1):
        for node1 in clusterArray:
            # print("children ", node1.elements)
            minmax = 10000000
            minmin = 10000000
            for e in C:

                ###########cluster to cluster distance##########

                id = cluster.documentMap[tuple(e)][l]
                distmin, distmax = cluster.dismatrix[l][id][node1.id]

                ######### item to cluster distance ############

                # id = indexMap[tuple(e)]
                # distmin , distmax = cluster.dismatrixitem[l][id][node1.id]

                if minmax > distmax:
                    minmax = distmax
                if minmin > distmin:
                    minmin = distmin

            LLmin.append(minmin)
            LLmax.append(minmax)

        maxofMin = max(LLmin)
        remainCluster = []
        i = 0
        if l == L and L == 1:
            for it in LLmax:
                if it >= maxofMin:
                    remainCluster.append(clusterArray[i])
                i = i + 1
            clusterArray = remainCluster
        elif l < L :
            for it in LLmax:
                if it >= maxofMin:
                    remainCluster.extend(clusterArray[i].children)
                i = i + 1
            clusterArray = remainCluster
    remainItems = []
    for node in clusterArray:
        remainItems.extend(node.elements)
    print("number of returned items by get next: ", len(remainItems))
    return remainItems


def GMM(X, K):

    global gmm_step1_time
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

    a = [0.99067444, 4.44921468]
    b = [1.05237385, 4.31595483]

    C.append(a)
    C.append(b)

    X.remove(a)
    X.remove(b)
    stop = timeit.default_timer()
    gmm_step1_time = stop - start
    print('Time for gmm step 1: ', gmm_step1_time)

    # print(a)
    # print(b)
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
        print(L)
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


def run(numberofSample, numberofCluster, numberofLevel, Kvalue):
    f = open('MMRoutput.txt', 'a')
    print('dataset size: ', numberofSample, 'k:', Kvalue, 'number of cluster: ',
          numberofCluster, 'number of level: ', numberofLevel, file=f)
    print('dataset size: ', numberofSample, 'k:', Kvalue, 'number of cluster: ',
          numberofCluster, 'number of level: ', numberofLevel)

    Xin, Y = make_blobs(n_samples=numberofSample, centers=10, cluster_std=0.010, random_state=0)

    # dataset=pd.read_csv('business.csv' , nrows=numberofSample)
    # dataset=pd.read_csv('ratings.csv' , nrows=numberofSample)

    # X = dataset.iloc[:, [2,3]].values
    # X = dataset.iloc[:, [6,7,8]].values

    # normalized_X = normalize(X, axis=0, norm='l2')*1000000
    # normalized_X = normalize(X, axis=0, norm='l2')*1000

    # X = normalized_X

    query = [3, 5]
    #
    # X = [[0.99067444, 4.44921468],
    #      [1.05237385, 4.31595483],
    #      [2.08657429, 0.81225409],
    #      [0.96594819, 4.34484718],
    #      [-1.44046039, 2.84366576],
    #      [-1.29992855, 2.77244569],
    #      [-1.78220299, 2.98324412],
    #      [2.20467543, 0.87714783],
    #      [2.09965384, 0.93103109],
    #      [1.07127892, 4.28865161]]
    # X = np.array(X)

    X = Xin
    indexMap = {}
    index = 0
    for e in X:
        indexMap[tuple(e)] = index
        index = index + 1

    start = timeit.default_timer()
    cluster = Clustering(X.tolist(), numberofCluster, numberofLevel)
    cluster.buildTree(cluster.root)
    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrix(numberofCluster, numberofLevel)
    cluster.createDistanceMatrixforelements(numberofCluster, numberofLevel)

    stop = timeit.default_timer()

    print('Time for indexing: ', stop - start)

    Xgmm = X.tolist()
    start = timeit.default_timer()

    gmmResult = GMM(Xgmm, Kvalue)
    print("gmm", gmmResult)
    # GMM(Xgmm, Kvalue)
    stop = timeit.default_timer()

    gmm_time = stop - start
    gmm_final = gmm_time - gmm_step1_time

    print('Time for gmm: ',gmm_final , file=f)
    print('Time for gmm: ', gmm_final)

    Xgmm = X.tolist()
    start = timeit.default_timer()

    augGmmResult = AugGMM(cluster, Xgmm, Kvalue,indexMap,1)
    print("aug", augGmmResult)

    stop = timeit.default_timer()

    auggmm_time = stop - start

    auggmm_final = auggmm_time - Auggmm_step1_time

    print('Time for aug-gmm: ',auggmm_final , file=f)
    print('Time for aug-gmm: ', auggmm_final)

    checkResult(augGmmResult, gmmResult)

def main():

    run(5000, 50, 1, 15)

main()



