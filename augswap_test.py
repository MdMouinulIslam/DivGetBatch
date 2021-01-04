from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
from clustering_final import Clustering
import heapq
from distance import min_distance, max_distance
import numpy as np
import timeit
# from normalization import normalized_X
import pandas as pd
from sklearn import preprocessing

from sklearn.preprocessing import normalize

inArrayTime = 0.0

def createSimMatrix(q, X):
    r = {}
    i = 0
    for p in X:
        d = euclidean_distance_square(q, p)
        r[i] = 1 / (1 + d)
        i = i + 1
    return r


def createDisMatrix(X):
    d = []
    i = 0
    for p1 in X:
        dd = []
        for p2 in X:
            dist = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            dd.append(dist)
        d.append(dd)
    return d


def topkitems(sortedrecitems, k):
    return sortedrecitems[0:k]


def calculateDiRetlist(X, retlistkeys, i):
    diret = 0
    for j in retlistkeys:
        diret = diret + euclidean_distance_square(X[i], X[j])
    return diret


SortedRecItems = []

swapnumber = 0


def swap(X,SortedRecItems, recitems,k, ub):
    global swapnumber,inArrayTime
    #SortedRecItems = sorted(recitems.items(), key=lambda x: x[1], reverse=True)
    retlist = topkitems(SortedRecItems, k)
    pos = k

    retlistkeys, retlistvalues = zip(*retlist)
    retlistkeys = list(retlistkeys)
    diretlist = []
    diretlistMap = {}
    for i in retlistkeys:
        diretval = calculateDiRetlist(X, retlistkeys, i)
        t = (diretval, i)
        diretlist.append(t)
        diretlistMap[i] = diretval

    M = []
    for item in diretlist:
        heapq.heappush(M, item)

    # print(M)

    i = heapq.heappop(M)
    # print(i)

    while ((recitems[i[1]] - SortedRecItems[pos][1]) < ub):
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)
        dsortedrec = calculateDiRetlist(X, retlistkeys, SortedRecItems[pos][0])
        start = timeit.default_timer()
        if (i[0] < dsortedrec):
            swapnumber = swapnumber + 1
            retlist.append(SortedRecItems[pos])
            retlistkeys.append(SortedRecItems[pos][0])
            heapq.heappush(M, (dsortedrec, SortedRecItems[pos][0]))
            # update d values

            diretlist = []
            diretlistMap = {}
            M = []
            for j in retlistkeys:
                diretval = calculateDiRetlist(X, retlistkeys, j)
                # print("item ", i, " di = ", diretval)
                t = (diretval, j)
                diretlist.append(t)
                diretlistMap[j] = diretval
                heapq.heappush(M, t)

            i = heapq.heappop(M)
            # print(i)

        else:
            retlist.append(rlist[0])
        pos = pos + 1
        end = timeit.default_timer()
        inArrayTime = inArrayTime + end - start

        if (pos == len(SortedRecItems)):
            break
    print("swap ", swapnumber)
    swapnumber = 0
    return retlist


def calculateDiRetlistCluster_new(cluster, l, minmaxDitc, insid, delid, id):
    maxdf, mindf = minmaxDitc[id]
    minin, maxin = cluster.dismatrixitem[l][insid][id]
    minddel, maxdel = cluster.dismatrixitem[l][delid][id]
    maxdf = maxdf + maxin - maxdel
    mindf = mindf + minin - minddel
    minmaxDitc[id] = (maxdf, mindf)
    return (maxdf, mindf)


def calculateDiRetlistCluster(cluster, l, retlistkeys, id):
    maxdf, mindf = (0.0, 0.0)

    for j in retlistkeys:
        mind, maxd = cluster.dismatrixitem[l][j][id]
        mindf = mindf + mind
        maxdf = maxdf + maxd
    return (maxdf, mindf)


def getNext(cluster, minmaxDitc, i, retlistkeys, itdel, itin):
    skipElements = {-1}
    if itin is None or itdel is None:
        for node in cluster.root.children:
            maxdis, mindis = calculateDiRetlistCluster(cluster, 1, retlistkeys, node.id)
            minmaxDitc[node.id] = (maxdis, mindis)
            if maxdis < i[0]:
                skipElements.add(node.id)
    else:
        for node in nodeArray:
            maxdis, mindis = calculateDiRetlistCluster_new(cluster, 1, minmaxDitc, itin, itdel, node)
            if maxdis < i[0]:
                # print("prune")
                skipElements.add(node)
    return skipElements


skip = 0

nodeArray = []

def AugSwap(X, cluster, SortedRecItems,recitems,s,k, ub):
    global swapnumber, skip,nodeArray,inArrayTime
    for i in range(1,len(cluster.root.children) +1):
        nodeArray.append(i)
    getNextTime = 0.0
    #SortedRecItems = sorted(recitems.items(), key=lambda x: x[1], reverse=True)
    retlist = topkitems(SortedRecItems, k)
    pos = k

    retlistkeys, retlistvalues = zip(*retlist)
    retlistkeys = list(retlistkeys)
    diretlist = []
    diretlistMap = {}
    for i in retlistkeys:
        diretval = calculateDiRetlist(X, retlistkeys, i)
        # print("item ", i, " di = ", diretval )
        t = (diretval, i)
        diretlist.append(t)
        diretlistMap[i] = diretval

    M = []
    for item in diretlist:
        heapq.heappush(M, item)

    # print(M)

    i = heapq.heappop(M)
    # print(i)

    recalSkipList = True
    firstTime = True
    skipList = {}
    minmaxdic = {}
    newitem = None
    preNode = -1
    inArrayTime = 0.0
    swapSet = {-1}
    while ((recitems[i[1]] - SortedRecItems[pos][1]) < ub):
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)

        # nodeId = cluster.documentMap[tuple(X[SortedRecItems[pos][0]])][1]
        # # print("nodeid = ",nodeId)
        # if preNode != nodeId and preNode != -1:
        #     nodeArray.remove(nodeId)
        # preNode = nodeId


        if recalSkipList and swapnumber > len(X)/100*s:
            start = timeit.default_timer()
            if firstTime:
                firstTime = False
                # print("skip number = ",skip)
                # skip = 0
                getNext(cluster, minmaxdic, i, retlistkeys, None, None)

            else:
                skipList = getNext(cluster, minmaxdic, i, retlistkeys, i[1], newitem[0])
                # print("skip number = ", skip)
                # skip = 0
            stop = timeit.default_timer()
            getNextTime = getNextTime + stop - start


        if cluster.documentMap[tuple(X[SortedRecItems[pos][0]])][1] not in skipList:
            # print("skiplist = ",skipList)
            # print("item =", cluster.documentMap[tuple(X[SortedRecItems[pos][0]])][1])
            dsortedrec = calculateDiRetlist(X, retlistkeys, SortedRecItems[pos][0])
        else:
            skip = skip + 1
            pos = pos + 1
            recalSkipList = False
            retlist.append(rlist[0])

            if (pos == len(SortedRecItems)):
                break
            continue

        start = timeit.default_timer()

        if (i[0] < dsortedrec):
            resClsid = cluster.documentMap[tuple(X[SortedRecItems[pos][0]])][1]
            swapSet.add(resClsid)
            #print(resClsid)
            recalSkipList = True
            swapnumber = swapnumber + 1
            retlist.append(SortedRecItems[pos])
            newitem = SortedRecItems[pos]
            retlistkeys.append(SortedRecItems[pos][0])

            minval = 1000000000
            candItem = None
            for j in retlistkeys:
                diretval = calculateDiRetlist(X, retlistkeys, j)
                t = (diretval, j)
                if minval > diretval:
                    minval = diretval
                    candItem = t

            i = candItem

        else:
            recalSkipList = False
            retlist.append(rlist[0])
        pos = pos + 1
        end = timeit.default_timer()
        inArrayTime = inArrayTime + end - start

        if (pos == len(SortedRecItems)):
            break
    print("awg swap number ", swapnumber)
    print("skip ", skip)
    print("getNext Time = ", getNextTime)
    print(len(swapSet))
    return retlist


def checkResult(augGmmResult, gmmResult):
    if sorted(augGmmResult) == sorted(gmmResult):
        print("array equal")
    else:
        print("array Not equal")


def run(datasetname, numberOfSamples, numberofCluster, numberofLevel, k):
    global inArrayTime
    print('datasetname', datasetname, 'dataset size: ', numberOfSamples, 'k:', k, 'number of cluster: ',
          numberofCluster, 'number of level: ', numberofLevel)

    # X, Y = make_blobs(n_samples=numberOfSamples, centers=1, n_features=3, random_state=2)

    # dataset=pd.read_csv('business.csv' , nrows=numberOfSamples)
    dataset = pd.read_csv('ratings.csv', nrows=numberOfSamples)

    X = dataset.iloc[:, [2, 3]].values
    # X = dataset.iloc[:, [6,7,8]].values

    # normalized_X = normalize(X, axis=0, norm='l2')
    normalized_X = normalize(X, axis=0, norm='l2') * 1000

    X = normalized_X

    X = X.tolist()

    indexMap = {}
    index = 0
    for e in X:
        indexMap[tuple(e)] = index
        index = index + 1

    cluster = Clustering(X, numberofCluster, numberofLevel)
    cluster.buildTree(cluster.root)
    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrixforelements(numberofCluster, numberofLevel)



    query = [0, 0]
    r = createSimMatrix(query, X)
    ub = 1000000
    SortedRecItems = sorted(r.items(), key=lambda x: x[1], reverse=True)

    start = timeit.default_timer()
    res = swap(X, SortedRecItems,r, k, ub)
    stop = timeit.default_timer()
    print('Time for calculate swap: ', stop - start)

    print("swap: ", res)

    print(inArrayTime)
    inArrayTime = 0.0

    SortedRecItems = sorted(r.items(), key=lambda x: x[1], reverse=True)
    start = timeit.default_timer()
    augres = AugSwap(X, cluster,SortedRecItems, r,0, k, ub)
    stop = timeit.default_timer()
    print('Time for calculate Aug swap: ', stop - start)

    print("AugSwap: ", augres)
    print(inArrayTime)
    checkResult(augres, res)


run('movielens', 10000, 250, 1, 20)