from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
from clustering_final import Clustering
import heapq
from distance import min_distance, max_distance
import numpy as np
import timeit
#from normalization import normalized_X
from sklearn import preprocessing

from sklearn.preprocessing import normalize



def createSimMatrix(q, X):
    r = {}
    i = 0
    for p in X:
        d = euclidean_distance_square(q,p)
        r[i] = 1/(1+d)
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

def swap(X,recitems,k,ub):
    global swapnumber
    SortedRecItems = sorted(recitems.items(), key=lambda x: x[1], reverse=True)
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
    #print(i)

    while ((recitems[i[1]] - SortedRecItems[pos][1]) < ub):
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)
        dsortedrec = calculateDiRetlist(X, retlistkeys, SortedRecItems[pos][0])
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
        if (pos == len(SortedRecItems)):
            break
    print("swap ",swapnumber)
    swapnumber = 0
    return retlist



def calculateDiRetlistCluster_new(cluster,l,minmaxDitc,insid,delid,id):
    maxdf, mindf = minmaxDitc[(id,l)]
    minin,maxin = cluster.dismatrixitem[l][insid][id]
    minddel,maxdel  = cluster.dismatrixitem[l][delid][id]
    maxdf = maxdf + maxin - maxdel
    mindf = mindf + minin - minddel
    minmaxDitc[(id,l)] = (maxdf, mindf)
    return (maxdf, mindf)



def calculateDiRetlistCluster(cluster,l,retlistkeys,id):
    maxdf, mindf = (0.0, 0.0)

    for j in retlistkeys:
        mind,maxd = cluster.dismatrixitem[l][j][id]
        mindf = mindf + mind
        maxdf = maxdf + maxd
    return (maxdf, mindf)


def getNext(cluster,numberOfLevels,minmaxDitc, i, retlistkeys,itdel,itin):

    ret = []
    retids = {-1}
    if itin is None or itdel is None:
        clusters = cluster.root.children
        l = 1
        while (l <= numberOfLevels):
            clusterTocheck = []

            for node in clusters:
                maxdis, mindis = calculateDiRetlistCluster(cluster, l, retlistkeys, node.id)
                minmaxDitc[(node.id,l)] = (maxdis,mindis)
                if maxdis >= i[0]:
                    if l == numberOfLevels:
                        clusterTocheck.append(node)
                    else:
                        clusterTocheck.extend(node.children)
            clusters = clusterTocheck
            ret = clusters
            l = l + 1

    else:
        clusters = cluster.root.children
        l = 1
        while (l < numberOfLevels):
            clusterTocheck = []

            for node in clusters:
                maxdis, mindis = calculateDiRetlistCluster_new(cluster, l,minmaxDitc,itin,itdel, node.id)
                if maxdis >= i[0]:
                    if l == numberOfLevels:
                        clusterTocheck.append(node)
                    else:
                        clusterTocheck.extend(node.children)
            clusters = clusterTocheck
            ret = clusters
            l = l + 1

    for n in ret:
        retids.add(n.id)
    return retids

skip = 0
def AugSwap(X,cluster,numberOfLevel,recitems,k,ub):
    global swapnumber,skip
    getNextTime = 0.0
    SortedRecItems = sorted(recitems.items(), key=lambda x: x[1], reverse=True)
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
    #print(i)

    recalSkipList = True
    candClsList = {}
    minmaxdic = {}
    newitem = None
    while ((recitems[i[1]] - SortedRecItems[pos][1]) < ub):
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)


        if recalSkipList:
            start = timeit.default_timer()
            if newitem is None:
                candClsList = getNext(cluster, numberOfLevel,minmaxdic, i, retlistkeys, None, None)

            else:
                candClsList = getNext(cluster,numberOfLevel,minmaxdic,i,retlistkeys,i[1],newitem[0])

            stop = timeit.default_timer()
            getNextTime = getNextTime + stop -start



        if cluster.documentMap[tuple(X[SortedRecItems[pos][0]])][numberOfLevel] in candClsList:
            dsortedrec = calculateDiRetlist(X, retlistkeys, SortedRecItems[pos][0])
        else:
            skip = skip + 1
            pos = pos+1
            recalSkipList = False
            retlist.append(rlist[0])
            if (pos == len(SortedRecItems)):
                break
            continue

        if (i[0] < dsortedrec):
            recalSkipList = True
            swapnumber = swapnumber + 1
            retlist.append(SortedRecItems[pos])
            newitem = SortedRecItems[pos]
            retlistkeys.append(SortedRecItems[pos][0])

            minval = 10000000
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
        if (pos == len(SortedRecItems)):
            break
    print("awg swap number ",swapnumber)
    print("skip ",skip)
    print("getNext Time = ",getNextTime)
    return retlist






def checkResult(augGmmResult, gmmResult):
    if sorted(augGmmResult) == sorted(gmmResult):
        print("array equal")
    else:
        print("array Not equal")



def run(numberOfSamples,numberofCluster,numberofLevel,k):
    X, Y = make_blobs(n_samples=numberOfSamples, centers=1, n_features=3, random_state=2)  #
    X = X.tolist()
    query = [0, 0,0]
    r = createSimMatrix(query, X)
    ub = 1000000

    start = timeit.default_timer()
    res = swap(X, r, k, ub)
    stop = timeit.default_timer()
    print('Time for calculate swap: ', stop - start)

    print("swap: ",res)

    indexMap = {}
    index = 0
    for e in X:
        indexMap[tuple(e)] = index
        index = index + 1

    cluster = Clustering(X,numberofCluster,numberofLevel)
    cluster.buildTree(cluster.root)
    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrixforelements(numberofCluster,numberofLevel)

    start = timeit.default_timer()
    augres = AugSwap(X, cluster,numberofLevel, r, k, ub)
    stop = timeit.default_timer()
    print('Time for calculate Aug swap: ', stop - start)

    print("AugSwap: ",augres)

    checkResult(augres,res)

run(10000,5,3,20)

