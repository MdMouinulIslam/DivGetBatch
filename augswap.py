from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
from clustering import  Clustering
import heapq

from distance import min_distance, max_distance
import numpy as np
import timeit
from normalization import normalized_X

numberOfCluster = 50
numberOfLevels = 1



def createSimMatrix(q,X):
    r = {}
    i = 0
    for p in X:
        d = (q[0] - p[0])**2 + (q[1] - p[1])**2
        r[i] = d
        i = i + 1
    return  r

def createDisMatrix(X):
    d = []
    i = 0
    for p1 in X:
        dd = []
        for p2 in X:
            dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            dd.append(dist)
        d.append(dd)
    return d


def topkitems(sortedrecitems,k):
    return sortedrecitems[0:k]

def calculateDiRetlist(distanceMatrix, retlistkeys,i):
    diret = 0
    for j in retlistkeys:
        diret = diret + distanceMatrix[i][j]
    return diret

SortedRecItems = []

def swap(recitems, distanceMatrix, k, ub):
    global SortedRecItems
    SortedRecItems = sorted(recitems.items(), key=lambda x: x[1],reverse=True)
    retlist = topkitems(SortedRecItems, k)
    pos = k + 1

    retlistkeys, retlistvalues = zip(*retlist)
    retlistkeys = list(retlistkeys)
    diretlist = []
    diretlistMap = {}



    for i in retlistkeys:
        diretval = calculateDiRetlist(distanceMatrix,retlistkeys,i)
        #print("item ", i, " di = ", diretval )
        t = (diretval,i)
        diretlist.append(t)
        diretlistMap[i] = diretval



    M = []
    for item in diretlist:
        heapq.heappush(M, item)

    #print(M)

    i = heapq.heappop(M)
    #print(i)
    #((recitems[i[1]] - SortedRecItems[pos][1]))
    while ((recitems[i[1]] - SortedRecItems[pos][1] ) < ub):
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)
        dsortedrec = calculateDiRetlist(distanceMatrix,retlistkeys,SortedRecItems[pos][0])
        if (diretlistMap[i[1]] < dsortedrec):
            retlist.append(SortedRecItems[pos])
            retlistkeys.append(SortedRecItems[pos][0])
            heapq.heappush(M, (dsortedrec,SortedRecItems[pos][0]))
            #update d values

            for i in retlistkeys:
                diretval = calculateDiRetlist(distanceMatrix, retlistkeys, i)
                #print("item ", i, " di = ", diretval)
                t = (diretval, i)
                diretlist.append(t)
                diretlistMap[i] = diretval
            i = heapq.heappop(M)
            #print(i)

        else:
            retlist.append(rlist[0])
        pos = pos + 1
        if( pos == len(SortedRecItems)):
            break
    return retlist

skip = 0
def augSwap(cluster,X,recitems, distanceMatrix, k, ub):
    global skip
    global SortedRecItems
    SortedRecItems = sorted(recitems.items(), key=lambda x: x[1],reverse=True)
    retlist = topkitems(SortedRecItems, k)
    pos = k + 1

    retlistkeys, retlistvalues = zip(*retlist)
    retlistkeys = list(retlistkeys)
    diretlist = []
    diretlistMap = {}



    for i in retlistkeys:
        diretval = calculateDiRetlist(distanceMatrix,retlistkeys,i)
        #print("item ", i, " di = ", diretval )
        t = (diretval,i)
        diretlist.append(t)
        diretlistMap[i] = diretval



    M = []
    for item in diretlist:
        heapq.heappush(M, item)

    #print(M)

    i = heapq.heappop(M)
    #print(i)
    #((recitems[i[1]] - SortedRecItems[pos][1]))
    skipArray = []
    while pos < len(recitems) and (recitems[i[1]] - SortedRecItems[pos][1] ) < ub:
        it = SortedRecItems[pos][0]
        #print(type(skipArray))
        if X[it] in skipArray:
            #print("ok")
            skip = skip + 1
            pos = pos + 1
            continue

        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)


        arr = getNext(cluster, i, it, diretlistMap, retlistkeys, X)



        if len(arr) != 0:
            skipArray = arr
            retlist.append(rlist[0])
            pos = pos + 1
            continue

        dsortedrec = calculateDiRetlist(distanceMatrix,retlistkeys,SortedRecItems[pos][0])
        if (diretlistMap[i[1]] < dsortedrec):
            retlist.append(SortedRecItems[pos])
            retlistkeys.append(SortedRecItems[pos][0])
            heapq.heappush(M, (dsortedrec,SortedRecItems[pos][0]))
            #update d values

            for i in retlistkeys:
                diretval = calculateDiRetlist(distanceMatrix, retlistkeys, i)
                #print("item ", i, " di = ", diretval)
                t = (diretval, i)
                diretlist.append(t)
                diretlistMap[i] = diretval
            i = heapq.heappop(M)
            skipArray=[]
            #print(i)

        else:
            retlist.append(rlist[0])
        pos = pos + 1
        if( pos == len(SortedRecItems)):
            break
    return retlist



def getNext(cluster,i,it, diretlistMap,retlistkeys,X):

    id = cluster.documentMap[tuple(X[it])][1]
    #print(id)
    maxdis, mindis  = calculateDiRetlistCluster(cluster, X, retlistkeys, 1, id)
    if maxdis <i[0]:
        #print("prune")
        return cluster.root.children[id-1].elements
    elif mindis > i[1]:
        #print("return top")
        return []
    else:
        #print("return")
        return  []

    return retlist





def augSwapNew(cluster, XX, recitems, distanceMatrix, k, ub):
    global SortedRecItems
    SortedRecItems = sorted(recitems.items(), key=lambda x: x[1], reverse=True)
    retlist = topkitems(SortedRecItems, k)
    pos = k + 1

    retlistkeys, retlistvalues = zip(*retlist)
    retlistkeys = list(retlistkeys)
    diretlist = []
    diretlistMap = {}



    for i in retlistkeys:
        diretval = calculateDiRetlist(distanceMatrix,retlistkeys,i)
        #print("item ", i, " di = ", diretval )
        t = (diretval,i)
        diretlist.append(t)
        diretlistMap[i] = diretval


    M = []
    for item in diretlist:
        heapq.heappush(M, item)

    #print(M)

    i = heapq.heappop(M)
    #print(i)
    #((recitems[i[1]] - SortedRecItems[pos][1]))
    while ((recitems[i[1]] - SortedRecItems[pos][1] ) < ub):
        #cluster.documentMap[tuple(d)][1]
        rlist = [item for item in retlist if item[0] == i[1]]
        retlist.remove(rlist[0])
        retlistkeys, retlistvalues = zip(*retlist)
        retlistkeys = list(retlistkeys)


        item = 0,0
        for node in cluster.root.children:
            mindis, maxdis = calculateDiRetlistCluster(cluster, XX, retlistkeys, 1, node.id)
            #print("mindis, maxdis : ", mindis, maxdis)
            if mindis < diretlistMap[i[1]]:
                print("prune")
                cluster.root.children.remove(node)
            elif maxdis > diretlistMap[i[1]]:
                print("return top")
                item = list(node.elements.pop(0))
                pos = [index for index in range(len(XX)) if (XX[index] == item).any()][0]
                if len(node.elements) == 0:
                    cluster.root.children.remove(node)
                #pos = XX.index(item)
                break
            else:
                print("inside else")
                item = list(node.elements.pop(0))
                pos  = [index for index in range(len(XX)) if (XX[index] == item).any()][0]#XX.index(item)
                if len(node.elements) == 0:
                    cluster.root.children.remove(node)
                break

        dsortedrec = calculateDiRetlist(distanceMatrix, retlistkeys, SortedRecItems[pos][0])

        if (diretlistMap[i[1]] < dsortedrec):
            retlist.append(SortedRecItems[pos])
            retlistkeys.append(SortedRecItems[pos][0])
            heapq.heappush(M, (dsortedrec,SortedRecItems[pos][0]))
            #update d values

            for i in retlistkeys:
                diretval = calculateDiRetlist(distanceMatrix, retlistkeys, i)
                #print("item ", i, " di = ", diretval)


                t = (diretval, i)
                diretlist.append(t)
                diretlistMap[i] = diretval
            i = heapq.heappop(M)
            #print(i)

        else:
            retlist.append(rlist[0])
        pos = pos + 1
        if( pos == len(SortedRecItems)):
            break
    return retlist




def calculateDiRetlistCluster(cluster,XX,retlistkeys,l,i):
    maxdf,mindf  = (0,0)

    for j in retlistkeys:
        id  = cluster.documentMap[tuple(XX[j])][l]
        maxd,mind =  cluster.dismatrix[l][i][id]
        mindf = mindf + mind
        maxdf = maxdf + maxd
    return (maxdf,mindf)



X, Y = make_blobs(n_samples=1000, centers=300, n_features=2, random_state=1)#
#X,Y = make_blobs(n_samples=35, centers=10, cluster_std=0.80, random_state=0)
X = X.tolist()

Xin = normalized_X
X = Xin.tolist()

#print(type(X))
query = [0, 0]

r = createSimMatrix(query,X)
d = createDisMatrix(X)

r = createSimMatrix(query,X)
d = createDisMatrix(X)

k = 10

start = timeit.default_timer()

res = swap(r, d, k, 10000000000)

stop = timeit.default_timer()
print('Time for calculate swap: ', stop - start)



print("results swap: ")
for rs in res:
    print(X[rs[0]])



# XX = []
# for item in SortedRecItems:
#     XX.append(X[item[0]])
# #print(XX)

cluster = Clustering(X)
cluster.buildTree(cluster.root)

cluster.createLevelMatrix(cluster.root)
cluster.createDistanceMatrix(numberOfCluster, numberOfLevels)


"""""
def printcluster(cluster):
    for node in cluster.root.children:
        print("node elements : ", node.elements)


printcluster(cluster)
        
"""""


start = timeit.default_timer()

res = augSwap(cluster,X,r, d, k, 10000000000)

stop = timeit.default_timer()
print('Time for calculate Aug swap: ', stop - start)


print("results Augswap: ")
for rs in res:
    print(X[rs[0]])


print("skip = ",skip)