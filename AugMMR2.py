from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
from clustering_final import Clustering
from distance import min_distance, max_distance
import numpy as np
import timeit

#from normalization import normalized_X


stopcondcoeff = 1.0


def aug_mmr(cluster,indexMap,lambda_score, q, data, k, numberOfCluster,numberOfLevels):

    docs_unranked = data

    docs_selected = []

    checkGetNext = True

    lastDoc = None

    for i in range (k):
        mmr = -100000000
        R = data.tolist()

        if checkGetNext:
            R = getNext(cluster,indexMap,q,lambda_score,lastDoc, numberOfCluster,numberOfLevels)
            if checkGetNext == True and (len(R)>= stopcondcoeff *  len(data)):
                checkGetNext = False
            best1 = [0,0]

        for item in docs_selected:
            if item in R:
                R.remove(item)


        for d in R:
            sim = 0
            for s in docs_selected:
                if euclidean_distance_square(d, s) == 0:
                    continue
                sim_current = 1/(1+euclidean_distance_square(d, s))
                if sim_current > sim:
                    sim = sim_current
                else:
                    continue

            rel = 1/(1+euclidean_distance_square(q, d))
            mmr_current = lambda_score * rel - (1 - lambda_score) * sim

            if mmr_current > mmr:
                mmr = mmr_current
                best1 = d
            else:
                continue

        docs_selected.append(best1)
        lastDoc = best1

    return docs_selected



def getNext(cluster,indexMap,q,lambda_score,lastDoc,numberOfCluster,numberOfLevels):
    simmin = np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1, numberOfLevels + 1),
        dtype=float)
    simmin.fill(100000)

    simmax = np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1, numberOfLevels + 1),
        dtype=float)
    simmax.fill(-1000000)

    min_mmr_min =  np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1, numberOfLevels + 1),
        dtype=float)
    min_mmr_min.fill(100000)
    max_mmr_max =  np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1, numberOfLevels + 1),
        dtype=float)
    max_mmr_max.fill(-1000000)

    queue = []

    for node in cluster.root.children:
        queue.append(node)

    levelClusters = []

    while queue:

        s = queue.pop(0)
        levelClusters.append(s)

        clsid = s.id
        l = s.level

        if lastDoc is not None:
            id = cluster.documentMap[tuple(lastDoc)][l]
            maxcdis, mincdis = cluster.dismatrix[l][id][clsid]

            #id = indexMap[tuple(lastDoc)]
            #maxcdis,mincdis = cluster.dismatrixitem[l][id][clsid]
            sim_current_max = 1 / (1+mincdis)
            sim_current_min = 1 / (1+maxcdis)

            if sim_current_max > simmax[clsid][l]:
                simmax[clsid][l] = sim_current_max
            if sim_current_min < simmin[clsid][l]:
                simmin[clsid][l] = sim_current_min

        maxdis = max_distance([q], s.elements)
        mindis = min_distance([q], s.elements)

        relmax = 1 / (1+mindis)
        relmin = 1 / (1+maxdis)

        if lastDoc is None:
            min_mmr_min[clsid][l] = lambda_score * relmin
            max_mmr_max[clsid][l] = lambda_score * relmax
        else:
            min_mmr_min[clsid][l] = lambda_score * relmin - (1 - lambda_score) * simmax[clsid][l]
            max_mmr_max[clsid][l] = lambda_score * relmax - (1 - lambda_score) * simmin[clsid][l]

        if len(queue) == 0:
            max_min_mmr_min = 0
            for node1 in levelClusters:
                if max_min_mmr_min < min_mmr_min[node1.id][l]:
                    max_min_mmr_min = min_mmr_min[node1.id][l]

            for node2 in levelClusters.copy():
                if max_mmr_max[node2.id][l] < max_min_mmr_min:
                        levelClusters.remove(node2)

            for node in levelClusters:
                for children in node.children:
                    queue.append(children)

            if queue:
                levelClusters.clear()

    R = []
    for c in levelClusters:
        for item in c.elements:
            R.append(item)
    print(len(R))
    return R



def _mmr(lambda_score, q, data, k):

    docs_unranked = data

    docs_selected = []

    best = [0,0]
    for i in range (k):
        mmr = -100000000
        for d in docs_unranked:
            sim = 0
            for s in docs_selected:
                sim_current = 1/(1+euclidean_distance_square(d, s))
                if sim_current > sim:
                    sim = sim_current
                else:
                    continue

            rel = 1/(1+euclidean_distance_square(q, d))
            mmr_current = lambda_score * rel - (1 - lambda_score) * sim

            if mmr_current > mmr:
                mmr = mmr_current
                best = d
            else:
                continue


        docs_selected.append(best)
        docs_unranked.remove(best)

    return docs_selected




def run(numberofSample,numberofCluster,numberofLevel,Kvalue,lambdavalue):

    f = open('MMRoutput.txt', 'a')
    print('dataset size: ', numberofSample, 'k:', Kvalue, 'lambda: ', lambdavalue, 'number of cluster: ',
          numberofCluster, 'number of level: ', numberofLevel, file=f)
    print('dataset size: ', numberofSample, 'k:', Kvalue, 'lambda: ', lambdavalue, 'number of cluster: ',
          numberofCluster, 'number of level: ', numberofLevel)

    X,Y = make_blobs(n_samples=numberofSample, centers=10, cluster_std=0.60, random_state=0)

    #print(X)

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
    #cluster.createDistanceMatrixforelements(numberofCluster, numberofLevel)

    stop = timeit.default_timer()

    print('Time for indexing: ', stop - start)

    query = [2,5]

    Xmmr = X.tolist()
    start = timeit.default_timer()
    print("mmr", _mmr(lambdavalue, query, Xmmr, Kvalue))
    stop = timeit.default_timer()
    print('Time for mmr: ', stop - start,  file=f)
    print('Time for mmr: ', stop - start)



    start = timeit.default_timer()
    print("aug", aug_mmr(cluster, indexMap, lambdavalue, query, X, Kvalue, numberofCluster,numberofLevel))
    stop = timeit.default_timer()
    print('Time for aug mmr: ', stop - start, file=f)
    print('Time for aug mmr: ', stop - start)



'''''

def mmrExample(simMatrix, simQuery, lambda_score,k):
    docs_unranked = []
    for i in range(0,len(simMatrix)):
        docs_unranked.append(i)
    docs_selected = []

    best = [0,0]
    for i in range (k):
        mmr = -100000000
        for d in docs_unranked:
            sim = 0
            for s in docs_selected:
                sim_current = simMatrix[d][s]
                if sim_current > sim:
                    sim = sim_current
                else:
                    continue

            rel = simQuery[d]

            mmr_current = lambda_score * rel - (1 - lambda_score) * sim
            print("iteration = ",i ," document = ", d, " mmr =", mmr_current)
            if mmr_current > mmr:
                mmr = mmr_current
                best = d
            else:
                continue


        docs_selected.append(best)
        docs_unranked.remove(best)
    print(docs_selected)
    return docs_selected

'''''

#def runExample():
    # simMatrix = [[1,  0.11 , 0.23 , 0.86 , 0.25,  0.8 , 0.3,  0.4 , 0.2 , 0.4],
    #              [0.11,  1,  0.29,  0.57 , 0.51 , 0.42 , 0.35 , 0.65,  0.7 , 0.1 ],
    #              [0.23 , 0.29,  1 , 0.02 , 0.36,  0.15 , 0.45 , 0.23 , 0.09 , 0.51],
    #              [0.86,  0.57,  0.02,  1,  0.33,  0.55,  0.05,  0.15,  0.22,  0.31 ],
    #              [0.25,  0.51,  0.36,  0.33,  1,  0.22,  0.48,  0.15,  0.19,  0.34 ],
    #              [0.8,  0.42,  0.15 , 0.55,  0.22,  1,  0.18,  0.29 , 0.34,  0.17],
    #              [0.3 , 0.35 , 0.45  ,0.05 , 0.48 , 0.18 , 1 , 0.14 , 0.21,  0.46 ],
    #              [0.4 , 0.65,  0.23,  0.15 , 0.15,  0.29,  0.14 , 1,  0.39 , 0.15 ],
    #              [0.2 , 0.7,  0.09,  0.22 , 0.19,  0.34 , 0.21,  0.39,  1 , 0.17],
    #              [0.4,  0.1 , 0.51 , 0.31,  0.34,  0.17,  0.46 , 0.15 , 0.17 , 1]]
    #
    #
    #
    #
    # simQuery = [0.91,0.28,0.5,0.06,0.63,0.45,0.8,0.15,0.2,0.3]


    # for i in range(0,len(simMatrix)):
    #     for j in range(0,len(simMatrix[0])):
    #         if simMatrix[i][j] == 0:
    #             simMatrix[i][j] = simMatrix[j][i]
    #
    #
    # for i in range(0,len(simMatrix)):
    #     for j in range(0,len(simMatrix[0])):
    #         print(simMatrix[i][j]," ", end='')
    #     print("")


    # mmrExample(simMatrix,simQuery,0.5,2)




def main():

    run(50000,300,1,15,0.5)

main()