from pyclustering.utils import euclidean_distance_square
from sklearn.datasets.samples_generator import make_blobs
from clustering import  Clustering
from distance import min_distance, max_distance
import numpy as np
import timeit

numberOfCluster = 20
numberOfLevels = 2

def aug_mmr(cluster,lambda_score, q, data, k):

    docs_unranked = data

    docs_selected = []


    for i in range (k):
        mmr = -100000000
        #start = timeit.default_timer()
        # Your statements here
        R = getNext(cluster,docs_unranked,docs_selected,q,lambda_score)
        print(len(R))
        #stop = timeit.default_timer()
        #print('Time for getnext: ', stop - start)

        best1 = [0,0]
        item = [0,0]
        for item in docs_selected:
            if item in R:
                R.remove(item)


        for d in R:
            sim = 0
            for s in docs_selected:
                if euclidean_distance_square(d, s) == 0:
                    continue
                sim_current = 1/euclidean_distance_square(d, s)
                if sim_current > sim:
                    sim = sim_current
                else:
                    continue

            rel = 1/euclidean_distance_square(q, d)
            mmr_current = lambda_score * rel - (1 - lambda_score) * sim

            if mmr_current > mmr:
                mmr = mmr_current
                best1 = d
            else:
                continue


        docs_selected.append(best1)
        #docs_unranked.remove(best)

    return docs_selected



def getNext(cluster,docs_unranked,docs_selected,q,lambda_score):

    min_mmr_min =  np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1,numberOfLevels + 1),
        dtype=float)
    min_mmr_min.fill(100000)
    max_mmr_max =  np.empty(
        shape=(numberOfCluster ** numberOfLevels + 1,numberOfLevels + 1),
        dtype=float)
    max_mmr_max.fill(-1000000)



    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it

    for node in cluster.root.children:
        queue.append(node)

    prevLevel = 1
    levelClusters = []

    while queue:

        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)
        levelClusters.append(s)

        clsid = s.id
        l = s.level


        simmax = 0
        simmin = 10000000
        #start = timeit.default_timer()
        # Your statements here
        for d in docs_selected:
            id = cluster.documentMap[tuple(d)][l]
            maxcdis,mincdis = cluster.dismatrix[l][id][clsid]

            sim_current_max = 1 / mincdis
            sim_current_min = 1 / maxcdis
            if sim_current_max > simmax:
                simmax = sim_current_max
            if sim_current_min < simmin:
                simmin = sim_current_min
            else:
                continue
        if len(docs_selected) == 0:
            simmax = 0
            simmin = 0

        maxdis = max_distance([q], s.elements)
        mindis = min_distance([q], s.elements)

        relmax = 1 / mindis
        relmin = 1 / maxdis

        min_mmr_min[clsid][l] = lambda_score * relmin - (1 - lambda_score) * simmax
        max_mmr_max[clsid][l] = lambda_score * relmax - (1 - lambda_score) * simmin
        mmm1 = min_mmr_min[clsid][l]
        mmm2 = max_mmr_max[clsid][l]
        #print("mmm1 ",mmm1)
        #print("mmm2 ",mmm2)
        #stop = timeit.default_timer()
        #print('Time for calculate mmr: ', stop - start)


        if len(queue) == 0:
            #start = timeit.default_timer()
            # Your statements here
            max_min_mmr_min = 0
            for node1 in levelClusters:
                if max_min_mmr_min < min_mmr_min[node1.id][l]:
                    max_min_mmr_min = min_mmr_min[node1.id][l]

            for node2 in levelClusters:
                if max_mmr_max[node2.id][l] < max_min_mmr_min:
                    if node1  in levelClusters:
                        levelClusters.remove(node2)

            #stop = timeit.default_timer()
            #print('Time for senjuti change: ', stop - start)

            #start = timeit.default_timer()
            # Your statements here
            for node in levelClusters:
                for children in node.children:
                    queue.append(children)

            #stop = timeit.default_timer()
            #print('Time for appending level cluster: ', stop - start)


            if queue:
                levelClusters.clear()

    #start = timeit.default_timer()
    # Your statements here
    R = []
    for c in levelClusters:
        for item in c.elements:
            R.append(item)

    #stop = timeit.default_timer()
    #print('Time for R: ', stop - start)


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
                sim_current = 1/euclidean_distance_square(d, s)
                if sim_current > sim:
                    sim = sim_current
                else:
                    continue

            rel = 1/euclidean_distance_square(q, d)
            mmr_current = lambda_score * rel - (1 - lambda_score) * sim

            if mmr_current > mmr:
                mmr = mmr_current
                best = d
            else:
                continue


        docs_selected.append(best)

        np.delete(docs_unranked,best)


    return docs_selected




def main():
    #X = [[1, 1], [1, 2],[1,3], [4, 4],[4, 5], [5, 4], [5, 5], [10, 9], [10,10], [20,19], [20, 20]]
    X,Y = make_blobs(n_samples=5000, centers=10, cluster_std=0.60, random_state=0)
    cluster = Clustering(X.tolist())
    cluster.buildTree(cluster.root)

    cluster.createLevelMatrix(cluster.root)
    cluster.createDistanceMatrix(numberOfCluster,numberOfLevels)

    query = [0,0]

    start = timeit.default_timer()
    # Your statements here
    print("aug", aug_mmr(cluster, 0.5, query, X, 15))
    stop = timeit.default_timer()
    print('Time for aug mmr: ', stop - start)

    start = timeit.default_timer()

    # Your statements here
    print("mmr", _mmr(0.5, query, X, 15))

    stop = timeit.default_timer()

    print('Time for mmr: ', stop - start)




main()