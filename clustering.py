import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from Node import Node
from distance import min_distance, max_distance
import numpy

numberOfCluster = 100
numberOfLevels = 1
class Clustering:
    documentMap = {}


    def __init__(self, data,indexMap):
        self.root = Node(None,None, 0, 1)
        self.root.elements = data
        self.indexMap = indexMap
        self.root.numberOfElement = len(data)
        for i in range(len(data)):
            d = data[i]
            self.documentMap[tuple(d)] = numpy.zeros(numberOfLevels+1, dtype=int)

    def buildTree(self,parent):
        #print(parent.elements)
        if parent.level == numberOfLevels:
            return

        kmeans = KMeans(n_clusters=numberOfCluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(parent.elements)

        for i in range(len(parent.elements)):
            self.documentMap[tuple(parent.elements[i])][parent.level] = parent.id

        parent.children = []
        for i in range(0,numberOfCluster):
            id =parent.id * numberOfCluster - numberOfCluster + 1  + i
            new_node = Node(self.indexMap,parent,parent.level+1,id)
            new_node.elements = []
            new_node.numberOfChildren = 0
            parent.setChildren(new_node)

        j = -1
        pdict = {}
        v = 0
        for i in pred_y:
            if i in pdict:
                i = pdict.get(i)
            else:
                pdict[i] = v
                i = v
                v = v + 1
            j = j + 1
            c = parent.children[i]
            c.insertElement(parent.elements[j])
            self.documentMap[tuple(parent.elements[j])][c.level] = c.id

        #print("elements = ", parent.children[1].elements,"\n")
        for i in range (0,parent.numberOfChildren):
            self.buildTree(parent.children[i])







    dismatrix = numpy.empty(
        shape=(numberOfLevels+1, numberOfCluster ** numberOfLevels + 1, numberOfCluster ** numberOfLevels + 1), dtype=tuple)
    levelMatrix = numpy.empty(
        shape=(numberOfLevels+1, numberOfCluster ** numberOfLevels+1), dtype=Node)


    def createLevelMatrix(self, currentNode):
        nodes = currentNode.children

        if currentNode.numberOfChildren == 0:
            return
        for node in currentNode.children:
            self.levelMatrix[currentNode.level + 1][node.id] = node
            self.createLevelMatrix(node)


    def createDistanceMatrix(self, numberOfCluster, numberOfLevels):
    # print(max_matrix)
        for l in range(1,numberOfLevels + 1):
            for i in range(1,numberOfCluster**l + 1):
                for j in range(1,numberOfCluster**l + 1):
                    cluster1 = self.levelMatrix[l][i].elements
                    cluster2 = self.levelMatrix[l][j].elements

                    self.dismatrix[l, self.levelMatrix[l][i].id, self.levelMatrix[l][j].id] = (max_distance(self.levelMatrix[l][i].elements, self.levelMatrix[l][j].elements), min_distance(self.levelMatrix[l][i].elements, self.levelMatrix[l][j].elements))


        #print(self.dismatrix)

        return self.dismatrix

    dismatrixitem = None

    def createDistanceMatrixforelements(self, numberOfCluster, numberOfLevels):
        global  dismatrixitem
        # print(max_matrix)
        self.dismatrixitem = numpy.empty(
            shape=(numberOfLevels + 1,  len(self.root.elements), numberOfCluster ** numberOfLevels + 1),
            dtype=tuple)
        for l in range(1, numberOfLevels + 1):
            for i in range(0, len(self.root.elements)):
                for j in range(1, numberOfCluster ** l + 1):
                    self.dismatrixitem[l, i , self.levelMatrix[l][j].id] = (
                    max_distance([self.root.elements[i]], self.levelMatrix[l][j].elements),
                    min_distance([self.root.elements[i]], self.levelMatrix[l][j].elements))

        # print(self.dismatrix)

        return self.dismatrixitem

'''
X = [[1, 1], [1, 2],[1,3], [4, 4],[4, 5], [5, 4], [5, 5], [10, 9], [10,10], [20,19], [20, 20]]
    #make_blobs(n_samples=200, centers=10, cluster_std=0.60, random_state=0)
c = Clustering(X)
c.buildTree(c.root)
print(c.documentMap)
c.createLevelMatrix(c.root)
c.createDistanceMatrix(numberOfCluster,numberOfLevels)
'''