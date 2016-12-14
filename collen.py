import numpy as np
from winerror import STG_E_FILENOTFOUND
#from ctypes.test import test_sizes
#import collections
#import sys
#import itertools
#import math




while True:
    
    filename = input("Input graph matrix file name (.txt not required)")
    
    # Doesn't care if .txt is added or not
    if filename[len(filename)-4:] != ".txt":
        filename += ".txt"
    
    try:
        # Load the adjacency matrix
        G = np.loadtxt(filename, int)
    except FileNotFoundError:
        print("Incorrect file name!")
    else:
        break


    
print("The adjacency matrix of G is: ")
print(G)


# Set the list of vertices 
VList = list(range(0, len(G)))
print("The list of vertices is: ", VList)


# Create a set of vertices
V = set(VList)
print("The set of vertices is: ", V)


# Order function returns the number of vertices of the graph
def order(G):
    return len(G)


n = order(G)
print("The number of vertices in G is: ", n)


def size(G):
    size2 = 0
    for i in range(0, len(G)):
        for j in range(0, len(G[i])):
            if G[i][j] == 1:
                size2 = size2 + 1
    size = size2 // 2   # // is floor division
    return size


size = size(G)
print("The size of G is: ", size)


# Neighbors function returns the set of neighbors of a given vertex
def OpenNeighbors(G, v):
    neighborhood = set()
    for i in range(0, order(G)):
        if G[v][i] == 1:
            neighborhood.add(i)
    return neighborhood


# Example 
print("The neighbors of vertex 2 are: ", OpenNeighbors(G, 2))

'''
# Finds a set of sets of neighbors for all vertices in a set
def OpenNeighborSet(G, v):
    neighborhood = set()
    for i in v:
        x = OpenNeighbors(G, v[i-1])
        neighborhood.union(x) - neighborhood.intersection(x) #TODO: Someone fix this piece of shit function, kthxbai
    return neighborhood


print("The open neighborhood set of 0 and 1 is: ", OpenNeighborSet(G, list([0, 1])))
'''



# neighbors function returns the set of closed neighbors of a given vertex
def closedNeighbors(G, v):
    closedNeighbors = OpenNeighbors(G,v).union([v])
    return closedNeighbors


def ClosedNeighborSet(G, BSet):
    neighborhood = set()
    for i in BSet:
        neighborhood = neighborhood.union(closedNeighbors(G, BSet[i]))
    return neighborhood        


print("The closed neighborhood set of 0 and 1 is: ", ClosedNeighborSet(G, list([0, 1])))
        
# Returns the degree of a given 
def degree(G, v):
    return len(OpenNeighbors(G, v))


print("The degree of vertex 1 is: ", degree(G, 1))


# Returns the degree sequence in nonincreasing order
def degreeSeq(G):
    D = []
    for i in range(0, order(G)):
        D.append(degree(G, i))
        D.sort(reverse = True)
    return D


DSeq = degreeSeq(G)
print("The degree sequence of G is: ", DSeq)




def residue(G):
    D = degreeSeq(G)
    maxD = D[0]
    while maxD > 0:
        D.remove(D[0])
        for i in range(0, maxD):
            D[i] = D[i] - 1
        D.sort(reverse = True)
        maxD = D[0]
    residue = len(D)
    return residue


def maxDegree(G):
    return degreeSeq(G)[0]


maxD = maxDegree(G)
print("max degree = ", maxD)


def minDegree(G):
    return degreeSeq(G)[order(G) - 1]


minD = minDegree(G)
print("min degree = ", minD)


'''

# Eccentricity function
def eccentricity(G, v):
    observed = set({v})
    distArray = [[v]]
    while observed != v:
        distArray = distArray.append(list(OpenNeighborSet(G, list(observed)) - observed))
        print(distArray)
        observed = observed.union(OpenNeighbors(G, list(observed)))
    ecc = len(distArray) - 1
    return ecc


print("The ecc of vertex 1 is: ", eccentricity(G, 1))


# Accepts a vertex set of a graph
def diameter(G, V):
    Vecc = []
    for i in V:
        Vecc = eccentricity(G, V[i])
    Vecc.sort()
    return Vecc[order(G) - 1]
        


#=======================================================================



print("The edge list is ", EdgeList)


def edgeWeight(e):
    edgeW = EdgeList[e][2]
    return edgeW


print("The edge weight of the first edge is ", edgeWeight(0))


v = 2




def vertexIncidentEdges(v):
    incidentEdges = []
    for i in range(0, len(EdgeList)):
        if EdgeList[i][0] == v or EdgeList[i][1] == v:
            incidentEdges.append(EdgeList[i])
    return incidentEdges


print("Edges incident with vertex 1 are ", vertexIncidentEdges(1))


S = set([0,1])


def setIncidentEdges(S):
    incidentEdges = []
    for i in S:
        incidentEdges.append(vertexIncidentEdges(i))
    return incidentEdges


print("The edges incident with vertex 0 and 1 are ", setIncidentEdges(S))


def minIncidentEdges(E):
    minEdge = E[0]
    leastCost = edgeWeight(E[0])
    for i in range(0, len(E)):
        if edgeWeight(E[i]) < leastCost:
            minEdge = E[i]
            leastCost = edgeWeight(E[i])
    return minEdge


print("A least-cost edge incident with the set S is ")

'''