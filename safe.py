import numpy as np
import collections 
import sys
import itertools 
import networkx as nx
import matplotlib.pyplot as plt


# Load the adjacancy matrix
G = np.loadtxt("paw.txt", int)
EdgeList = np.loadtxt("WTest.txt", int)
#---------------------------------------------------------
#print("The adjacency matrix of G is: ")
#print(G)
#----------------------------------------------------------
#f = open('WTest.txt','w')
G2=nx.Graph()

#---------------------------------------------------------

# Set the list of vertices 
VList = list(range(0,len(G)))
print("The list of vertices is: ", VList)

#---------------------------------------------------------

# Create a set of vertices
V = set(VList)
print("The set of vertices is: ", V)

#---------------------------------------------------------

# Order function returns the number of vertices of the graph
def order(G):
	n = len(G)
	return n
n = order(G)
print("The number of vertices in G is: ", n)
G2.add_nodes_from([n-1])
#----------------------------------------------------------

# Neighbors function returns the set of neighbors of a given vertex
def neighbors(G,v):
	neighborhood = set()
	for i in range(0,order(G)):
		if G[v][i] == 1:
			neighborhood.add(i)
	return neighborhood

# Example 
print("The open neighborhood of vertex 2 are: ", neighbors(G,2))

#-----------------------------------------------------------

# neighbors function returns the set of closed neighbors of a given vertex
def closedNeighbors(G,v):
	closedNeighbors = neighbors(G,v).union([v])
	return closedNeighbors
print("The closed neighborhood of vertex 2 are: ", closedNeighbors(G,2))
'''
#-----------------------------------------------------------
# Finds a set of sets of neighbors for all vertices in a set
def OpenNeighborSet(G, v):
    neighborhood = set()
    for i in v:
        x = neighbors(G, v)
        neighborhood.union(x) - neighborhood.intersection(x)
    return neighborhood

print("The open neighborhood set of G is: ", OpenNeighborSet(G, set([0, 2])))
'''
def edgeWeight(e):
    edgeW = EdgeList[e][2]
    return edgeW
#-----------------------------------------------------------
def size(G):
    size2 = 0
    for i in range(0, len(G)):
        for j in range(0, len(G[i])):
            if G[i][j] == 1:
                G2.add_edge(i,j)
                G2[i][j]['weight']=5
                size2 = size2 + 1

    size = size2 // 2
    return size
size = size(G)
print("The size of G is: ", size)

#-----------------------------------------------------------

def degree(G, v):
    return len(neighbors(G, v))

print("The degree of vertex 1 is: ", degree(G, 1))

#-----------------------------------------------------------

# Returns the degree sequence in nonincreasing order
def degreeSeq(G):
    D = []
    for i in range(0, order(G)):
        D.append(degree(G, i))
        D.sort(reverse = True)
    return D

DSeq = degreeSeq(G)
print("The degree sequence of G is: ", DSeq)

#------------------------------------------------------------
def MaximumDegree(G):
    maxd=DSeq[0]  
    return maxd    
MaxDeg=MaximumDegree(G)
print("Maximum Degree is ",MaxDeg)

def MinimumDegree(G):
    D=degreeSeq(G)
    D.sort()
    return D

MinDeg=MinimumDegree(G)
print("Minimum Degree is" ,MinDeg[0])
    

#-----------------------------------------------------------

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

res=residue(G)
print("RESIDUE",res)  
    

def get_dominating_sets(G2):
    """get all possible dominating sets by iterating all nodes"""
    dominating_sets = set() # list of sets
 
    for start_with in G2:
        dominating_set = frozenset(nx.dominating_set(G2, start_with))  #　sets of sets, the inner sets must be frozenset objects.
        dominating_sets.add(dominating_set)
 
    return dominating_sets
domset=get_dominating_sets(G2)
print("dom is ",domset)
#-----------------------------------------------------------
#print("The edge list is ", EdgeList)



print("The edge weight of the first edge is ", edgeWeight(4))
'''
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
#for i in range(0, order(G)):
#    G2.add_node(i)
    
print("Nodes of graph: ")
print(G2.nodes())
print("Edges of graph: ")
print(G2.edges())
pos = nx.spring_layout(G2)
nx.draw(G2,pos)
#plt.show()
edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G2.edges(data=True)])
nx.draw_networkx_edge_labels(G2,pos,edge_labels=edge_weight)
nx.draw_networkx_nodes(G2,pos)
nx.draw_networkx_edges(G2,pos)
nx.draw_networkx_labels(G2,pos)
plt.axis('off')
plt.show()


#nx.draw(G2,pos, node_color='green', labels=None, edge_color='blue', edge_labels=edge_weight)

