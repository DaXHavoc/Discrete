import numpy as np
import collections 
import sys
import itertools 
import networkx as nx
import matplotlib.pyplot as plt
import random

# Load the adjacancy matrix
A = np.loadtxt("paw.txt", int)

#---------------------------------------------------------
#print("The adjacency matrix of G is: ")
#print(G)
#----------------------------------------------------------
#f = open('WTest.txt','w')
G=nx.Graph()
G2=nx.Graph()
#---------------------------------------------------------

# Set the list of vertices 
VList = list(range(0,len(A)))
print("The list of vertices is: ", VList)

#---------------------------------------------------------

# Create a set of vertices
V = set(VList)
print("The set of vertices is: ", V)

n=len(A)

for i in range(0,n):
    G.add_node(i)
    
def order(G):
	n = len(G)
	return n
n = order(G)
print("The number of vertices in G is: ", n)

def size(G):
    size2 = 0
    for i in range(0, len(A)):
        for j in range(0, len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i,j)
#                G2[i][j]['weight']=5
                size2 = size2 + 1

    size = size2 // 2
    return size
size = size(G)
print("The size of G is: ", size)


def degree(G):
    return G.degree(G)
print("Degree of G is ",degree(G))


def degreeSeq(G):
    degree_sequence=list(degree(G).values())
    degree_sequence.sort(reverse = True)
    return degree_sequence
print("Degree Sequence is ", degreeSeq(G))


def maxDegree(G):
    return degreeSeq(G)[0]
maxD = maxDegree(G)
print("max degree = ", maxD)


def minDegree(G):
    return degreeSeq(G)[order(G) - 1]
minD = minDegree(G)
print("min degree = ", minD)


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
    
print("Residue ",residue(G))

def openNeighborhood(i):
    return set(G.neighbors(i))
print("Open Neighborhood of the vertex is",openNeighborhood(2))

def closedNeighborhood(i):
    sa=set(G.neighbors(i))
    sb={i}
    cn=set.union(sa,sb)
    return cn
print ("Closed Neighborhood of the vertex is ",closedNeighborhood(1))   


def openNeighborhoodSet(i,v):
    return set().union(G.neighbors(i),G.neighbors(v))
print("Open neighborhood of the set is ",openNeighborhoodSet(2,3))

def closedNeighborhoodSet(i,v):
    openset=openNeighborhoodSet(i,v)
    ownverts={i,v}
    closedset=set.union(openset,ownverts)
    return closedset
print("Closed Set is ",closedNeighborhoodSet(2,0))

def eccentricity(G, v=None, sp=None):
    order=G.order()

    e={}
    for n in G.nbunch_iter(v):
        if sp is None:
            length=nx.single_source_shortest_path_length(G,n)
            L = len(length)
        else:
            try:
                length=sp[n]
                L = len(length)
            except TypeError:
                raise nx.NetworkXError('Format of "sp" is invalid.')
        if L != order:
            msg = "Graph not connected: infinite path length"
#            raise nx.NetworkXError(msg)
            
        e[n]=max(length.values())

    if v in G:
        return e[v]  # return single value
    else:
        return e
print("Eccentricity ",eccentricity(G,None))


def diameter(G, e=None):
    if e is None:
        e=eccentricity(G)
    return max(e.values())   
print("Diameter ",diameter(G))


def radius(G, e=None):
    if e is None:
        e=eccentricity(G)
    return min(e.values())
print("Radius ",radius(G))


def dominating_set(G, start_with=None):
    all_nodes = set(G)
    if start_with is None:
        v = set(G).pop() # pick a node
    else:
        if start_with not in G:
            raise nx.NetworkXError('node %s not in G' % start_with)
        v = start_with
    D = set([v])
    ND = set(G[v])
    other = all_nodes - ND - D
    while other:
        w = other.pop()
        D.add(w)
        ND.update([nbr for nbr in G[w] if nbr not in D])
        other = all_nodes - ND - D
    return D    
print("Dominating Set ", dominating_set(G))


def find_cliques(G):
    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = [None]

    subg = set(G)
    cand = set(G)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass

print("Cliques of the graph ", list(find_cliques(G)))

def graph_clique_number(G,cliques=None):
    if cliques is None:
        cliques=find_cliques(G)
    return   max( [len(c) for c in cliques] )
    
print("Clique number ", graph_clique_number(G))

'''
def get_dominating_sets(G):
    """get all possible dominating sets by iterating all nodes"""
    dominating_sets = set() # list of sets
 
    for start_with in G:
        dominating_set = frozenset(nx.dominating_set(G, start_with))  #　sets of sets, the inner sets must be frozenset objects.
        dominating_sets.add(dominating_set)
 
    return dominating_sets
domset=get_dominating_sets(G)
print("dom is ",domset)
'''

def maximal_independent_set(G, nodes=None):
 
    if not nodes:
        nodes = set([random.choice(G.nodes())])     # pick a random node
    else:
        nodes = set(nodes)
    if not nodes.issubset(G):
        raise nx.NetworkXUnfeasible("%s is not a subset of the nodes of G" % nodes)
 
    # All neighbors of nodes
    neighbors = set.union(*[set(G.neighbors(v)) for v in nodes])
    if set.intersection(neighbors, nodes):
        raise nx.NetworkXUnfeasible("%s is not an independent set of G" % nodes)
 
    indep_nodes = list(nodes)       # initial
    available_nodes = set(G.nodes()).difference(neighbors.union(nodes)) # available_nodes = all nodes - (nodes + nodes' neighbors)
 
    while available_nodes:
        # pick a random node from the available nodes
        node = random.choice(list(available_nodes))
        indep_nodes.append(node)
 
        available_nodes.difference_update(G.neighbors(node) + [node])   # available_nodes = available_nodes - (node + node's neighbors)
 
    return indep_nodes
maxIndset=maximal_independent_set(G)
print("MAX INDEPENDENT SET", maximal_independent_set(G))
print("INDEPENDENCE NUMBER ",maximal_independent_set(G),len(maximal_independent_set(G)))
def complement(G, name=G2):

    if name is None:
        name = "complement(%s)" % (G.name)
    R = G.__class__()
    R.name = name
    R.add_nodes_from(G)
    R.add_edges_from(((n, n2)
                      for n, nbrs in G.adjacency_iter()
                      for n2 in G if n2 not in nbrs
                      if n != n2))
    return R
G2=complement(G)
#print(nx.draw(G2, node_color='blue', labels=None, edge_color='black',))

print(set(itertools.combinations(G,2)))
'''
sa = set(G.neighbors(5))
sb = set(G.neighbors(0))
c = sa.intersection(sb)

print(G.neighbors(5))
print(G.neighbors(6))
print("Open Neighborhood of 0 and 3 are ",set().union(G.neighbors(5),G.neighbors(6)))
print(set.intersection(sa,sb))
print(sorted(nx.common_neighbors(G, 0, 2)))
'''
nx.draw(G, node_color='white', labels=None, edge_color='black',)