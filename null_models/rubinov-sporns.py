from collections import Counter
import networkx as nx
import numpy as np

def transfer_weights_rubinov_sporns(G_source, G_dest, weight_name='weight'):
    """ Uses a greedy approach to assign weights to edges so that the initial strength sequence is preserved.

    >>> transfer_weights_rubinov_sporns(G, G_swap, 'weight')
    """

    in_strengths = dict(G_source.in_degree(weight=weight_name))
    out_strengths = dict(G_source.out_degree(weight=weight_name))

    nx.set_edge_attributes(G_dest, 0, name=weight_name)

    to_be_assigned = sorted([edge[2][weight_name] for edge in list(G_source.edges(data=True))]) # increasing

    to_be_weighted = list(G_dest.edges())

    # find the highest expected strength diff (how can this go wrong?)    
    while len(to_be_weighted) > 0:
    
        in_strengths_current = Counter()
        out_strengths_current = Counter()
        
        for u, v, atts in G_dest.edges(data=True):
            out_strengths_current[u] += atts[weight_name]
            in_strengths_current[v] += atts[weight_name]
        
        highest_eij = -np.inf
        ij = None
        
        for (u, v) in to_be_weighted:
            new_eij = (out_strengths[u] - out_strengths_current[u]) * (in_strengths[v] - in_strengths_current[v])
        
            if new_eij > highest_eij:
                ij = (u, v)
                highest_eij = new_eij
    
        nx.set_edge_attributes(G_dest, {ij: to_be_assigned.pop()}, 'weight')
        to_be_weighted.remove(ij)

    