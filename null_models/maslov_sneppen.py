from networkx.utils import py_random_state
import networkx as nx
import numpy as np

def get_near_successor(G, key, distance_matrix, key_to_ind, ind_to_key, seed=None):

    if G.out_degree(key) == 0:
        return None
    
    nodes_list = list(G.nodes)
        
    start_ind = key_to_ind[key]
    succ_inds = [key_to_ind[succ] for succ in list(G.succ[key])]

    full_distances = distance_matrix[start_ind, :]
    mask = np.ones(len(full_distances), dtype=bool)
    mask[succ_inds] = False
    masked_distances = full_distances.copy()
    masked_distances[mask] = np.inf
    masked_distances[np.where(masked_distances == 0)] = np.inf # might have some wrong distances equal to zero
    
    masked_chances = 1 / masked_distances

    cdf = nx.utils.cumulative_distribution(masked_chances)

    discrete_sequence = nx.utils.discrete_sequence

    near_successor_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]

    out_key = ind_to_key[near_successor_index]

    # this is to avoid outputting nodes that exist in the total stations but not in our current network
    while out_key not in nodes_list: 

        near_successor_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]

        out_key = ind_to_key[near_successor_index]
    
    return out_key

def get_near_node(G, key, distance_matrix, key_to_ind, ind_to_key, seed=None):
        
    if G.degree(key) == 0:
        return None
    
    nodes_list = list(G.nodes())
        
    start_ind = key_to_ind[key]

    full_distances = distance_matrix[start_ind, :]

    masked_distances = full_distances.copy()
    masked_distances[start_ind] = np.inf
    masked_distances[np.where(masked_distances == 0)] = np.inf # might have some wrong distances equal to zero


    masked_chances = 1 / masked_distances

    cdf = nx.utils.cumulative_distribution(masked_chances)

    discrete_sequence = nx.utils.discrete_sequence

    near_successor_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]

    out_key = ind_to_key[near_successor_index]

    # this is to avoid outputting nodes that exist in the total stations but not in our current network
    while out_key not in nodes_list:

        near_successor_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]

        out_key = ind_to_key[near_successor_index]
    
    return out_key

@py_random_state('seed')
def double_swap(G, *, nswap=1, max_tries=100, seed=None, keep=None, ):
    """Swap two edges in a directed graph while keeping the node degrees fixed.
    
    >>> G_swap = double_swap(G.copy(), nswap=G.number_of_edges(), max_tries=20 * G.number_of_edges(), keep=None)
    """
    
    if nswap > max_tries:
        raise Exception("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise Exception("DiGraph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise Exception("DiGraph has fewer than 2 edges")
    if keep not in ['in', 'out', None]:
        raise Exception("keep should be any of the following: 'in', 'out', None")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    while swapcount < nswap:
        # choose source node index from discrete distribution
        starting_points = discrete_sequence(2, cdistribution=cdf, seed=seed)

        start1 = keys[starting_points[0]]
        start2 = keys[starting_points[1]]
        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise Exception(msg)

        # If the two chosen outgoing nodes are the same, skip.
        if starting_points[0] == starting_points[1]:
            continue
            
        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start1) == 0:
            continue
        end1 = seed.choice(list(G.succ[start1]))
        if start1 == end1: # won't happen, no self-loop
            continue

        if G.out_degree(start2) == 0:
            continue
        end2 = seed.choice(list(G.succ[start2]))
        if start2 == end2: # won't happen, no self-loop
            continue

        if ( 
            end2 not in G.succ[start1]
            and end1 not in G.succ[start2]
            and start1 != end2
            and start2 != end1
        ):
            # Swap nodes
            att_ab, att_cd = G.edges[(start1, end1)], G.edges[(start2, end2)]

            if keep is None:
                G.add_edge(start1, end2)
                G.add_edge(start2, end1)
                
            elif keep == 'in':
                G.add_edge(start1, end2, **att_cd)
                G.add_edge(start2, end1, **att_ab)

            else: # out
                G.add_edge(start1, end2, **att_ab)
                G.add_edge(start2, end1, **att_cd)

            G.remove_edge(start1, end1)
            G.remove_edge(start2, end2)
                
            swapcount += 1

    return G

@py_random_state('seed')
def triple_swap(G, *, nswap=1, max_tries=100, seed=None, keep=None):
    """Swap three edges in a directed graph while keeping the node degrees fixed.
    
    >>> G_swap = triple_swap(G.copy(), nswap=G.number_of_edges(), max_tries=20 * G.number_of_edges(), keep=None)
    """
    
    if nswap > max_tries:
        raise Exception("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise Exception("DiGraph has fewer than four nodes.")
    if len(G.edges) < 3:
        raise Exception("DiGraph has fewer than 3 edges")
    if keep not in ['in', 'out', None]:
        raise Exception("keep should be any of the following: 'in', 'out', None")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    while swapcount < nswap:
        # choose source node index from discrete distribution
        start_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
        start = keys[start_index]
        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise Exception(msg)

        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start) == 0:
            continue
        second = seed.choice(list(G.succ[start]))
        if start == second:
            continue

        if G.out_degree(second) == 0:
            continue
        third = seed.choice(list(G.succ[second]))
        if second == third:
            continue

        if G.out_degree(third) == 0:
            continue
        fourth = seed.choice(list(G.succ[third]))
        if third == fourth:
            continue

        if (
            third not in G.succ[start]
            and fourth not in G.succ[second]
            and second not in G.succ[third]
        ):
            # Swap nodes
            att12, att23, att34 = G.edges[(start, second)], G.edges[(second, third)], G.edges[(third, fourth)]

            if keep is None:
                G.add_edge(start, third)
                G.add_edge(third, second)
                G.add_edge(second, fourth)
                
            elif keep == 'in':
                G.add_edge(start, third, **att23)
                G.add_edge(third, second, **att12)
                G.add_edge(second, fourth, **att34)

            else: # out
                G.add_edge(start, third, **att12)
                G.add_edge(third, second, **att34)
                G.add_edge(second, fourth, **att23)

            G.remove_edge(start, second)
            G.remove_edge(second, third)
            G.remove_edge(third, fourth)
                
            swapcount += 1

    return G

@py_random_state('seed')
def double_swap_distances(G, distance_matrix, key_to_ind, ind_to_key, *, nswap=1, max_tries=100, seed=None, keep=None):
    """Swap two edges in a directed graph while keeping the node degrees fixed. Use the distance matrix to choose closer nodes with higher chance.

    >>> G_swap = double_swap_distances(G.copy(), distance_matrix, key_to_ind, ind_to_key, nswap=G.number_of_edges(), max_tries=20 * G.number_of_edges(), keep=None)
    """
    
    if nswap > max_tries:
        raise Exception("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise Exception("DiGraph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise Exception("DiGraph has fewer than 2 edges")
    if keep not in ['in', 'out', None]:
        raise Exception("keep should be any of the following: 'in', 'out', None")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    while swapcount < nswap:
        # choose source node index from discrete distribution
        start1_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
        start1 = keys[start1_index]

        start2 = get_near_node(G, start1, distance_matrix, key_to_ind, ind_to_key)

        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise Exception(msg)

        # If the two chosen outgoing nodes are the same, skip.
        if start1 == start2:
            continue
            
        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start1) == 0:
            continue

        end1 = get_near_successor(G, start1, distance_matrix, key_to_ind, ind_to_key)
        if start1 == end1: # won't happen, no self-loop
            continue

        if G.out_degree(start2) == 0:
            continue

        end2 = get_near_successor(G, start2, distance_matrix, key_to_ind, ind_to_key)
        if start2 == end2: # won't happen, no self-loop
            continue

        if ( 
            end2 not in G.succ[start1]
            and end1 not in G.succ[start2]
            and start1 != end2
            and start2 != end1
        ):
            
            # Swap nodes
            att_ab, att_cd = G.edges[(start1, end1)], G.edges[(start2, end2)]

            if keep is None:
                G.add_edge(start1, end2)
                G.add_edge(start2, end1)
                
            elif keep == 'in':
                G.add_edge(start1, end2, **att_cd)
                G.add_edge(start2, end1, **att_ab)

            else: # out
                G.add_edge(start1, end2, **att_ab)
                G.add_edge(start2, end1, **att_cd)

            G.remove_edge(start1, end1)
            G.remove_edge(start2, end2)
                
            swapcount += 1

    return G

@py_random_state('seed')
def triple_swap_distances(G, distance_matrix, key_to_ind, ind_to_key, *, nswap=1, max_tries=100, seed=None, keep=None):
    """Swap three edges in a directed graph while keeping the node degrees fixed. Use the distance matrix to choose closer nodes with higher chance.

    >>> G_swap = triple_swap_distances(G.copy(), distance_matrix, key_to_ind, ind_to_key, nswap=G.number_of_edges(), max_tries=20 * G.number_of_edges(), keep=None)
    """
    
    if nswap > max_tries:
        raise Exception("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise Exception("DiGraph has fewer than four nodes.")
    if len(G.edges) < 3:
        raise Exception("DiGraph has fewer than 3 edges")
    if keep not in ['in', 'out', None]:
        raise Exception("keep should be any of the following: 'in', 'out', None")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence

    while swapcount < nswap:
        # choose source node index from discrete distribution
        start_index = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
        start = keys[start_index]
        tries += 1

        if tries > max_tries:
            msg = f"Maximum number of swap attempts ({tries}) exceeded before desired swaps achieved ({nswap})."
            raise Exception(msg)

        # If the given node doesn't have any out edges, then there isn't anything to swap
        if G.out_degree(start) == 0:
            continue

        second = get_near_successor(G, start, distance_matrix, key_to_ind, ind_to_key)
        if start == second:
            continue

        if G.out_degree(second) == 0:
            continue

        third = get_near_successor(G, second, distance_matrix, key_to_ind, ind_to_key)
        if second == third:
            continue

        if G.out_degree(third) == 0:
            continue

        fourth = get_near_successor(G, third, distance_matrix, key_to_ind, ind_to_key)
        if third == fourth:
            continue

        if (
            third not in G.succ[start]
            and fourth not in G.succ[second]
            and second not in G.succ[third]
        ):
            # Swap nodes
            att12, att23, att34 = G.edges[(start, second)], G.edges[(second, third)], G.edges[(third, fourth)]

            if keep is None:
                G.add_edge(start, third)
                G.add_edge(third, second)
                G.add_edge(second, fourth)
                
            elif keep == 'in':
                G.add_edge(start, third, **att23)
                G.add_edge(third, second, **att12)
                G.add_edge(second, fourth, **att34)

            else: # out
                G.add_edge(start, third, **att12)
                G.add_edge(third, second, **att34)
                G.add_edge(second, fourth, **att23)

            G.remove_edge(start, second)
            G.remove_edge(second, third)
            G.remove_edge(third, fourth)
                
            swapcount += 1

    return G