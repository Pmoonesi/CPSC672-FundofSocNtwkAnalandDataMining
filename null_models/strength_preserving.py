import numpy as np
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_random_state

from .maslov_sneppen import double_swap_distances
from .maslov_sneppen import triple_swap_distances

def strength_preserving_rand_sa_dir_dist(
    G,
    distance_matrix,
    key_to_ind,
    ind_to_key,
    weight_list=['weight'],
    rewiring_method='double',
    rewiring_iter=1,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    energy_type="sse",
    energy_func=None,
    connected=True,
    verbose=False,
    seed=None,
    show_tqdm=False,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize a directed weighted network, while preserving
    the in- and out-degree and strength sequences using simulated annealing.

    
    This function is adapted from netneurotools package and using our custom
    rewiring algorithm, takes nodes distances into account when selecting 
    edges to swap.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed weighted networkx graph
    distance_matrix: (N, N) array-like
        Pairwise distance between the nodes of the graph.
    key_to_ind: dict
        Mapping from graph nodes to their respective indices in the 
        distance matrix.
    ind_to_key: dict
        Reverse mapping of key_to_ind.
    weight_list : list
        List of weights to be perturbed while keeping strenght sequence.
    rewiring_method: str
        Either one of "double" and "triple"
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squared errors between strength sequence vectors
                   of the original network and the randomized network
            'max': Maximum absolute error
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        Default = True.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.
    show_tqdm: bool, optional
        Whether to show progress bars or not. Default = False.

    Returns
    -------
    G_swap_annealing : networkx.DiGraph
        Randomized directed weighted networkx graph

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and in- and out-degree sequences as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.
    Both in- and out-strengths are preserved.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Rubinov, M. (2016) Constraints and spandrels of interareal connectomes.
    Nature Communications.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)
    
    if rewiring_method not in ["double", "triple"]:
        msg = "you should choose between \"double\" and \"triple\" for rewiring method."
        raise ValueError(msg)

    rs = check_random_state(seed)

    nodes = list(G.nodes())

    # My own rewiring
    if rewiring_method == "double":
        G_swap = double_swap_distances(G.copy(), distance_matrix, key_to_ind, ind_to_key, nswap=rewiring_iter * G.number_of_edges(), max_tries=100 * rewiring_iter * G.number_of_edges(), keep='in')
    else:
        G_swap = triple_swap_distances(G.copy(), distance_matrix, key_to_ind, ind_to_key, nswap=rewiring_iter * G.number_of_edges(), max_tries=100 * rewiring_iter * G.number_of_edges(), keep='in')

    G_swap_annealing = nx.empty_graph(0, nx.DiGraph)
    G_swap_annealing.add_nodes_from(nodes)

    for weight in weight_list:
        
        #The rows and columns are ordered according to the nodes in nodelist. If nodelist is None, then the ordering is produced by G.nodes().
        A = nx.to_numpy_array(G, weight=weight)
    
        n = A.shape[0]
        s_in = np.sum(A, axis=0)  # in-strengths of A
        s_out = np.sum(A, axis=1)  # out-strengths of A
        
        B = nx.to_numpy_array(G_swap, weight=weight)
    
        u, v = B.nonzero()  # nonzero indices of B
        wts = B[(u, v)]  # nonzero values of B
        m = len(wts)
        sb_in = np.sum(B, axis=0)  # in-strengths of B
        sb_out = np.sum(B, axis=1)  # out-strengths of B
    
        if energy_func is not None:
            energy = energy_func(s_in, sb_in) + energy_func(s_out, sb_out)
        elif energy_type == "sse":
            energy = np.sum((s_in - sb_in) ** 2) + np.sum((s_out - sb_out) ** 2)
        elif energy_type == "max":
            energy = np.max(np.abs(s_in - sb_in)) + np.max(np.abs(s_out - sb_out))
        elif energy_type == "mae":
            energy = np.mean(np.abs(s_in - sb_in)) + np.mean(np.abs(s_out - sb_out))
        elif energy_type == "mse":
            energy = np.mean((s_in - sb_in) ** 2) + np.mean((s_out - sb_out) ** 2)
        elif energy_type == "rmse":
            energy = np.sqrt(np.mean((s_in - sb_in) ** 2)) + np.sqrt(
                np.mean((s_out - sb_out) ** 2)
            )
        else:
            msg = (
                "energy_type must be one of 'sse', 'max', "
                "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type)
            )
            raise ValueError(msg)
    
        energymin = energy
        wtsmin = wts.copy()
    
        if verbose:
            print("\ninitial energy {:.5f}".format(energy))
    
        for istage in tqdm(range(nstage), desc="annealing progress", disable=not show_tqdm):
            naccept = 0
            for _ in range(niter):
                # permutation
                e1 = rs.randint(m)
                e2 = rs.randint(m)
    
                a, b = u[e1], v[e1]
                c, d = u[e2], v[e2]
    
                sb_prime_in = sb_in.copy()
                sb_prime_out = sb_out.copy()
                sb_prime_in[b] = sb_prime_in[b] - wts[e1] + wts[e2]
                sb_prime_out[a] = sb_prime_out[a] - wts[e1] + wts[e2]
                sb_prime_in[d] = sb_prime_in[d] - wts[e2] + wts[e1]
                sb_prime_out[c] = sb_prime_out[c] - wts[e2] + wts[e1]
    
                if energy_func is not None:
                    energy_prime = energy_func(sb_prime_in, s_in) + energy_func(
                        sb_prime_out, s_out
                    )
                elif energy_type == "sse":
                    energy_prime = np.sum((sb_prime_in - s_in) ** 2) + np.sum(
                        (sb_prime_out - s_out) ** 2
                    )
                elif energy_type == "max":
                    energy_prime = np.max(np.abs(sb_prime_in - s_in)) + np.max(
                        np.abs(sb_prime_out - s_out)
                    )
                elif energy_type == "mae":
                    energy_prime = np.mean(np.abs(sb_prime_in - s_in)) + np.mean(
                        np.abs(sb_prime_out - s_out)
                    )
                elif energy_type == "mse":
                    energy_prime = np.mean((sb_prime_in - s_in) ** 2) + np.mean(
                        (sb_prime_out - s_out) ** 2
                    )
                elif energy_type == "rmse":
                    energy_prime = np.sqrt(np.mean((sb_prime_in - s_in) ** 2)) + np.sqrt(
                        np.mean((sb_prime_out - s_out) ** 2)
                    )
                else:
                    msg = (
                        "energy_type must be one of 'sse', 'max', "
                        "'mae', 'mse', or 'rmse'. "
                        "Received: {}.".format(energy_type)
                    )
                    raise ValueError(msg)
    
                # permutation acceptance criterion
                if energy_prime < energy or rs.rand() < np.exp(
                    -(energy_prime - energy) / temp
                ):
                    sb_in = sb_prime_in.copy()
                    sb_out = sb_prime_out.copy()
                    wts[[e1, e2]] = wts[[e2, e1]]
                    energy = energy_prime
                    if energy < energymin:
                        energymin = energy
                        wtsmin = wts.copy()
                    naccept = naccept + 1
    
            # temperature update
            temp = temp * frac
            if verbose:
                print(
                    "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                    "frac of accepted moves {:.3f}".format(
                        istage, temp, energymin, naccept / niter
                    )
                )
    
        B = np.zeros((n, n))
        B[(u, v)] = wtsmin
    
        G_swap_annealing.add_edges_from([(nodes[e[0]], nodes[e[1]], {weight: B[e]}) for e in zip(*B.nonzero())])
    
    return G_swap_annealing
