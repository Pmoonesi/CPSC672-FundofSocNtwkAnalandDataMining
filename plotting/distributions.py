import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def compare_dists(G1, G2, name, deg_ext=None, bins=9, weight=None):

    nodes1, edges1 = G1.number_of_nodes(), G1.number_of_edges()
    nodes2, edges2 = G2.number_of_nodes(), G2.number_of_edges()

    in_degrees1 = [deg for (id, deg) in G1.in_degree(weight=weight) if deg > 0]
    out_degrees1 = [deg for (id, deg) in G1.out_degree(weight=weight) if deg > 0]
    
    in_degrees2 = [deg for (id, deg) in G2.in_degree(weight=weight) if deg > 0]
    out_degrees2 = [deg for (id, deg) in G2.out_degree(weight=weight) if deg > 0]

    if deg_ext:
        in_kmin, in_kmax, out_kmin, out_kmax = deg_ext
    else:
        in_kmin = min(min(in_degrees1), min(in_degrees2))
        in_kmax = max(max(in_degrees1), max(in_degrees2))
        out_kmin = min(min(out_degrees1), min(out_degrees2))
        out_kmax = max(max(out_degrees1), max(out_degrees2))

    # Get 10 logarithmically spaced bins between kmin and kmax
    in_bin_edges_log = np.logspace(np.log10(in_kmin), np.log10(in_kmax), num=bins + 1)
    out_bin_edges_log = np.logspace(np.log10(out_kmin), np.log10(out_kmax), num=bins + 1)

    # histogram the data into these bins
    in_density_log1, _ = np.histogram(in_degrees1, bins=in_bin_edges_log, density=True)    
    out_density_log1, _ = np.histogram(out_degrees1, bins=out_bin_edges_log, density=True)
    in_density_log2, _ = np.histogram(in_degrees2, bins=in_bin_edges_log, density=True)    
    out_density_log2, _ = np.histogram(out_degrees2, bins=out_bin_edges_log, density=True)
    
    # "x" should be midpoint (IN LOG SPACE) of each bin
    in_log_be_log = np.log10(in_bin_edges_log)
    in_x_log = 10**((in_log_be_log[1:] + in_log_be_log[:-1])/2)
    
    out_log_be_log = np.log10(out_bin_edges_log)
    out_x_log = 10**((out_log_be_log[1:] + out_log_be_log[:-1])/2)

    # Get 20 logarithmically spaced bins between kmin and kmax
    in_bin_edges = np.linspace(in_kmin, in_kmax, num=bins + 1)
    out_bin_edges = np.linspace(out_kmin, out_kmax, num=bins + 1)
    
    # histogram the data into these bins
    in_density1, _ = np.histogram(in_degrees1, bins=in_bin_edges, density=True)
    out_density1, _ = np.histogram(out_degrees1, bins=out_bin_edges, density=True)
    in_density2, _ = np.histogram(in_degrees2, bins=in_bin_edges, density=True)
    out_density2, _ = np.histogram(out_degrees2, bins=out_bin_edges, density=True)

    # "x" should be midpoint (IN LOG SPACE) of each bin
    in_log_be = np.log10(in_bin_edges)
    in_x = 10**((in_log_be[1:] + in_log_be[:-1])/2)
    
    out_log_be = np.log10(out_bin_edges)
    out_x = 10**((out_log_be[1:] + out_log_be[:-1])/2)

    fig, axis = plt.subplots(2, 2, figsize=(12,9))

    fig.suptitle(f"{name}{f' - {weight}' if weight is not None else ''}: nodes = {nodes1}/{nodes2}, edges = {edges1}/{edges2}")

    axis = axis.ravel()

    axis[0].loglog(in_x_log, in_density_log1, marker='o', linestyle='none', c='b', label='G1')
    axis[0].loglog(in_x_log, in_density_log2, marker='o', linestyle='none', c='y', label='G2')
    axis[0].set_xlabel(r"in degree $k$ - loglog", fontsize=16)
    axis[0].set_ylabel(r"$P(k)$", fontsize=16)
    
    axis[1].loglog(out_x_log, out_density_log1, marker='o', linestyle='none', c='b', label='G1')
    axis[1].loglog(out_x_log, out_density_log2, marker='o', linestyle='none', c='y', label='G2')
    axis[1].set_xlabel(r"out degree $k$ - loglog", fontsize=16)
    axis[1].set_ylabel(r"$P(k)$", fontsize=16)
    
    axis[2].plot(in_x, in_density1, marker='o', linestyle='none', c='b', label='G1')
    axis[2].plot(in_x, in_density2, marker='o', linestyle='none', c='y', label='G2')
    axis[2].set_xlabel(r"in degree $k$", fontsize=16)
    axis[2].set_ylabel(r"$P(k)$", fontsize=16)
    
    axis[3].plot(out_x, out_density1, marker='o', linestyle='none', c='b', label='G1')
    axis[3].plot(out_x, out_density2, marker='o', linestyle='none', c='y', label='G2')
    axis[3].set_xlabel(r"out degree $k$", fontsize=16)
    axis[3].set_ylabel(r"$P(k)$", fontsize=16)
    
    # save the plot
    plt.tight_layout()
    plt.legend()
    plt.show()
    # fig.savefig(f"figs3/{name.split('.')[0]}.png")
    # plt.close()
