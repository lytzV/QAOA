from collections import defaultdict
import numpy as np
from itertools import combinations
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from matplotlib import pyplot as plt
import networkx as nx


def brute_best(maxcut_graph):
    sub_lists = []
    for i in range(0, len(maxcut_graph.nodes())+1):
        temp = [list(x) for x in combinations(maxcut_graph.nodes(), i)]
        sub_lists.extend(temp)
    
    # Calculate the cut_size for all possible cuts
    cut_size = []
    for sub_list in sub_lists:
        cut_size.append(nx.algorithms.cuts.cut_size(maxcut_graph,sub_list))
    return cut_size
def generate_graph(model, a, b):
    model_2_param = {
        nx.generators.community.connected_caveman_graph:"You are generating a caveman graph with {} cliques of size {}. This simulates families with representative outreach.".format(a, b),
        nx.generators.geometric.random_geometric_graph:"You are generating a geometric graph with {} nodes of radius at most {}. This simulates interactions within a geographical radius.".format(a, b),
        nx.generators.random_graphs.dense_gnm_random_graph:"You are generating a dense graph with {} nodes and {} edges. This simulates a dense gathering".format(a,b)
    }
    print(model_2_param[model])
    G = model(a,b)
    adj = nx.to_numpy_matrix(G)
    max_cut = max(brute_best(G))
    return G, adj, max_cut
def dwave_solver(G, chainstrength = 2, numruns = 10):
    # ------- Set up our QUBO dictionary -------
    # Initialize our h vector, J matrix
    h = defaultdict(int)
    J = defaultdict(int)

    # Update J matrix for every edge in the graph
    for u, v in G.edges:
        J[(u,v)]+= 1

    # ------- Run our QUBO on the QPU -------
    # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_ising(h, J, chain_strength=chainstrength, num_reads=numruns)
    energies = iter(response.data())

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
    print('-' * 60)
    for line in response:
        S0 = [k for k,v in line.items() if v == -1]
        S1 = [k for k,v in line.items() if v == 1]
        E = next(energies).energy
        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int((6-E)/2))))

    # ------- Display results to user -------
    # Grab best result
    # Note: "best" result is the result with the lowest energy
    # Note2: the look up table (lut) is a dictionary, where the key is the node index
    #   and the value is the set label. For example, lut[5] = 1, indicates that
    #   node 5 is in set 1 (S1).
    lut = response.lowest().first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if lut[node]==-1]
    S1 = [node for node in G.nodes if lut[node]==1]
    cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
    uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

    # Display best result
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(G, pos)

    filename = "maxcut_plot_ising.png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))



if __name__ == "__main__":
    G, adj, max_cut = generate_graph(nx.generators.random_graphs.dense_gnm_random_graph, 8, 16)
    dwave_solver(G)