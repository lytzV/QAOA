import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import nlopt
import networkx as nx
import qiskit
from qiskit import BasicAer
from qiskit.optimization.ising import max_cut
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.optimizers.nlopts.esch import ESCH
from qiskit.aqua.algorithms import QAOA
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator
import pandas as pd

from IPython.display import clear_output
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

from itertools import combinations

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def generate_random_graph(graph_model, v_num, edges_or_prob, seed=0):
    G = graph_model(v_num, edges_or_prob, seed)
    colors = ['r' for node in G.nodes()]
    pos = nx.spring_layout(G)
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    plt.show()
    return G
def brute_optimize(v_num, adj):
    best_cost = 0
    ctr = 0
    for b in range(2**v_num):
        x = [int(t) for t in list(bin(b)[2:].zfill(v_num))]
        cost = 0
        for i in range(v_num):
            for j in range(v_num):
                ctr += 1
                cost = cost + adj[i, j]*x[i]*(1-x[j])
                update_progress(ctr / (v_num*v_num*(2**v_num)))
        if best_cost < cost:
            best_cost = cost
            x_best = x
    update_progress(1)
    print('Best solution = ' + str(x_best) + '\nBest profit = ' + str(best_cost))
    return x_best, best_cost

### Preparing for QAOA simulation ###
# Compute the tensor product pauli gates == (g_i tensor g_j)
def QAOA_pauli(adj):
    v_num = adj.shape[0]
    pauli_list = []
    for i in range(v_num):
        for j in range(i):
            if adj[i, j] != 0:
                # Only compute tensor product pauli gates for existing edges
                # Max-Cut formulation requires Z gates, so zero-vec X here to leave Z only
                xp = np.zeros(v_num, dtype=np.bool) 
                zp = np.zeros(v_num, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                # Max-Cut formulation has a 0.5 weight (no effect on optimization)
                pauli_list.append([0.5 * adj[i, j], Pauli(zp, xp)])
    qubitOp = WeightedPauliOperator(paulis=pauli_list)
    return qubitOp

# Simulate in QAOA and retrieve approximated optimizer
def simulate_optimize(n_shots, p_steps, qubitOp, G):
    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=n_shots)

    qaoa = QAOA(qubitOp, ESCH(max_evals=100), p=p_steps)
    result = qaoa.run(quantum_instance)

    # QAOA converts the problem into finding the maximum eigenval/eigenvec pair
    solution = max_cut.sample_most_likely(result['eigvecs'][0]) #returns vector with highest counts
    #print('energy:', result['energy'])
    print('solution:', solution)
    
    # For p steps, there should be 2*p parameters (beta,gamma)
    #print('optimal parameters', result['opt_params'])
    cost = 0
    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            cost = cost + adj[i, j]*solution[i]*(1-solution[j])
    print("Sample profit = {}".format(cost))
    
    return result, solution

def calc_profit(cuts_qaoa, maxcut_graph):
    sub_lists = []
    for i in range(0, len(maxcut_graph.nodes())+1):
        temp = [list(x) for x in combinations(maxcut_graph.nodes(), i)]
        sub_lists.extend(temp)
    
    # Calculate the cut_size for all possible cuts
    cut_size = []
    for sub_list in sub_lists:
        cut_size.append(nx.algorithms.cuts.cut_size(maxcut_graph,sub_list))

    # Calculate the cut_size for the cuts found with QAOA
    cut_size_qaoa = []
    for cut in cuts_qaoa:
        cut_size_qaoa.append(nx.algorithms.cuts.cut_size(maxcut_graph,cut))
        
    print("The average QAOA approximation profit = {} is {}% of the max profit = {}".format(np.mean(cut_size_qaoa), np.mean(cut_size_qaoa)*100/np.max(cut_size), np.max(cut_size)))
def parse_bit_to_cut(bit_strings, tfq=True):
    cuts_qaoa = []
    for bit_string in bit_strings:
        temp = []
        for pos, bit in enumerate(bit_string):
            if not tfq:
                if bit=='1':
                    temp.append(pos)
            else:
                if bit==1:
                    temp.append(pos)
        cuts_qaoa.append(temp)
    return cuts_qaoa
def tfq_optimize(maxcut_graph, epochs = 500, n_shot = 512):
    """
        Complete credit to Google https://github.com/tensorflow/quantum/blob/research/qaoa/qaoa.ipynb
    """
    
    # define 10 qubits
    cirq_qubits = cirq.GridQubit.rect(1, len(maxcut_graph.nodes))
    
    # create layer of hadamards to initialize the superposition state of all computational states
    hadamard_circuit = cirq.Circuit() 
    for node in maxcut_graph.nodes():
        qubit = cirq_qubits[node] 
        hadamard_circuit.append(cirq.H.on(qubit))
    
    # define the two parameters for one block of QAOA
    qaoa_parameters = sympy.symbols('a b')
    
    # define the the mixing and the cost Hamiltonians
    mixing_ham = 0
    for node in maxcut_graph.nodes(): 
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.X(qubit) )
    cost_ham = maxcut_graph.number_of_edges()/2
    for edge in maxcut_graph.edges(): 
        qubit1 = cirq_qubits[edge[0]]
        qubit2 = cirq_qubits[edge[1]]
        cost_ham += cirq.PauliString(1/2*(cirq.Z(qubit1)*cirq.Z(qubit2)))
        
    qaoa_circuit = tfq.util.exponential(operators = [cost_ham , mixing_ham], coefficients = qaoa_parameters)
    
    # define the model and training data
    model_circuit , model_readout = qaoa_circuit , cost_ham
    input_ = [hadamard_circuit]
    input_ = tfq.convert_to_tensor(input_)
    optimum = [0]
    
    # Build the Keras model.
    print("Training...")
    optimum=np.array(optimum)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string))
    model.add(tfq.layers.PQC(model_circuit, model_readout))
    model.compile(loss=tf.keras.losses.mean_absolute_error,
              optimizer=tf.keras.optimizers.Adam())
    history = model.fit(input_,optimum,epochs=epochs,verbose=1)
    
    # Visualizing
    plt.plot(history.history['loss'])
    plt.title("QAOA with TFQ")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    
    # Read out the optimal paramters and sample from the final state n_shot times
    print("Sampling")
    # Adding the qaoa circuit after the hadamard circuit
    params = model.trainable_variables
    add = tfq.layers.AddCircuit()
    output_circuit = add(input_, append=qaoa_circuit)

    sample_layer = tfq.layers.Sample()
    output = sample_layer(output_circuit, symbol_names = qaoa_parameters, symbol_values = params,repetitions=n_shot)
    
    print("Calculating approximation ratio...")
    # Translate output in cut sets
    cuts_qaoa = parse_bit_to_cut(output.values)
    calc_profit(cuts_qaoa, maxcut_graph)




if __name__ == "__main__":
    G = nx.random_regular_graph(3,10)
    adj = nx.to_numpy_matrix(G)
    tfq_optimize(G,1000,1000)