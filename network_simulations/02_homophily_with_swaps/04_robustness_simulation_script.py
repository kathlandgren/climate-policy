import itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pickle
import os
import sys
import networkx as nx

# Get the current working directory
current_dir = os.getcwd()

# Navigate to 'network_simulations'
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Navigate to the sibling directory '00_opinion_function_setup'
sibling_dir = os.path.join(parent_dir, "00_opinion_function_setup")

# Add the sibling directory to sys.path for importing
if sibling_dir not in sys.path:
    sys.path.append(sibling_dir)

# Import the module from 00_opinion_function_setup
import opinion_functions as fun  
import generate_homophilic_graph_symmetric


# Parameters
homophilyvec = [0.75, 1, 0.5, 0.25, 0.]
num_agents_vec = [1000,10**4]
m_vec = [2, 5]
num_sim = 1000
minority_fraction = 0.33333

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists("output"):
        os.makedirs("output")

    # Create parameter grid
    param_grid = list(itertools.product(homophilyvec, m_vec, num_agents_vec))

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=75) as executor:
        futures = [
            executor.submit(fun.run_simulation_wrapper_with_swaps, (homophily, m, num_agents, num_sim, int(minority_fraction * num_agents), minority_fraction))
            for homophily, m, num_agents in param_grid
        ]

    for future in futures:
        future.result()
