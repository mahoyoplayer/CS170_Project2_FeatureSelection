import numpy as np
import os
from feature_selection import forward_selection, backward_elimination
from time import perf_counter

INDENT = "  "
PRUNE = True # Can vastly speed up search if set to True

if __name__ == "__main__":
    print("Feature Selection Program using Nearest Neighbor")

    # Get name of file to use
    while True:
        file_name = input("Enter name of file to test: ")
        if not os.path.isfile(file_name):
            print(f"Could not find {file_name}. Please enter a valid file name.\n")
        else:
            data = np.loadtxt(file_name)
            # The class column should be integers.
            data[:, 0] = data[:, 0].astype(int)
            break

    # Get choice of search algorithm
    print("\nChoose a Search Algorithm (1, 2)")
    print(f"{INDENT}1) Forward Selection")
    print(f"{INDENT}2) Backward Elimination")
    algo_choice = None
    while True:
        algo_choice = input("Answer: ")
        if not (algo_choice.isdigit() and int(algo_choice) in [1, 2]):
            print('Invalid response. Choose either "1" or "2".\n')
        else:
            algo_choice = int(algo_choice)
            break

    print("")
    start = perf_counter()

    if algo_choice == 1:
        forward_selection(data, prune = PRUNE, verbose = True)
    else:
        backward_elimination(data, prune = PRUNE, verbose = True)

    end = perf_counter()
    time_diff = end - start # in seconds
    print(f"\nTotal time taken: {time_diff:.2f} seconds")
