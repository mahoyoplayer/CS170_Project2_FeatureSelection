import numpy as np
import os
from first import forward_selection, backward_elimination

if __name__ == "__main__":
    print("Feature Selection Program using Nearest Neighbor")

    while True:
        file_name = input("Enter name of file to test: ")
        if not os.path.isfile(file_name):
            print(f"Could not find {file_name}. Please enter a valid file name.\n")
        else:
            data = np.loadtxt(file_name)
            break

    INDENT = "  "
    print("Choose a Search Algorithm (1, 2)")
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
    if algo_choice == 1:
        forward_selection(data, verbose = True)
    else:
        backward_elimination(data, verbose = True)