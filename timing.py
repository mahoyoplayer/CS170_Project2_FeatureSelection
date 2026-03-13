from feature_selection import backward_elimination, forward_selection
from time import perf_counter
import numpy as np

# We want to speed up
prune = True

data_small = "data120.txt"
data_large = "sanity1.txt"

if __name__ == "__main__":
    print("Program Start")
    small_data = np.loadtxt(data_small)

    # Small Dataset
    start = perf_counter()
    best_subset, best_acc = forward_selection(small_data, prune = prune)
    end = perf_counter()
    time_diff = end - start # in seconds
    print(f"Forward Selection on small dataset took {time_diff:.2f} seconds.")


    start = perf_counter()
    best_subset, best_acc = backward_elimination(small_data, prune = prune)
    end = perf_counter()
    time_diff = end - start # in seconds
    print(f"Backwards Elimination on small dataset took {time_diff:.2f} seconds.")

    # Large Dataset
    large_data = np.loadtxt(data_large)
    start = perf_counter()
    best_subset, best_acc = forward_selection(large_data, prune = prune)
    end = perf_counter()
    time_diff = end - start # in seconds
    print(f"Forward Selection on large dataset took {time_diff:.2f} seconds.")


    start = perf_counter()
    best_subset, best_acc = backward_elimination(large_data, prune = prune)
    end = perf_counter()
    time_diff = end - start # in seconds
    print(f"Backwards Elimination on large dataset took {time_diff:.2f} seconds.")


