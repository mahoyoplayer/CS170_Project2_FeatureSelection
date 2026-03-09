from first import forward_selection, backward_elimination
import numpy as np


data = np.loadtxt("SanityCheckDataSet__2.txt")
#acc1 = backward_elimination(data)
#print(acc1)
print("Sanity Check 1")
print()
print("Expected Answer: Best Feature Subset = [7, 10, 12] with an accuracy of 0.950")

print("Sanity Check 2")
print("Expected Answer: Best Feature Subset = [2, 8, 10] with an accuracy of 0.960")

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

a = backward_elimination(data)
print(a)

profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)