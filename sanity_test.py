from feature_selection import forward_selection, backward_elimination, accuracy
import numpy as np
import math

start_msgs = [
    "Sanity Check 1",
    "Sanity Check 2"
]
expected_ans = [
    "Expected Answer: Best Feature Subset = [7, 10, 12] with an accuracy of 0.950",
    "Expected Answer: Best Feature Subset = [2, 8, 10] with an accuracy of 0.960"
]

algo_names = [
    "Forward Selection",
    "Backwards Elimination" 
]

TEST_COUNT = 2


if __name__ == "__main__":

    tests_passed = 0

    # Load testing data
    data1 = np.loadtxt("sanity1.txt")
    data2 = np.loadtxt("sanity2.txt")
    data1[:, 0] = data1[:, 0].astype(int)
    data2[:, 0] = data2[:, 0].astype(int)
    data = (data1, data2)

    # These features are 1-indexed
    features = [[7, 10, 12], [2, 8, 10]]
    correct_accs = [0.950, 0.960]
    print("Accuracy Function Test")
    for i in range(2):
        print(f"\nOn Sanity Check {i+1} when only using features {features[i]} - ")
        acc = accuracy(data[i], features[i])
        print(f"Accuracy Function Found: {acc:.3f}")
        print(f"Expected Accuracy: {correct_accs[i]:.3f}")
        if math.isclose(acc, correct_accs[i]):
            print("Test Passed")
            tests_passed += 1
        else:
            print("Test Failed")

    if tests_passed == TEST_COUNT:
        print("\nAll Tests Passed!")
    exit()

    # Print out results
    for j, d in enumerate((data1, data2)):
        print(start_msgs[j])

        i = 0
        s, a = forward_selection(d)
        print(s, a)
        s, a = backward_elimination(d)
        print(s, a)
        """
        for subset, acc in (forward_selection(d, prune = True), backward_elimination(d, prune = True)):
            print(f"{algo_names[i]} found feature set: {subset}, Accuracy = {acc*100:.1f}%")
            i += 1
        """
        print(expected_ans[j])
        print("")


