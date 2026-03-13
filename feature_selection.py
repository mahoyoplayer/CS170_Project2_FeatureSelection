import numpy as np
import bisect

INDENT = "  "

def default_rate(data) -> float:
    # Class is located in first column
    label_data = data[:, 0]
    instance_count = data.shape[0]
    population = { 1: 0, 2: 0 }
    for label in label_data:
        population[label] += 1
    # Default rate = Percentage of most popular class
    return max(population.values()) / instance_count

def forward_selection(data, prune = True, verbose = False) -> list:
    instance_count, feature_count = data.shape[0], data.shape[1] - 1
    selected_features = [] # Start with using no features
    level = 1
    unused = set(range(1, feature_count+1))

    # Starting accuracy, subset
    best_accuracy = default_rate(data)
    best_feature_subset = []
    
    if verbose:
        print(f"The dataset has {feature_count} features (excluding class attribute) and {instance_count} instances.")
        print(f"Using no features, an accuracy of {best_accuracy*100:.1f}% is possible (default rate).\n")

    while len(selected_features) != feature_count:
        if verbose:
            print(f"Level {level} of search tree - ")

        best_add, best_add_acc = None, -1.0
        # Iterate through unused features
        for candidate_feature in unused:
            selected_features.append(candidate_feature)
            # Find accuracy with new addition of feature
            add_acc = accuracy(data, selected_features, best_add_acc if prune else None)
            # Compare vs highest accuracy found on current level
            if add_acc is not None and add_acc > best_add_acc:
                best_add = candidate_feature
                best_add_acc = add_acc
            if verbose:
                print(f"{INDENT}Using feature(s) {selected_features} ", end = "")
                if add_acc is not None:
                    print(f"gives an accuracy of {add_acc*100:.1f}%")
                else:
                    print("was pruned.")
            selected_features.pop()

        # Permanently add feature that resulted in best accuracy.
        bisect.insort(selected_features, best_add)
        unused.remove(best_add)
        # Compare accuracy with best global accuracy found so far.
        if best_add_acc > best_accuracy:
            best_feature_subset = selected_features[:]
            best_accuracy = best_add_acc
        if verbose:
            print(f"Feature set {selected_features} gives the best accuracy - {best_add_acc*100:.1f}%.\n")
        level += 1

    if verbose:
        print(f"The best feature subset that was found using forward selection was {best_feature_subset} with an accuracy of {best_accuracy*100:.1f}%.")
    return [best_feature_subset, best_accuracy]

def backward_elimination(data, prune = True, verbose = False) -> list:
    instance_count, feature_count = data.shape[0], data.shape[1] - 1
    # Set up 1-indexed
    selected_features = list(range(1, feature_count + 1))
    level = 1

    # Initial accuracy and subset
    best_accuracy = accuracy(data, selected_features)
    best_feature_subset = selected_features[:]

    if verbose:
        print(f"The dataset has {feature_count} features (excluding class attribute) and {instance_count} instances.")
        print(f"Using all {feature_count} features, I get an accuracy of {best_accuracy*100:.1f}%.\n")

    while len(selected_features) > 1:
        if verbose:
            print(f"Level {level} of search tree - ")
        best_remove, best_remove_acc = None, -1.0
        # Iterate through used features
        for candidate_feature in list(selected_features):
            selected_features.remove(candidate_feature)
            # Find accuracy with removal of feature
            remove_acc = accuracy(data, selected_features, best_remove_acc if prune else None)
            if verbose:
                print(f"{INDENT}Using feature(s) {[f + 1 for f in selected_features]} ", end = "")
                if remove_acc is not None:
                    print(f"gives an accuracy of {remove_acc*100:.1f}%")
                else:
                    print("was pruned.")
            # Compare vs highest accuracy found in this level
            if remove_acc is not None and remove_acc > best_remove_acc:
                best_remove = candidate_feature
                best_remove_acc = remove_acc
            bisect.insort(selected_features, candidate_feature)

        # Permanently remove.
        selected_features.remove(best_remove)
        if verbose:
            print(f"Feature set {selected_features} gives the best accuracy - {best_remove_acc*100:.1f}%.\n")
        # Compare vs highest global accuracy found so far
        if best_remove_acc > best_accuracy:
            best_feature_subset = list(selected_features)
            best_accuracy = best_remove_acc
        level += 1

    # Now test using no features.
    if (d_rate := default_rate(data)) > best_accuracy:
        best_accuracy = d_rate
        best_feature_subset = []
    
    best_feature_subset.sort()
    if verbose:
        print(f"Using no features, an accuracy of {d_rate*100:.1f}% is possible (default rate).\n")
        print(f"The best feature subset that was found using backward elimination was {best_feature_subset} with an accuracy of {best_accuracy*100:.1f}%.")
    return [best_feature_subset, best_accuracy]

# Returns accuracy of feature subset. Has ability to prune for speed if instructed.
def accuracy(data, selected_features, best_accuracy=None) -> float | None:
    # Adjust features from 1-indexed to 0-indexed
    features_data = data[:, 1:][:, [f - 1 for f in selected_features]]
    label_data = data[:, 0]
    instance_count = data.shape[0]
    correct = 0

    # Loop through all points
    for i in range(instance_count):
        best_label, min_distance = None, float("inf")
        test_label, test_vector = label_data[i], features_data[i]
        for j in range(instance_count):
            # Do not compare against same point
            if i == j: 
                continue
            train_label, train_vector = label_data[j], features_data[j] 
            # There is no need for square root in Euclid here, comparisons will not change.
            #euclid_dist = sum([x**2 for x in test_vector - train_vector])
            
            # DELETE START
            dist = train_vector - test_vector
            euclid_dist = np.dot(dist, dist)
            # DELETE END
            if euclid_dist < min_distance:
                min_distance = euclid_dist
                best_label = train_label
        # If labels match, algorithm was correct!
        if best_label == test_label:
            correct += 1
        else:
            # Prune if attaining better accuracy is no longer possible
            if best_accuracy is not None and (correct + instance_count - i - 1) / instance_count <= best_accuracy:
                return None
    return correct / instance_count