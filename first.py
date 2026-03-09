import numpy as np

def forward_selection(data, verbose = False):
    feature_count = data.shape[1] - 1
    selected_features = []
    unused = set(list(range(feature_count)))
    best_accuracy = 0.0
    best_feature_subset = []

    while len(selected_features) != feature_count:
        best_add, best_add_acc = None, -1.0
        for candidate_feature in unused:
            selected_features.append(candidate_feature)
            add_acc = accuracy(data, selected_features, best_add_acc)
            if add_acc > best_add_acc:
                best_add = candidate_feature
                best_add_acc = add_acc
            selected_features.pop()

        # Permanently add best one.
        selected_features.append(best_add)
        unused.remove(best_add)
        if best_add_acc > best_accuracy:
            best_feature_subset = selected_features[:]
            best_accuracy = best_add_acc

    print(f"The best feature subset that was found using forward selection was {[f + 1 for f in best_feature_subset]} with an accuracy of {best_accuracy}.")

def backward_elimination(data, verbose = False):
    feature_count = data.shape[1] - 1
    selected_features = set(range(feature_count))
    best_accuracy = accuracy(data, selected_features)
    best_feature_subset = list(selected_features)
    level = 1

    while len(selected_features) != 0:
        print(f"On the {level}th level of the search tree")
        best_remove, best_remove_acc = None, -1.0
        for candidate_feature in list(selected_features):
            print(f"--Considering removing feature {candidate_feature+1}")
            selected_features.remove(candidate_feature)
            remove_acc = accuracy(data, selected_features, best_remove_acc)
            if remove_acc > best_remove_acc:
                best_remove = candidate_feature
                best_remove_acc = remove_acc
            selected_features.add(candidate_feature)

        # Permanently remove.
        selected_features.remove(best_remove)
        if best_remove_acc > best_accuracy:
            best_feature_subset = list(selected_features)
            best_accuracy = best_remove_acc
        level += 1

    best_feature_subset.sort()
    print(f"The best feature subset that was found using backward elimination was {[f + 1 for f in best_feature_subset]} with an accuracy of {best_accuracy}.")

def accuracy(data, selected_features, best_accuracy=0.0) -> float:
    instance_count = data.shape[0]
    feature_indices = np.array(list(selected_features), dtype=int)

    correct = 0
    label_data = data[:, 0]
    features_data = data[:, 1:][:, feature_indices]

    for i in range(instance_count):
        best_label, min_distance = None, float("inf")
        test_label, test_vector = label_data[i], features_data[i]
        for j in range(instance_count):
            if i == j: continue
            train_label, train_vector = label_data[j], features_data[j] 
            dist = test_vector - train_vector #euclidean_distance(test_vector, train_vector)
            dist = np.dot(dist, dist)
            if dist < min_distance:
                    min_distance = dist
                    best_label = train_label
        if best_label == test_label:
            correct += 1
        else:
            # Try to prune
            if (correct + instance_count - i - 1) / instance_count <= best_accuracy:
                return 0.0
    return correct / instance_count


