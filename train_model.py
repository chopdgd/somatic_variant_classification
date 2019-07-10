__author__ = 'chaowu, DGD'

import argparse
import csv
import datetime
import operator
import pickle
import uuid
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_model(dataset, neg_w=1, pos_w=10, num_trees=100, num_leaves=10):

    features = dataset.columns[:6]
    dataset_y = dataset['Label']

    clf = RandomForestClassifier(n_jobs=2, n_estimators=num_trees, criterion='entropy',
                                 max_depth=num_leaves, class_weight={0: neg_w, 1: pos_w})
    clf.fit(dataset[features], dataset_y)

    picmod = "som_var_model_" + str(uuid.uuid4()) + ".sav"
    pickle.dump(clf, open(picmod, 'wb'))

    predicted = clf.predict(test_set[features])
    expected = test_set['Label']
    expected_idx = test_set.index.values

    expected_map = []
    for sample in range(0, len(test_samples)):
        for idx in range(0, len(expected_idx)):
            if test_samples[sample][1] == str(expected_idx[idx]):
                expected_map.append(
                    [test_samples[sample][0], expected_idx[idx]])

    model = "RandomForestClassifier(n_estimators=" \
        + str(num_trees) + ", criterion=entropy, max_depth=" + str(num_leaves) \
        + ", class_weight={0:" + str(neg_w) + ", 1:" + str(pos_w) + "})"
    print "**********************************************"
    print arg
    print "Pickle model: " + picmod
    print "**********************************************"

    misses = []
    misses_map = []
    for i, _ in enumerate(predicted_list):
        predicted_result = predicted_list[i]
        expected_result = expected_list[i]
        if predicted_result != expected_result:
            print "misclassification case in sample " + expected_map[i][0]
            line = ""
            miss = []
            misses_map.append([expected_map[i][0]])
            for j, _ in enumerate(test_set_list):
                line += str(test_set_list[j][i])+"\t"
                miss.append(test_set_list[j][i])
            print line
            misses.append(miss)

    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)


def main():

    parser = argparse.ArgumentParser(description='RandomForest classification')
    parser.add_argument("--training_data", help="Training data; stacked .txt (tsv) file",
                        dest="training_data", required=True)
    parser.add_argument("--iterate", help="Iterative option; boolean",
                        dest="iterate", action='store_true')
    parser.add_argument("--trees",
                        help="Number of trees in forest; int or [start stop step] e.g. 25 126 50",
                        dest="trees", nargs='+', default=100)
    parser.add_argument("--leaves",
                        help="Number of leaves/depth; int or [start stop step] e.g. 8 13 2",
                        dest="leaves", nargs='+', default=10)
    parser.add_argument("--weights",
                        help="Class weights iterr variable; \
                        ratio [neg_start neg_stop neg_step pos_start pos_stop pos_step] \
                        e.g. 1 102 10 1 102 10",
                        dest="w", nargs='+', default=[1, 1, 1, 10, 10, 1])

    args = parser.parse_args()

    training_data = args.training_data
    iterate = args.iterate
    trees = args.trees
    leaves = args.leaves
    weights = args.weights

    # input checking & error handling
    if iterate is True:
        if isinstance(trees, list) is False and isinstance(leaves, list) is False:
            raise ValueError(
                'Lists are needed for both --trees and --leaves if the iterate flag is set')
        else:
            if len(leaves) != 3 or len(trees) != 3 or len(weight) != 6:
                raise ValueError(
                    'Please input 3 parameters for --trees and --leaves and 6 parameters to --weights to create a valid loop')
            else:
                for num_trees in range(trees[0], trees[1], trees[2]):
                    for num_leaves in range(leaves[0], leaves[1], leaves[2]):
                        for neg_w in range(weights[0], weights[1], weights[2]):
                            for pos_w in range(weights[3], weights[4], weights[5]):
                                train_model(training_data, neg_w,
                                            pos_w, num_trees, num_leaves)
    else:
        train_model(dataset)


if __name__ == "__main__":
    main()
