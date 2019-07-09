import argparse, pickle, uuid
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits import mplot3d
from sklearn import metrics


def graph_ci(dataset, l_limit, u_limit):

    labeled_positives = dataset[dataset["Label"] == 1]["CI_Score"].values
    labeled_artifacts = dataset[dataset["Label"] == 0]["CI_Score"].values

    minor_ticks_list = [l_limit, 0.25, 0.5, 0.75, u_limit]
    minorLocator = MultipleLocator(5)
    ci_bins_list = [0.00, 0.056, 0.112, 0.168, 0.224,
                    0.28, 0.336, 0.392, 0.448, 0.504,
                    0.56, 0.616, 0.672, 0.728, 0.784,
                    0.84, 0.896, 0.952, 1.008]
    
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(top=0.9)
    n, bins, _ = ax.hist(labeled_positives, bins=ci_bins_list,
                         histtype='stepfilled', label='Labeled true positives')
    ax.hist(labeled_artifacts, bins=ci_bins_list,
            histtype='stepfilled', label='Labeled artifacts')
    ax.set_xlabel('Classification score')
    ax.set_ylabel('Number of observations')
    plt.xticks(minor_ticks_list)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.tick_params(axis='x', which='minor')
    plt.axvline(x=l_limit, linestyle='dashed', color='red')
    plt.axvline(x=u_limit, linestyle='dashed', color='red')
    
    plt.legend(loc='best')
    plt_str = 'confidence_interval_' + str(uuid.uuid4()) + '.png'
    plt.savefig(plt_str) 
    print("CI plot saved successfully: {0}".format(plt_str))


def apply_model(picmod, test_set, l_limit, u_limit):

    classifier = pickle.load(open(picmod, 'rb'))

    features = ['Coverage', 'Bias', 'VAF', 'Control Sim1', 'Control Sim2', 'Batch Sim'] 
    expected = test_set['Label'].values
    predicted = classifier.predict(test_set[features])
    variant_y_hat = classifier.predict_proba(test_set[features])
    test_set['CI_Score'] = [x[1] for x in variant_y_hat]

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    num_positives = len(test_set[test_set.CI_Score > u_limit])
    num_artifacts = len(test_set[test_set.CI_Score < l_limit]) 
    num_uncertain = len(test_set) - num_positives - num_artifacts
    print("Number of variants with CI scores > {0}:\t{1}".format(u_limit, num_positives))
    print("Number of variants with CI scores < {0}:\t{1}".format(l_limit, num_artifacts))
    print("Number of variants between {0} and {1}:\t{2}".format(l_limit, u_limit, num_uncertain))

    graph_ci(test_set, l_limit, u_limit)

    misclass_list = list()
    for i, (exp_class, pre_class) in enumerate(zip(expected, predicted)):
        if exp_class != pre_class:
            misclass = test_set.iloc[i].values.tolist()
            misclass_list.append(misclass)

    if misclass_list:
        misclass_df = pd.DataFrame(data=misclass_list, columns=test_set.columns)
        misclass_df.sort_values(by=['CI_Score'], inplace=True)
        print("\n{0} misclassifications occurred:\n".format(len(misclass_list)))
        print(misclass_df.to_string())
        graph_ci(misclass_df, l_limit, u_limit) 


def main():
    parser = argparse.ArgumentParser(description='RandomForest classification')

    parser.add_argument("--pickle_model", 
                        help="Pickle classifier; .sav file", 
                        dest="pickle_model",
                        required=True)
    parser.add_argument("--test_set", 
                        help="Test data; tsv file", 
                        dest="test_set",
                        required=True)
    parser.add_argument("--lower_limit",
                        help="Lower-limit for CI interval (exclusive); float",
                        dest="lower_limit",
                        default=0.05)
    parser.add_argument("--upper_limit",
                        help="Upper-limit for CI interval (exclusive); float",
                        dest="upper_limit",
                        default=0.90)

    args = parser.parse_args()

    pickle_model = args.pickle_model
    test_set = args.test_set
    lower_limit = args.lower_limit
    upper_limit = args.upper_limit

    test_df = pd.read_csv(test_set, sep='\t', header=0, index_col=False)
    apply_model(pickle_model, test_df, lower_limit, upper_limit)

if __name__ == '__main__':
    main()
