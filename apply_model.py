import argparse, csv, pickle, datetime
import pandas as pd
import numpy as np
from sklearn import metrics

def get_ci(dataset, features, l_limit, u_limit):

    minor_ticks_list = [l_limit, 0.25, 0.5, 0.75, u_limit]
    minorLocator = MultipleLocator(5)
    bins_list = [0.00, 0.056, 0.112, 0.168, 0.224,
                 0.28, 0.336, 0.392, 0.448, 0.504,
                 0.56, 0.616, 0.672, 0.728, 0.784,
                 0.84, 0.896, 0.952, 1.008]
    
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(top=0.9)
    n, bins, _ = ax.hist(mis_variant_y_hat[mis_idx_true, 1], bins=bins_list,
                         histtype='stepfilled', label='Misclassified positives')
    ax.hist(mis_variant_y_hat[mis_idx_false, 1], bins=bins_list,
            histtype='stepfilled', label='Misclassified artifacts')
    ax.set_xlabel('Classification score')
    ax.set_ylabel('Number of observations')
    plt.xticks(minor_ticks_list)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.tick_params(axis='x', which='minor')
    plt.axvline(x=l_limit, linestyle='dashed', color='red')
    plt.axvline(x=u_limit, linestyle='dashed', color='red')
    
    if traceback is not None: #search for the variants in the traceback directory
        caption = ""
        for element in mis_list:
            variant = getVariant(traceback, str(element[0]), str(element[2]), str(element[4]))
            caption += "\t" + str(element[1]) + "\t" + variant + "\n"
    else:
        caption = ""
        for element in mis_list:
            caption += "\t" + str(element[1]) + ":\t" + str(element[0]) + "\t" + str(element[2]) + "\t" + str(element[3]) + "\t" + str(element[4]) + "\n"

    plt.legend(loc='best')
    plt_str = 'confidence_interval_' + str(uuid.uuid4()) + '.png'
    plt.savefig(plt_str)
    
    print "CI plot saved: " + plt_str
    print "\n\tCI Score\t\tVariant"
    print caption


def apply_model(picmod, test_set, l_limit, u_limit):

    classifier = pickle.load(open(picmod, 'rb'))

    features = ['Coverage', 'Bias', 'VAF', 'Control Sim1', 'Control Sim2', 'Batch Sim'] 
    expected = test_set['Label'].values
    predicted = classifier.predict(test_set[features])
    variant_y_hat = classifier.predict_proba(test_set[features])
    test_set['CI_Score'] = [x[1] for x in variant_y_hat]

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("\n")

    num_positives = len(test_set[test_set.CI_Score > u_limit])
    num_artifacts = len(test_set[test_set.CI_Score < l_limit]) 
    num_uncertain = len(test_set) - num_positives - num_artifacts
    print("Number of variants with CI scores > {0}:\t{1}".format(u_limit, num_positives))
    print("Number of variants with CI scores < {0}:\t{1}".format(l_limit, num_artifacts))
    print("Number of variants between {0} and {1}:\t{2}".format(l_limit, u_limit, num_uncertain))

    misclass_list = list()
    for i, (exp_class, pre_class) in enumerate(zip(expected, predicted)):
        if exp_class != pre_class:
            misclass = test_set.iloc[i].values.tolist()
            misclass_list.append(misclass)

    if misclass_list:
        misclass_df = pd.DataFrame(data=misclass_list, columns=test_set.columns)
        print("\n{0} misclassifications occurred:\n".format(len(misclass_list)))
        print(misclass_df.to_string())
    

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
