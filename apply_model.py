import argparse, csv, pickle
import pandas as pd
import numpy as np
from sklearn import metrics

def get_ci(classifier, dataset, expected_map, features, traceback, l_limit, u_limit):

    test_set_list = dataset.values.T.tolist()

    if mis_dataset is None:
        variant_y_hat = classifier.predict_proba(dataset[features])

        smaller = 0
        greater = 0
        other = 0

        timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
        uncertain_log = 'uncertain_variants_' + timestamp + '.log' 
        with open(uncertain_log, 'a+') as uc_log:
            expected_list = []
            for i in range(0, len(variant_y_hat)):
                expected_list.append([expected_map[i][0], variant_y_hat[i][1]])
                if float(variant_y_hat[i][1]) < float(l_limit):
                    smaller += 1
                elif float(variant_y_hat[i][1]) > float(u_limit):
                    greater += 1
                else:
                    other += 1
                    line = ""
                    variant = getVariant(traceback, str(expected_map[i][0]), str(test_set_list[0][expected_map[i][1]]), str(test_set_list[2][expected_map[i][1]]))
                    line += variant
                    line += "\tScore:\t" + str(variant_y_hat[i][1]) + '\n'
                    uc_log.write(line)
                
        print "\nNum of scores less than " + str(l_limit) + ": " + str(smaller)
        print "Num of scores greater than " + str(u_limit) + ": " + str(greater)
        print "Num of scores " + str(l_limit) + " < & < " + str(u_limit) + ": " + str(other)

        dataset_y = dataset['Label']
        idx_true = np.where(dataset_y == 1)[0]
        idx_false = np.where(dataset_y == 0)[0]

        bins_list = [0.00, 0.05, 0.10, 0.15, 0.20,
                     0.25, 0.30, 0.35, 0.40, 0.45,
                     0.50, 0.55, 0.60, 0.65, 0.70,
                     0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.subplots_adjust(left=0.1)
        fig.subplots_adjust(top=0.9)
        n, bins, _ = ax.hist(variant_y_hat[idx_true, 1], bins=bins_list,
                             histtype='stepfilled', label='Positives')
        ax.hist(variant_y_hat[idx_false, 1], bins=bins_list,
                histtype='stepfilled', label='Artifacts')
        ax.set_xlabel('Classification score')
        ax.set_ylabel('Number of observations')

        plt.legend()
        plt_str = 'aiqc_confidence_interval_' + str(uuid.uuid4()) + '.png'
        plt.savefig(plt_str)
        print "\nGenerating confidence interval plot..."
        print "CI plot saved: " + plt_str

    else: #we've got some misclassifications
        mis_variant_y_hat = classifier.predict_proba(mis_dataset[['Coverage', 'Bias', 'VAF',
                                                                  'Control Sim1', 'Control Sim2',
                                                                  'Batch Sim']])

        new_map = ([mis_dataset['Coverage'].tolist(), mis_dataset['Bias'].tolist(), mis_dataset['VAF'].tolist()])

        mis_list = []
        for i in range(0, len(mis_variant_y_hat)):
            mis_list.append([misses_map[i][0], mis_variant_y_hat[i][1], new_map[0][i], new_map[1][i], new_map[2][i]])

        mis_list = sorted(mis_list, key=operator.itemgetter(1))

        smaller = 0
        smaller_list = []
        greater = 0
        greater_list = []
        other = 0
        other_list = []

        expected_list = []
        for i in range(0, len(mis_variant_y_hat)):
            expected_list.append([expected_map[i][0], mis_variant_y_hat[i][1]])
            if float(mis_variant_y_hat[i][1]) < float(l_limit):
                smaller += 1
                smaller_list.append(expected_map[i][0])
            elif float(mis_variant_y_hat[i][1]) > float(u_limit):
                greater += 1
                greater_list.append(expected_map[i][0])
            else:
                other += 1
                other_list.append(expected_map[i][0])

        print "\nNum of scores less than " + str(l_limit) + ": " + str(smaller)
        for l in smaller_list:
            print "\t" + l
        print "Num of scores greater than " + str(u_limit) + ": " + str(greater)
        for g in greater_list:
            print "\t" + g
        print "Num of scores " + str(l_limit) + " < & < " + str(u_limit) + ": " + str(other)
        for o in other_list:
            print "\t" + o

        mis_dataset_y = mis_dataset['Label']
        mis_idx_true = np.where(mis_dataset_y == 1)[0]
        mis_idx_false = np.where(mis_dataset_y == 0)[0]
        
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
        #ax.set_xticklabels(([0, 0], [1, 1]), minor=True)
        plt.xticks(minor_ticks_list)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.tick_params(axis='x', which='minor')
        plt.axvline(x=l_limit, linestyle='dashed', color='red')
        plt.axvline(x=u_limit, linestyle='dashed', color='red')
        
        if traceback is not None: #search for the variants in the traceback directory
            caption = ""
            for element in mis_list:
                #print (str(element[0]), str(element[2]), str(element[4]))
                variant = getVariant(traceback, str(element[0]), str(element[2]), str(element[4]))
                caption += "\t" + str(element[1]) + "\t" + variant + "\n"
        else:
            caption = ""
            for element in mis_list:
                caption += "\t" + str(element[1]) + ":\t" + str(element[0]) + "\t" + str(element[2]) + "\t" + str(element[3]) + "\t" + str(element[4]) + "\n"

        plt.legend(loc='best')
        plt_str = 'aiqc_confidence_interval_' + str(uuid.uuid4()) + '.png'
        plt.savefig(plt_str)
        print "\nGenerating confidence interval plot..."
        print "CI plot saved: " + plt_str
        print "\n\tCI Score\t\tVariant"
        print caption


def apply_model(picmod, test_set, test_samples, traceback, l_limit, u_limit):

    classifier = pickle.load(open(picmod, 'rb'))

    features = test_set.columns[:6]
    predicted = classifier.predict(test_set[features])
    
    expected = test_set['Label']
    expected_idx = test_set.index.values

    expected_map = []
    for sample in range(0, len(test_samples)):
        for idx in range(0, len(expected_idx)):
            if test_samples[sample][1] == str(expected_idx[idx]):
                expected_map.append([test_samples[sample][0], expected_idx[idx]])

    predicted_list = list(predicted)
    expected_list = list(expected)
    test_set_list = test_set.values.T.tolist()

    misses = []
    misses_map = []
    for i, prediction in enumerate(predicted_list):
        predicted_result = prediction
        expected_result = expected_list[i]
        if predicted_result != expected_result:
            print "misclassification case in sample " + expected_map[i][0]
            line = ""
            miss = []
            misses_map.append([expected_map[i][0]])
            for j, _ in enumerate(test_set_list):
                line += str(test_set_list[j][i])+"\t"
            print line

    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)

    #get_ci(classifier, test_set, expected_map, features, traceback, l_limit, u_limit)


def make_dataframe(csv_in):
    traceback = None
    variants = list(csv.reader(open(csv_in, "rU"), delimiter="\t"))

    if str(variants[0])[2:4] == "tb": #then this tsv has a traceback to it's features directory
        header = variants[1]
        begin = 2
        if str(variants[0])[-3] == "/":
            traceback = str(variants[0])[5:-3]
        else:
            traceback = str(variants[0])[5:-2]
    else:
        header = variants[0]
        begin = 1

    features = header[1:]
    m = len(features)

    samples = []
    variant_data = []
    variant_num = 0
    for variant in variants[begin:]:
        variant_metrics = [None] * m
        for i in range(m):
            variant_metrics[i] = float(variant[i+1])
        samples.append([variant[0], variant_num])
        variant_data.append(variant_metrics)
        variant_num += 1

    samples_map = np.array(samples)
    variant_data_array = np.array(variant_data)
    df = pd.DataFrame(variant_data_array, columns=features)
    
    return df, samples_map, traceback


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

    test_data, test_samples, traceback = make_dataframe(test_set)
    apply_model(pickle_model, test_data, test_samples, traceback, lower_limit, upper_limit)

if __name__ == '__main__':
    main()
