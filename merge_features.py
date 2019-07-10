__author__ = 'markwelsh, DGD'

import csv
import os
import argparse
from argparse import RawTextHelpFormatter


def merge(filepaths, output_filename):

    features = ['Sample', 'Chrm', 'Pos', 'Alt', 'Coverage', 'Bias', 'VAF',
                'Control Sim1', 'Control Sim2', 'Batch Sim', 'Label']

    with open(output_filename, 'w') as outfile:
        outfile.write('\t'.join(features))
        outfile.write('\n')

        for fi in filepaths:
            with open(str(fi), 'r') as f:
                next(f)  # skip header
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 5:
                        print("WARNING: " + fi + " may not have features generated!")
                    if "NEG" in fi:
                        row.extend(str(0))  # label for NEG samples
                    elif "POS" in fi:
                        row.extend(str(1))  # label for POS samples
                    else:
                        raise ValueError(fi + ": Bad filename, cannot merge. " + 
                                         "Please use 'featuregeneration.py' to create feature files")
                    outfile.write('\t'.join(row))
                    outfile.write('\n')


def main():
    parser = argparse.ArgumentParser(description=('* Use this tool to stack TSV feature files into one file for RF input.'
                                                  '\n* Makefile compatible.\n* Will write .err file if errors occur.'),
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d, --directory",
                        help="Directory of feature files", dest="directory")
    parser.add_argument("-l, --list", 
                        help="A list of files, one feature file per line", dest="filelist")
    parser.add_argument("-o, --output_filename",
                        help="The file to write all merged features", dest="output_filename")

    args = parser.parse_args()

    directory = args.directory
    filelist = args.filelist
    output_filename = args.output_filename

    if directory is None and filelist is None:
        raise ValueError(
            '\nPlease provide a directory or list of files to merge. Use the --help flag for more info.\n')

    if directory is not None and filelist is not None:
        raise ValueError(
            '\nBoth options cannot be set. Please choose to use either the -d flag OR -l flag.\n')

    if directory is not None:
        if directory[-1:] != '/':
            directory = directory + '/'
        filepaths = []
        for _, _, files in os.walk(directory):
            for f in files:
                filepath = directory + f
                filepaths.append(filepath)

    if filelist is not None:
        filepaths = []
        with open(str(filelist)) as filepaths:
            for filepath in filepaths:
                filepaths.append(filepath.strip())

    merge(filepaths, output_filename)


if __name__ == '__main__':
    main()
