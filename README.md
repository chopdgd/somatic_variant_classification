# Identification of True Somatic Variants Using Machine Learning

## Usage

### Requirements

#### Environment
+ Python 2.7
+ [virtualenv](https://virtualenv.pypa.io/en/latest/installation/)

### Per Sample
+ two tab-delimited files containing variants in format `Chr    Pos    Allele`,
one file containing positive variants and another containing artifacts
+ a sorted sample BAM and index file
+ two sorted negative-control BAM files and their respective index files
+ at least one other sorted sample BAM file sequenced in the same batch

### Setup
Create a Python environment via virtualenv, activating it, and installing 
the provided requirements.
```console
$ virtualenv --python=python2.7 som-var-env
$ source som-var-env/bin/activate
$ pip install -r requirements.txt
```

### Generating Features
In order to train/apply a Random Forest model, features need to be generated for
coverage, bias, VAF, similarity score vs two control samples, and similarity vs a intra-run batch sample. This
process can be automated using `auto_featuregeneration.sh`, a wrapper for the
`featuregeneration.py` script. 

Once a datafile (e.g. `inputs.txt`) is in the proper tab separated format
`SAMPLE_ID CLASS_TYPE VARIANT_FILE SAMPLE_BAM NORMAL_BAMS BATCH_BAMS`, run:
```console
$ ./auto_featuregeneration.sh inputs.txt
```

This will create a `features/` directory and populate it with an updated
"VARIANT_FILE" that now includes features.

#### Example
For two samples sequenced in the same batch, "TEST01" and "TEST02", `inputs.txt` would contain 4 lines:
```
TEST01  NEG test01_neg.txt  test01.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
TEST01  POS test01_pos.txt  test01.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
TEST02  NEG test02_neg.txt  test02.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
TEST02  POS test02_pos.txt  test02.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
```

If "TEST01" and "TEST02" were sequenced in _different_ batches, `inputs.txt`
would change for the BATCH_BAMS field:
```
TEST01  NEG test01_neg.txt  test01.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
TEST01  POS test01_pos.txt  test01.bam  ctrl1.bam,ctrl2.bam batchX-1.bam,batchX-2.bam
TEST02  NEG test02_neg.txt  test02.bam  ctrl1.bam,ctrl2.bam batchY-1.bam,batchY-2.bam
TEST02  POS test02_pos.txt  test02.bam  ctrl1.bam,ctrl2.bam batchY-1.bam,batchY-2.bam
```

### Training a Model
While a single "VARIANT_FILE" can be used to a train a model, this is not very
useful. All the files in the new `features/` directory can be combined into one
text file using `merge_features.py`:

```console
$ python merge_features.py --directory features/ --output_filename allfeatures.txt
```

`train_model.py` makes use of sklearn's
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) utility and
Pickle to save the trained model for later use. A single model can be created
with default arguments:
```console
$ python train_model.py --training_data allfeatures.txt
```

A brief report will be output to the console displaying the parameters provided
to the classifier, the Pickle filename, as well as any variants that were
mislabeled by this newly trained model.

This tool also provides arguments for creating many models iteratively. This is
done by using a combination of a) the number of models to include in the Random 
forest, b) the depth of each tree in the forest, and c) weights for the positive
and negative classes. 

For example, over 1,000 models can be generated with one command:
```console
$ python train_model.py \
    --training_data allfeatures.txt \
    --iterate \
    --trees 50 201 50 \
    --leaves 8 22 2 \
    --weights 1 101 10 1 101 10
```

### Applying a Model
A model that was saved can be loaded and applied to another dataset made with
`merge_features.py` by using `apply_model.py`:
```console
$ python apply_model.py \
    --pickle_model mymodel.sav \
    --test_set mytestset.tsv \
    --lower_limit 0.05 \
    --upper_limit 0.90
```
where `--lower_limit` and `--upper_limit` are the boundaries of the "uncertain"
region. Ideally, no misclassifications occur outside the region of uncertainty.
This can be a major criteria in model selection.

This tool will output a histogram of confidence interval scores for all variants
in the `--test_set` as well as any misclassifications, if applicable. 

