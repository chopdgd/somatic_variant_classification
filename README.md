# Somatic Variant Classification

## Usage

### Requirements
+ Python 2.7
+ tab-delimited files containing variants in format `Chr    Pos    Allele`

### Setup
Create a Python environment via pyenv by creating a new virtualenv, activating
it, and installing the provided requirements.
```shell
pyenv 2.7.10 virtualenv som-var-env
pyenv activate som-var-env
pip install -r requirements.txt
```

### Generating Features
In order to train/apply a Random Forest model, features need to be generated for
coverage, bias, VAF, similarity score vs a control sample, similarity score vs
another control samples, and similarity vs a intra-run batch sample. This
process can be automated using `auto_featuregeneration.sh`, a wrapper for the
`featuregeneration.py` script. Procuring the data needed for this script is the
only time-consuming/manual aspect of this workflow.

Once a datafile (e.g. `inputs.txt`) is in the proper tab separated format
`SAMPLE_ID CLASS_TYPE VARIANT_FILE SAMPLE_BAM NORMAL_BAMS BATCH_BAMS`, run:
```shell
./auto_featuregeneration.sh inputs.txt
```

This will create a `features/` directory and populate it with an updated
"VARIANT_FILE" that now includes features.

### Training a Model
While a single "VARIANT_FILE" can be used to a train a model, this is not very
useful. All the files in the new `features/` directory can be combined into one
text file using `merge_features.py`:

```shell
python merge_features.py --directory features/ --output_filename allfeatures.txt
```

`train_model.py` makes use of sklearn's RandomForestClassifier utility and
Pickle to save the trained model for later use. A single model can be created
with default arguments:
```shell
python train_model.py --training_data allfeatures.txt
```

A brief report will be output to the console displaying the parameters provided
to the classifier, the Pickle filename, as well as any variants that were
mislabeled by this newly trained model.

This tool also provides arguments for creating many models iterativly. This is
done by using a combination of the number of models to include in the Random 
forest, the depth of each tree in the forest, and weights for the positive and
negative classes. 

For example, over 1,000 models can be generated with one command:
```shell
python train_model.py \
  --training_data allfeatures.txt \
  --iterate \
  --trees 50 201 50 \
  --leaves 8 22 2 \
  --weights 1 101 10 1 101 10
```

### Applying a Model
A model that was saved can be loaded and applied to another dataset made with
`merge_features.py` by using `apply_model.py`:
```shell
python apply_model.py \
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

