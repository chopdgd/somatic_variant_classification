#!/bin/bash

# this wrapper script expects one tab delimited input file in the format:
# SAMPLE_ID CLASS_TYPE VARIANT_FILE SAMPLE_BAM NORMAL_BAMS BATCH_BAMS

# the python command is also output to stdout for denbugging/logging purposes

PSEUDO="combined_pseudoregions.txt"

IFS=$'\t'
while read -r SAMPLE_ID CLASS_TYPE VARIANT_FILE SAMPLE_BAM NORMAL_BAMS BATCH_BAMS; do

    echo "Generating features for ${SAMPLE_ID}, ${CLASS_TYPE} with cmd:"
    echo -e "\tpython featuregeneration.py"
    echo -e "\t\t--input_bam ${SAMPLE_BAM}"
    echo -e "\t\t--input_variants ${VARIANT_FILE}"
    echo -e "\t\t--sample_id ${SAMPLE_ID}"
    echo -e "\t\t--class_type ${CLASS_TYPE}"
    echo -e "\t\t--pseudoregions_file ${PSEUDO}"
    echo -e "\t\t--normal_control_bams ${NORMAL_BAMS}"
    echo -e "\t\t--batch_control_bams ${BATCH_BAMS}"

    python featuregeneration.py \
        --input_bam ${SAMPLE_BAM} \
        --input_variants ${VARIANT_FILE} \
        --sample_id ${SAMPLE_ID} \
        --class_type ${CLASS_TYPE} \
        --pseudoregions_file ${PSEUDO} \
        --normal_control_bams ${NORMAL_BAMS} \
        --batch_control_bams ${BATCH_BAMS}

    status=$?

    if [ $status -eq 0 ]; then
        echo -e "\tSuccess!"
    fi

done < ${1}

mkdir features/
mv *.features.txt features/
