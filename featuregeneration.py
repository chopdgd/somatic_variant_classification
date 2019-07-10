__author__ = 'chaowu, DGD'

import pysam
import csv
import argparse
import math


def mean(numbers):
    return round(float(sum(numbers)) / max(len(numbers), 1), 2)


def calc_bias(forward_strand, reverse_strand):
    if forward_strand > 0 or reverse_strand > 0:
        return round(float(abs(reverse_strand - forward_strand)) / float((reverse_strand + forward_strand)), 2)
    else:
        return float(0)


def vaf_value(all_coverage, alt_coverage):
    if all_coverage > 0:
        vaf = round(float(float(alt_coverage) / float(all_coverage)), 2)
    else:
        vaf = 0.0
    return vaf


def calc_sim(pa_vector, norm_vector):
    similarity = float(0)
    vec1 = list(pa_vector)
    vec2 = list(norm_vector)
    if len(vec1) == len(vec2):
        i = 0
        while i < len(vec1):
            vec1_value = float(vec1[i])
            vec2_value = float(vec2[i])
            if max(vec1_value, vec2_value) > 0:
                if vec1_value >= vec2_value:
                    vec1[i] = float(vec1_value/vec1_value)
                    vec2[i] = float(vec2_value/vec1_value)
                else:
                    vec1[i] = float(vec1_value/vec2_value)
                    vec2[i] = float(vec2_value/vec2_value)
            else:
                vec1[i] = float(0.0)
                vec2[i] = float(0.0)
            i += 1
        similarity = math.sqrt(math.pow(
            vec1[0]-vec2[0], 2)+math.pow(vec1[1]-vec2[1], 2)+math.pow(vec1[2]-vec2[2], 2))
    return round(similarity, 2)


def calc_vaf(input_bam, chr, pos, alt_allele):
    samfile = pysam.AlignmentFile(input_bam, "rU")
    sam_pileup = samfile.pileup(chr, pos-1, pos+1, truncate=True)
    alt_coverage = int(0)
    all_coverage = int(0)

    for pileupcolumn in sam_pileup:
        for pileupread in pileupcolumn.pileups:
            if pileupcolumn.pos == pos - 1:
                try:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                except:
                    base = ''
                all_coverage += 1
                if base == alt_allele:
                    alt_coverage += 1
        vaf = vaf_value(all_coverage, alt_coverage)
        samfile.close()
        return vaf


def calc_coverage(input_bam, chr, pos, alt_allele):
    samfile = pysam.AlignmentFile(input_bam, "rU")
    sam_pileup = samfile.pileup(chr, pos-1, pos+1, truncate=True)
    alt_coverage = int(0)
    for pileupcolumn in sam_pileup:
        for pileupread in pileupcolumn.pileups:
            if pileupcolumn.pos == pos - 1:
                try:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                except:
                    base = ''
                if base == alt_allele:
                    alt_coverage += 1
        samfile.close()
        return alt_coverage


def calc_strandbias(input_bam, chr, pos, alt_allele):
    samfile = pysam.AlignmentFile(input_bam, "rU")
    sam_pileup = samfile.pileup(chr, pos-1, pos+1, truncate=True)
    forward_strand = float(0)
    reverse_strand = float(0)
    for pileupcolumn in sam_pileup:
        for pileupread in pileupcolumn.pileups:
            if pileupcolumn.pos == pos - 1:
                try:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                except:
                    base = ''
                if base == alt_allele:
                    reverse = pileupread.alignment.is_reverse
                    if reverse:
                        reverse_strand += 1
                    else:
                        forward_strand += 1
        bias = calc_bias(forward_strand, reverse_strand)
        samfile.close()
        return bias


def calc_sim_feature(sample, chrm, start, alt, vaf, sim_features):
    cov = calc_coverage(sample, chrm, start, alt)
    bias = calc_strandbias(sample, chrm, start, alt)
    features = [cov, vaf, bias]
    sim = calc_sim(sim_features, features)
    return sim


def fetch_mq(input_bam, chr, pos, alt_allele):
    samfile = pysam.AlignmentFile(input_bam, "rU")
    sam_pileup = samfile.pileup(chr, pos-1, pos+1, truncate=True)
    mapping_qual = []
    for pileupcolumn in sam_pileup:
        for pileupread in pileupcolumn.pileups:
            if pileupcolumn.pos == pos - 1:
                try:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                except:
                    base = ''
                if base == alt_allele:
                    mapping_qual.append(
                        float(pileupread.alignment.mapping_quality))
        samfile.close()
        return mapping_qual


def fetch_read(input_bam, chr, pos, alt_allele):
    samfile = pysam.AlignmentFile(input_bam, "rU")
    sam_pileup = samfile.pileup(chr, pos-1, pos+1, truncate=True)
    for pileupcolumn in sam_pileup:
        for pileupread in pileupcolumn.pileups:
            if pileupcolumn.pos == pos - 1:
                try:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                except:
                    base = ''
                if base == alt_allele:
                    read = pileupread.alignment.query_sequence
                    samfile.close()
                    return read


def is_psuedoregion(sample_id, chr, pos, pseudoregion_file, pseudo_log=None):
    pseudo_region_file = csv.reader(
        open(pseudoregion_file, "rU"), delimiter="\t")
    is_pseudo = False

    for pseudo_region in pseudo_region_file:
        pseudo_chr = pseudo_region[0]
        pseudo_start = int(pseudo_region[1])
        pseudo_end = int(pseudo_region[2])
        if chr == pseudo_chr and pos >= pseudo_start and pos <= pseudo_end:
            region = str(pseudo_chr) + "\t" + \
                str(pseudo_start) + "\t" + str(pseudo_end)
            info = "Pseudoregion found for " + \
                str(sample_id) + ':' + str(chr) + ":" + \
                str(pos) + " - (" + region + ")"
            print(info)
            is_pseudo = True
    return is_pseudo


def split_list(parser, string):
    split_list = [testcode.strip() for testcode in string.split(',')]
    return split_list


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bam", help="input .bam file",
                        dest="input_bam", required=True, type=str)
    parser.add_argument("--input_variants", help="input variants, tab separated",
                        dest="input_variants", required=True, type=str)
    parser.add_argument("--sample_id", help="sample ID",
                        dest="sample_id", required=True, type=str)
    parser.add_argument("--class_type", help='either "POS" or "NEG"',
                        dest="class_type", required=True, type=str)
    parser.add_argument("--pseudoregions_file", help="pseudoregions file",
                        dest="pseudoregions", required=True, type=str)
    parser.add_argument("--normal_control_bams",
                        help=".bam comma-separated list to use as normal controls",
                        dest="normal_controls",
                        required=True,
                        type=lambda x: split_list(parser, x))
    parser.add_argument("--batch_control_bams",
                        help=".bam comma-separated list to use as batch controls",
                        dest="batch_controls",
                        required=True,
                        type=lambda x: split_list(parser, x))

    args = parser.parse_args()

    input_bam = args.input_bam
    input_variants = args.input_variants
    sample_id = args.sample_id
    class_type = args.class_type
    pseudoregions = args.pseudoregions
    normal_controls = args.normal_controls
    batch_controls = args.batch_controls

    if len(normal_controls) != 2:
        print("Two normal controls are required")
        exit(1)

    if len(batch_controls) < 1:
        print("At least one batch control is required")
        exit(1)

    if class_type not in ["POS", "NEG"]:
        print('"POS" or "NEG" are the only two valid class types')
        exit(1)

    samfile = pysam.AlignmentFile(input_bam, "rU")
    roi = csv.reader(open(input_variants, "rU"), delimiter="\t")

    output_file = sample_id + '_' + class_type + ".features.txt"
    output_file_writer = open(output_file, "w")
    output_file_writer.write("Sample" + "\t" + "Chr" + "\t" + "Pos" + "\t" + "Alt" + "\t" +
                             "Coverage" + "\t" + "Bias" + "\t" + "VAF" + "\t" +
                             "Control Sim1" + "\t" + "Control_Sim2" + "\t" + "Batch Sim" + "\n")

    for variant in roi:
        try:
            features = []

            chrm = variant[0]
            start = int(variant[1])
            alt = variant[2]
            features.append(chrm)
            features.append(start)

            # set default values for each feature
            alt_coverage = int(0)
            forward_strand = float(0)
            reverse_strand = float(0)
            strand_bias = float(0.0)
            all_coverage = int(0)
            read = ""

            is_pseudo = is_psuedoregion(sample_id, chrm, start, pseudoregions)
            if not is_pseudo:
                alt_coverage = calc_coverage(input_bam, chrm, start, alt)
                vaf = calc_vaf(input_bam, chrm, start, alt)
                strand_bias = calc_strandbias(input_bam, chrm, start, alt)
                sim_features = [alt_coverage, vaf, strand_bias]

                # calculate control_sim1 feature for control with highest VAF
                normal_dict = {}
                for normal_control in normal_controls:
                    vaf = calc_vaf(normal_control, chrm, start, alt)
                    normal_dict[normal_control] = vaf

                norm_sample = max(normal_dict, key=normal_dict.get)
                norm_max_vaf = normal_dict[norm_sample]
                control_sim1 = calc_sim_feature(
                    norm_sample, chrm, start, alt, norm_max_vaf, sim_features)
                features.append(control_sim1)

                # calculate control_sim2 feature for remaining control
                del normal_dict[norm_sample]
                norm_sample = max(normal_dict, key=normal_dict.get)
                norm_max_vaf = normal_dict[norm_sample]
                control_sim2 = calc_sim_feature(
                    norm_sample, chrm, start, alt, norm_max_vaf, sim_features)
                features.append(control_sim2)

                # calculate batch_sim feature for batch sample with highest VAF
                batch_dict = {}
                for batch_control in batch_controls:
                    vaf = calc_vaf(batch_control, chrm, start, alt)
                    batch_dict[batch_control] = vaf

                batch_sample = max(batch_dict, key=batch_dict.get)
                batch_max_vaf = batch_dict[batch_sample]
                batch_sim = calc_sim_feature(
                    batch_sample, chrm, start, alt, batch_max_vaf, sim_features)

                # write features to output file
                output_file_writer.write(sample_id + "\t" + chrm + "\t" + str(start) + "\t" + alt +
                                         "\t" + str(alt_coverage) + "\t" + str(strand_bias) +
                                         "\t" + str(vaf) + "\t" + str(control_sim1) + "\t" +
                                         str(control_sim2) + "\t" + str(batch_sim) + "\n")
        except IndexError as e:
            print "Variant is not in proper format: ", variant, e
            pass

        samfile.close()


if __name__ == '__main__':
    main()
