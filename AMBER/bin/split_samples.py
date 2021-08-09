"""
Evan Cofer June 29, 2020
"""
import argparse
import os


def split_samples(input_file, feature_name_file, output_dir):
    # Read in feature names.
    i_to_feat = list()
    with open(feature_name_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                i_to_feat.append(line)
    feat_to_i = dict()
    for i, feat in enumerate(i_to_feat):
        feat_to_i[feat] = i

    # Create file handles for features.
    i_to_fp = list()
    for feat in i_to_feat:
        f = "{}.txt".format(feat)
        f = os.path.join(output_dir, f)
        i_to_fp.append(open(f, "w"))

    # Read in data and write out as we go.
    with open(input_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                line = line.split("\t")
                stub = "\t".join(line[:4]) + "\t"
                for i, fp in enumerate(i_to_fp):
                    fp.write(stub + line[4 + i] + "\n")

    # Close files.
    for x in i_to_fp:
        x.close()

if __name__ == "__main__":
    # Get command line arguments.
    parser = argparse.ArgumentParser( description="split input training examples for several models")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input samples to split.")
    parser.add_argument("--feature-name-file", type=str, required=True, help="Path to file with feature names listed in order.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to directory to write outputs to.")
    args = parser.parse_args()

    # Validate arguments.
    for x in [args.input_file, args.feature_name_file]:
        if not os.path.exists(x):
            s = "Found no file at {}".format(x)
            raise ValueError(s)

    if not os.path.isdir(args.output_dir):
        s = "Found no directory at {}".format(args.output_dir)
        raise ValueError(s)

    # Run main function.
    split_samples(args.input_file, args.feature_name_file, args.output_dir)
