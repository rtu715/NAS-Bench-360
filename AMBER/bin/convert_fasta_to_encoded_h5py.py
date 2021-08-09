"""
Author: Evan M. Cofer
Created on: June 29, 2020
"""
import argparse
import os

import h5py
import numpy
import pyfaidx

from amber.utils.sequences import EncodedGenome


def convert_file(input_file, output_file):
    g = EncodedGenome(input_file, in_memory=False)
    h5 = h5py.File(output_file, "w")

    # Convert.
    for chrom, chrom_len in g.chrom_len_dict.items():
        ds = h5.create_dataset(chrom, (chrom_len, 4), dtype=g.ALPHABET_TO_ARRAY['N'].dtype)
        s = str(g.data[chrom]).upper()
        for i in range(chrom_len):
            if s[i] == 'A':
                ds[i, 0] = 1
            elif s[i] == 'C':
                ds[i, 1] = 1
            elif s[i] == 'T':
                ds[i, 2] = 1
            elif s[i] == 'G':
                ds[i, 3] = 1
            elif s[i] == 'N':
                ds[i, :] = 0.25
            else:
                s = "Failed on chrom {} in position {} with character \"{}\"".format(chrom, i, s[i])
                raise ValueError(s)

    # Close.
    h5.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fasta file to H5PY file with array for genome.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to FASTA input file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to store H5PY file at")
    args = parser.parse_args()


    # Validate arguments.
    for x in [args.input_file]:
        if not os.path.exists(x):
            s = 'Found no input file at {}'.format(x)
            raise ValueError(s)
    for x in [args.output_file]:
        if os.path.exists(x):
            s = 'Found output file already exists at {}'.format(x)
            raise ValueError(s)
    # Run command.
    convert_file(args.input_file, args.output_file)
