"""
Author: Evan M. Cofer
Created on: June 29, 2020
"""
import argparse
import os

import h5py
import numpy
import pyfaidx


def convert_file(input_file, output_file):
    fasta = pyfaidx.Fasta(input_file)
    h5 = h5py.File(output_file, "w")

    # Convert.
    for k in fasta.keys():
        s = str(fasta[k][:].seq).upper()
        ds = h5.create_dataset(k, (len(s),), dtype='S1')
        for i in range(len(s)):
            ds[i] = numpy.string_(s[i])

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
