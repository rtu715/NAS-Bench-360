"""
Evan Cofer, 2020
"""
import argparse
import collections
import os
import re
import sys

import bx.intervals.intersection
import gzip
import numpy
import pyfaidx

def draw_samples(genome_file, bed_file, output_file, feature_name_file,
                 bin_size, cvg_frac, n_examples,
                 chrom_pad, chrom_pattern, max_unk, interval_file):
    # Read feature names from file.
    feature_name_set = set()
    i_to_feature_name = list()
    feature_name_to_i = dict()
    i = 0
    with open(feature_name_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                feature_name_set.add(line)
                i_to_feature_name.append(line)
                feature_name_to_i[line] = i
                i += 1
    n_feats = len(i_to_feature_name)
    print("Loaded features", file=sys.stderr)

    # Load genome and get estimate of chromosome weights.
    genome = pyfaidx.Fasta(genome_file)
    chroms = list()
    chrom_lens = list()
    max_examples = 0
    for k in genome.keys():
        if chrom_pattern.match(k) is not None:
            l = len(genome[k])
            if l > chrom_pad * 3:
                for s in ["+", "-"]: # Different mass on strands.
                    chroms.append((s, k))
                    chrom_lens.append(l - 2 * chrom_pad - bin_size)
    n_chrom = len(chroms)
    chrom_to_i = {k: i for (i, k) in enumerate(chroms)}
    print("Loaded chroms", file=sys.stderr)

    # Get intervals.
    chrom_bound_ivals = {k: list() for k in chrom_to_i.values()}
    chrom_bound_ival_weights = {k: list() for k in chrom_to_i.values()}
    chrom_lens = numpy.array(chrom_lens)
    if interval_file is None:
        chrom_weighting_lens = chrom_lens.copy()
        for x, chrom_len in zip(chroms, chrom_lens.tolist()):
            c_i = chrom_to_i[x]
            chrom_bound_ivals[c_i].append((chrom_pad, chrom_len + chrom_pad))
            chrom_bound_ival_weights[c_i].append(chrom_len)
    else:
        chrom_weighting_lens = numpy.zeros_like(chrom_lens)
        with open(interval_file, "r") as read_file:
            for line_i, line in enumerate(read_file):
                line = line.strip()
                if line:
                    if not line.startswith("#"):
                        line = line.split("\t")
                        if len(line) != 6:
                            s = "Found that line #{} has {} elements and not 6".format(line_i, len(line))
                            raise ValueError(s)
                        chrom, start, end, _, _, strand = line
                        start = int(start)
                        end = int(end)
                        start, end = min(start, end - 1), max(start + 1, end)
                        if (strand, chrom) in chrom_to_i:
                            i = chrom_to_i[(strand, chrom)]
                            start = max(start, chrom_pad)
                            end = min(end, chrom_lens[i] - chrom_pad - bin_size)
                            if end - start <= 0:
                                continue
                            chrom_bound_ivals[i].append((start, end))
                            dist = abs(end - start)
                            chrom_weighting_lens[i] += dist
                            chrom_bound_ival_weights[i].append(dist)
    print("Loaded chrom ivals", file=sys.stderr)

    # Calculate weighting and number of examples.
    chrom_weights = chrom_weighting_lens / chrom_weighting_lens.sum()
    max_examples = numpy.sum(chrom_weighting_lens)
    if max_examples < n_examples:
        msg = "Got {} max examples possible, but need {} examples".format(
            max_examples, n_examples)
        raise ValueError(msg)

    # Create interval tree for fast label query.
    ivt = {k : {kk: bx.intervals.intersection.IntervalTree() for kk in feature_name_to_i.values()} for k in chrom_to_i.values()}
    with gzip.open(bed_file, "rt") as read_file:
        for line in read_file:
            line = line.strip()
            if not line.startswith("#"):
                line = line.split("\t")
                chrom, start, end, name = line[:4]
                if len(line) >= 6:
                    if line[5] == ".":
                        strand = ["+", "-"]
                    else:
                        strand = [line[5]]
                else:
                    strand = ["+", "-"]
                start = int(start)
                end = int(end)
                if name in feature_name_set:
                    for x in strand:
                        i = chrom_to_i[(x, chrom)]
                        ivt[i][feature_name_to_i[name]].insert_interval(
                            bx.intervals.intersection.Interval(start, end))
    print("Loaded labels", file=sys.stderr)

    # Create outputs.
    outputs = list()
    i = 0
    while i < n_examples:
        c_i = numpy.random.choice(n_chrom, p=chrom_weights)
        strand, chrom = chroms[c_i]
        try:
            ival_bin_i = numpy.random.choice(len(chrom_bound_ivals[c_i]),
                                         p=numpy.array(chrom_bound_ival_weights[c_i]).flatten() / chrom_weighting_lens[c_i])
        except:
            print("***FAILED ON RANDOM CHOICE",
                  c_i,
                  chrom_bound_ivals[c_i],
                  chrom_weights,
                  chrom_weighting_lens, sep="\n", flush=True, file=sys.stderr)
            raise RuntimeError()
        start, end = chrom_bound_ivals[c_i].pop(ival_bin_i)
        cur_weight = chrom_bound_ival_weights[c_i].pop(ival_bin_i) # Remove weight.
        try:
            pos = numpy.random.choice(end - start) + start
        except:
            print("***FAILED ON RANDOM CHOICE",
                  c_i,
                  chrom_bound_ivals[c_i],
                  chrom_weights,
                  chrom_weighting_lens, sep="\n", flush=True, file=sys.stderr)
            raise RuntimeError()
        if end - start > 1:
            if pos == start:
                chrom_bound_ival_weights[c_i].append(cur_weight - 1)
                chrom_bound_ivals[c_i].append((start + 1, end))
            elif pos == end - 1:
                chrom_bound_ival_weights[c_i].append(cur_weight - 1)
                chrom_bound_ivals[c_i].append((start, end - 1))
            else:
                chrom_bound_ivals[c_i].append((start, pos))
                chrom_bound_ival_weights[c_i].append(pos - start)
                chrom_bound_ivals[c_i].append((pos + 1, end))
                chrom_bound_ival_weights[c_i].append(end - (pos + 1))
        start = pos
        end = pos + bin_size

        # Adjust chrom weights.
        chrom_weighting_lens[c_i] -= 1
        chrom_weights = chrom_weighting_lens / numpy.sum(chrom_weighting_lens)

        # Determine label etc w/ ivt.
        cvg = numpy.zeros(n_feats)
        for feat_i in feature_name_to_i.values():
            for x in ivt[c_i][feat_i].find(start, end):
                cvg[feat_i] += min(x.end, end) - max(x.start, start)
        cvg /= bin_size
        cvg = (cvg > cvg_frac).astype(int).tolist()
        outputs.append((chrom, start, end, strand, *cvg))
        i += 1
        if i % 10000 == 0:
            print(i)
    print("Loaded all examples.", file=sys.stderr)

    # write outputs to file.
    with open(output_file, "w") as write_file:
        for x in sorted(outputs):
            x = [str(y) for y in list(x)]
            write_file.write("\t".join(x) + "\n")
    print("Wrote examples.", file=sys.stderr)


if __name__ == "__main__":
    # Get command line arguments.
    parser = argparse.ArgumentParser( description="sampling data for tf genomics models")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--genome", type=str, required=True, help="Path to the indexed fasta file.")
    parser.add_argument("--bed", type=str, required=True, help="Path to the input bed file.")
    parser.add_argument("--feature-name-file", type=str, required=True, help="Name of feature in bed file to look for.")
    parser.add_argument("--bin-size", type=int, required=True, help="Size of the bin composing minimal examples.")
    parser.add_argument("--cvg-frac", type=float, required=True, help="Fraction of bin that must be covered to be positive example.")
    parser.add_argument("--n-examples", type=int, required=True, help="Number of examples to include.")
    parser.add_argument("--chrom-pad", type=int, required=True, help="Length of region to ignore at the start and end of chromosomes.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for RNG.")
    parser.add_argument("--include-chroms", type=str, required=True, help="Regex for chromosomes to include.")
    parser.add_argument("--max-n", type=int, required=True, help="Maximum N chars in sequences.")
    parser.add_argument("--interval-file", type=str, required=False, default=None, help="Path to file with intervals to draw positives from")
    args = parser.parse_args()

    # Validate arguments.
    for x in [args.genome, args.bed, args.feature_name_file] + ([] if args.interval_file is None else [args.interval_file]):
        if not os.path.exists(x):
            raise ValueError(x + " does not exist")

    if args.n_examples <= 0:
        raise ValueError("--n-examples must be > 0")

    if args.chrom_pad < 0:
        raise ValueError("--chrom-pad must be >= 0")

    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    if args.cvg_frac < 0:
        raise ValueError("--cvg-frac must be >= 0")
    elif args.cvg_frac > 1:
        raise ValueError("--cvg-frac must be < 1")

    if args.max_n < 0:
        raise ValueError("--max-n must be >= 0")


    # Prepare for function.
    numpy.random.seed(args.seed)
    pattern = re.compile("^" + args.include_chroms + "$")

    # Run function.
    draw_samples(args.genome, args.bed, args.output, args.feature_name_file,
                 args.bin_size, args.cvg_frac, args.n_examples,
                 args.chrom_pad, pattern, args.max_n, args.interval_file)
