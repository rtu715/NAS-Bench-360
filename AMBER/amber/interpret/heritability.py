"""utils for interpreting variant effect prediction for Heritability
"""

import gzip
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd


def read_vep(vep_dir, check_sanity=False):
    _label_fn = [x for x in os.listdir(vep_dir) if x.endswith("_row_labels.txt")]
    _data_fn = [x for x in os.listdir(vep_dir) if x.endswith("_abs_diffs.h5")]
    assert len(_label_fn) == len(
        _data_fn) == 1, "Each folder must have exact one row_labels and one abs_diffs file; found %i row_labels and " \
                        "%i abs_diffs" % (len(_label_fn), len(_data_fn))
    label_fn = os.path.join(vep_dir, _label_fn[0])
    data_fn = os.path.join(vep_dir, _data_fn[0])
    vep_df = pd.read_csv(label_fn, sep='\t')
    data_fh = h5py.File(data_fn, 'r')
    try:
        vep_data = data_fh['data'].value
    except:
        print("read in h5 file failed")
        sys.exit(250)
    if check_sanity:
        assert vep_data.shape[0] == np.sum(vep_df['ref_match'])
    return vep_df, vep_data


def read_vep_logfc(vep_dir):
    _label_fn = [x for x in os.listdir(vep_dir) if x.endswith("_row_labels.txt")]
    _data_fn = [x for x in os.listdir(vep_dir) if x.endswith("_abs_logfc.npz")]
    _data_fn1 = [x for x in os.listdir(vep_dir) if x.endswith("ref_predictions.h5")]
    _data_fn2 = [x for x in os.listdir(vep_dir) if x.endswith("alt_predictions.h5")]

    label_fn = os.path.join(vep_dir, _label_fn[0])
    vep_df = pd.read_csv(label_fn, sep='\t')

    if len(_data_fn):
        assert len(_data_fn) == 1
        vep_data = np.load(os.path.join(vep_dir, _data_fn[0]))['arr_0']
    else:
        assert len(_label_fn) == len(_data_fn1) == len(
            _data_fn2) == 1, "Each folder must have exact one row_labels and one abs_diffs file; found %i row_labels " \
                             "and %i, %i abs_diffs" % ( len(_label_fn), len(_data_fn1), len(_data_fn2))
        data_fn1 = os.path.join(vep_dir, _data_fn1[0])
        data_fn2 = os.path.join(vep_dir, _data_fn2[0])
        data_fh1 = h5py.File(data_fn1, 'r')
        data_fh2 = h5py.File(data_fn2, 'r')
        try:
            vep_data1 = data_fh1['data'].value
            vep_data2 = data_fh2['data'].value
        except:
            print("read in h5 file failed")
            sys.exit(250)
        vep_data1 = np.clip(vep_data1, 0.0001, 0.9999)
        vep_data2 = np.clip(vep_data2, 0.0001, 0.9999)
        vep_data = np.abs(np.log(vep_data1 / (1 - vep_data1)) - np.log(vep_data2 / (1 - vep_data2)))
        colmax = np.apply_along_axis(np.max, 0, vep_data)  # vep_data is lower-bounded by 0
        vep_data /= colmax
        np.savez(os.path.join(vep_dir, "VEP_abs_logfc.npz"), vep_data)

    return vep_df, vep_data


def convert_to_ldsc_annot_by_label(vep_df, vep_data, label_fp, baselineLD_dir, output_dir, resume_prev_run=False):
    """read in the h5 vep data snp annot and numerical values, convert to
    the existing baselineLD annotations for next steps
    """
    baselineLDs = [x for x in os.listdir(baselineLD_dir) if x.endswith("annot.gz")]
    # label_df is annotation for output chromatin features
    label_df = pd.read_table(label_fp)
    # vep_dict is a mapping from chrom,bp to vep_data row index
    vep_dict = defaultdict(list)
    print('making vep mapper..')
    for i in range(vep_df.shape[0]):
        vep_dict[(vep_df.chrom[i], str(vep_df.pos[i]))].append(i)

    # iterate through each labels in label_df, make an independent ldsc-annot
    for k in range(label_df.shape[0]):
        label_idx = label_df['label_idx'][k]
        label_name = label_df['label_name'][k]
        # normalize label names
        label_name = label_name.replace('|', '--')
        label_name = label_name.replace('(', '_').replace(')', '_')
        label_output_dir = os.path.join(output_dir, label_name)
        os.makedirs(label_output_dir, exist_ok=True)
        print("%i/%i %s" % (k, label_df.shape[0], label_name))
        for chrom_fn in baselineLDs:
            chrom = chrom_fn.split(".")[-3]
            print(chrom)
            if resume_prev_run and os.path.isfile(
                    os.path.join(label_output_dir, "%s.%s.annot.gz" % (label_name, chrom))):
                print("found %s, skip" % chrom)
                continue
            with gzip.GzipFile(os.path.join(baselineLD_dir, chrom_fn), 'rb') as fi, gzip.GzipFile(
                    os.path.join(label_output_dir, "%s.%s.annot.gz" % (label_name, chrom)), 'wb') as fo:
                fi.readline()  # pop first line
                fo.write(("\t".join(['CHR', 'BP', 'SNP', 'CM', label_name]) + '\n').encode('utf-8'))
                # for line in tqdm(fi):
                for line in fi:
                    line = line.decode('utf-8')
                    ele = line.strip().split()
                    _chr, _bp, _snp, _cm = ele[0:4]
                    # _bp = str(int(_bp) - 1)
                    # _annot_idx = np.where(label_df.eval("pos==%s & chrom=='chr%s'"%(_bp, _chr)))[0]
                    _annot_idx = vep_dict[("chr%s" % _chr, _bp)]
                    if len(_annot_idx) == 0:
                        # this is less than 0.5% - ignored
                        # warnings.warn("baselineLD variant not found in vep: %s,%s"%(_chr, _bp))
                        # continue
                        _annot = "0"
                    else:
                        _annot = "%.5f" % np.max(vep_data[_annot_idx, label_idx])
                    fo.write(("\t".join([
                        _chr,
                        _bp,
                        _snp,
                        _cm,
                        _annot]) + '\n').encode('utf-8')
                             )


def make_vep_mapper(vep_df):
    # vep_dict is a mapping from chrom,bp to vep_data row index
    vep_dict = defaultdict(list)
    print('making vep mapper..')
    for i in range(vep_df.shape[0]):
        vep_dict[(vep_df.chrom[i], str(vep_df.pos[i]))].append(i)
    return vep_dict


def convert_to_ldsc_annot(vep_dir, label_fp, baselineLD_dir, output_dir, chroms_part=None, use_temp=None,
                          use_logfc=False):
    """read in the h5 vep data snp annot and numerical values, convert to
    the existing baselineLD annotations based on a set of baselineLD SNPs
    """
    if use_logfc:
        vep_df, vep_data = read_vep_logfc(vep_dir)
    else:
        vep_df, vep_data = read_vep(vep_dir, check_sanity=False)
    chroms_target = chroms_part.split(',')
    baselineLDs = [x for x in os.listdir(baselineLD_dir) if x.endswith("annot.gz")]
    # label_df is annotation for output chromatin features
    label_df = pd.read_table(label_fp)
    # vep_dict is a mapping from chrom,bp to vep_data row index
    vep_dict = defaultdict(list)
    print('making vep mapper..')
    for i in range(vep_df.shape[0]):
        if vep_df.chrom[i].strip('chr') not in chroms_target:
            continue
        vep_dict[(vep_df.chrom[i], str(vep_df.pos[i]))].append(i)
    print("done")

    # iterate through each labels in label_df, make a joint ldsc-annot
    label_names = []
    for k in range(label_df.shape[0]):
        label_idx = label_df['label_idx'][k]
        label_name = label_df['label_name'][k]
        # DO NOT RUN: label names should already 
        # be normalized
        # normalize label names
        # label_name = label_name.replace('|', '--')
        # label_name = label_name.replace('(', '_').replace(')', '_')
        # label_name = label_name.replace('+', '_').replace(' ','')
        label_names.append(label_name)

    num_labels = len(label_names)
    assert num_labels == vep_data.shape[1]

    for chrom_fn in baselineLDs:
        chrom = chrom_fn.split(".")[-3]
        if not chrom in chroms_target:
            continue
        print(chrom)
        if use_temp:
            fo = open(os.path.join(use_temp, "joint.%s.annot" % (chrom)), 'w')
        else:
            fo = open(os.path.join(output_dir, "joint.%s.annot" % (chrom)), 'w')
        with gzip.GzipFile(os.path.join(baselineLD_dir, chrom_fn), 'rb') as fi:
            fi.readline()  # pop first line
            fo.write(("\t".join(['CHR', 'BP', 'SNP', 'CM'] + label_names) + '\n'))
            # for line in tqdm(fi):
            counter = 0
            for line in fi:
                if counter % 100000 == 0:
                    print("processed %i" % counter)
                counter += 1
                line = line.decode('utf-8')
                ele = line.strip().split()
                _chr, _bp, _snp, _cm = ele[0:4]
                _annot_idx = vep_dict[("chr%s" % _chr, _bp)]
                if len(_annot_idx) == 0:
                    # this is less than 0.5% - ignored
                    # warnings.warn("baselineLD variant not found in vep: %s,%s"%(_chr, _bp))
                    # continue
                    _annot = ["0"] * num_labels
                else:
                    # _annot = ["%.5f"%np.mean(vep_data[_annot_idx, label_idx]) for label_idx in range(num_labels)]
                    if len(_annot_idx) > 1:
                        _annot = np.apply_along_axis(np.mean, 0, vep_data[_annot_idx, :]).flatten()
                    else:
                        _annot = vep_data[_annot_idx, :].flatten()
                    _annot = ["%.5f" % x for x in _annot]
                fo.write(("\t".join([
                                        _chr,
                                        _bp,
                                        _snp,
                                        _cm,
                                    ] + _annot) + '\n')
                         )
        fo.close()


def split_labels_to_folders(l2_prefix, output_dir):
    l2_df = pd.read_csv(l2_prefix + '.l2.ldscore.gz', sep="\t")
    M_df = pd.read_csv(l2_prefix + '.l2.M', sep="\t", header=None)
    M_5_50_df = pd.read_csv(l2_prefix + ".l2.M_5_50", header=None, sep="\t")
    annot_df = pd.read_csv(l2_prefix + ".annot", sep="\t")
    chrom = l2_prefix.split('.')[-1]
    for i in range(3, l2_df.shape[1]):
        label_name = l2_df.columns.values[i]
        label_folder = os.path.join(output_dir, label_name)
        os.makedirs(label_folder, exist_ok=True)
        l2_df.iloc[:, [0, 1, 2, i]].to_csv(
            os.path.join(label_folder, "%s.%s.l2.ldscore.gz" % (label_name, chrom)),
            index=False,
            sep="\t")
        M_df.iloc[0, [i - 3]].to_csv(os.path.join(label_folder, "%s.%s.l2.M" % (label_name, chrom)), index=False,
                                     header=False)
        M_5_50_df.iloc[0, [i - 3]].to_csv(os.path.join(label_folder, "%s.%s.l2.M_5_50" % (label_name, chrom)),
                                          index=False, header=False)
        annot_df.iloc[:, [0, 1, 2, 3, i + 1]].to_csv(os.path.join(label_folder, "%s.%s.annot.gz" % (label_name, chrom)),
                                                     sep="\t", index=False, header=True)
