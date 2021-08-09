# -*- coding: UTF-8 -*-

"""Functions for interpret and explain the variant effects of the genomic sequence-based models

Currently this module heavily relies on Selene.

"""

# selene-related imports of V0.4.4; 2020.1.17 ZZ

import os
import sys
import time
import warnings
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
try:
    import pyfaidx
except ImportError:
    print("amber.interpret.sequence_model: cannot load pyfaidx")
    pyfaidx = None

from keras.models import load_model
try:
    from selene_sdk.predict._in_silico_mutagenesis import in_silico_mutagenesis_sequences, mutate_sequence, _ism_sample_id
    from selene_sdk.predict._variant_effect_prediction import read_vcf_file, _process_alt, _handle_standard_ref, \
        _handle_long_ref
    from selene_sdk.predict.model_predict import AnalyzeSequences, ISM_COLS, VARIANTEFFECT_COLS
    from selene_sdk.sequences import Genome as Genome
except ImportError:
    print("amber.interpret.sequence_model: cannot load selene")
    AnalyzeSequences = object
    Genome = None

from sklearn import metrics

from ..utils import motif as motif_fn


# Module-level over-write for compatibility with Selene v0.4.4
def predict(model, data, **kwargs):
    return model.predict(data)


def _handle_ref_alt_predictions(model,
                                batch_ref_seqs,
                                batch_alt_seqs,
                                batch_ids,
                                reporters,
                                use_cuda=False):
    """
    Helper method for variant effect prediction. Gets the model
    predictions and updates the reporters.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model, on mode `eval`.
    batch_ref_seqs : list(np.ndarray)
        One-hot encoded sequences with the ref base(s).
    batch_alt_seqs : list(np.ndarray)
        One-hot encoded sequences with the alt base(s).
    reporters : list(PredictionsHandler)
        List of prediction handlers.
    use_cuda : bool, optional
        Default is `False`. Specifies whether CUDA-enabled GPUs are available
        for torch to use.


    Returns
    -------
    None

    """
    batch_ref_seqs = np.array(batch_ref_seqs)
    batch_alt_seqs = np.array(batch_alt_seqs)
    ref_outputs = predict(model, batch_ref_seqs, use_cuda=use_cuda)
    alt_outputs = predict(model, batch_alt_seqs, use_cuda=use_cuda)
    for r in reporters:
        if r.needs_base_pred:
            r.handle_batch_predictions(alt_outputs, batch_ids, ref_outputs)
        else:
            r.handle_batch_predictions(alt_outputs, batch_ids)


# main analyzer class
class AnalyzeSequencesNAS(AnalyzeSequences):
    """AnalyzeSequence for NAS/keras model inherited from Selene

    Score sequences and their variants using the predictions made
    by a trained model.

    Parameters
    ----------
    model : keras.models.Model
        A sequence-based model architecture.
    trained_model_path : str or list(str)
        The path(s) to the weights file for a trained sequence-based model.
        For a single path, the model architecture must match `model`. For
        a list of paths, assumes that the `model` passed in is of type
        `selene_sdk.utils.MultiModelWrapper`, which takes in a list of
        models. The paths must be ordered the same way the models
        are ordered in that list. `list(str)` input is an API-only function--
        Selene's config file CLI does not support the `MultiModelWrapper`
        functionality at this time.
    sequence_length : int
        The length of sequences that the model is expecting.
    features : list(str)
        The names of the features that the model is predicting.
    batch_size : int, optional
        Default is 64. The size of the mini-batches to use.
    reference_sequence : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence on
        which this analysis will be performed. Please note that if you need
        to use variant effect prediction, you cannot only pass in the
        class--you must pass in the constructed `selene_sdk.sequences.Sequence`
        object with a particular sequence version (e.g. `Genome("hg19.fa")`).
        This version does NOT have to be the same sequence version that the
        model was trained on. That is, if the sequences in your variants file
        are hg19 but your model was trained on hg38 sequences, you should pass
        in hg19.
    write_mem_limit : int, optional
        Default is 5000. Specify, in MB, the amount of memory you want to
        allocate to storing model predictions/scores. When running one of
        in silico mutagenesis, variant effect prediction, or prediction,
        prediction/score handlers will accumulate data in memory and only
        write this data to files periodically. By default, Selene will write
        to files when the total amount of data (across all handlers) takes up
        5000MB of space. Please keep in mind that Selene will not monitor the
        memory needed to actually carry out the operations (e.g. variant effect
        prediction) or load the model, so `write_mem_limit` should always be
        less than the total amount of CPU memory you have available on your
        machine. For example, for variant effect prediction, we load all
        the variants in 1 file into memory before getting the predictions, so
        your machine must have enough memory to accommodate that. Another
        possible consideration is your model size and whether you are
        using it on the CPU or a CUDA-enabled GPU (i.e. setting
        `use_cuda` to True).

    Examples
    --------
    Pending

    """

    def __init__(self,
                 trained_model_path=None,
                 sequence_length=1000,
                 features=None,
                 batch_size=64,
                 reference_sequence=Genome,
                 swapbase=None,
                 write_mem_limit=1500):
        """
        Constructs a new `AnalyzeSequences` object.
        """

        self.trained_model_path = trained_model_path
        if self.trained_model_path:
            self.model = load_model(trained_model_path)

        self.sequence_length = sequence_length
        self._start_radius = sequence_length // 2
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._start_radius += 1

        self.batch_size = batch_size
        self.features = features
        self.reference_sequence = reference_sequence
        self.swapbase = swapbase
        if self.swapbase:
            Genome.update_bases_order(self.swapbase)
        self._write_mem_limit = write_mem_limit

        # for Selene compatbility; not useful
        self.use_cuda = None

    def get_predictions_for_bed_file(self,
                                     input_path,
                                     output_dir,
                                     output_format="tsv",
                                     strand_index=None):
        """
        Get model predictions for sequences specified as genome coordinates
        in a BED file. Coordinates do not need to be the same length as the
        model expected sequence input--predictions will be centered at the
        midpoint of the specified start and end coordinates.

        Parameters
        ----------
        input_path : str
            Input path to the BED file.
        output_dir : str
            Output directory to write the model predictions.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strand_index : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath.

        """
        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        seq_coords, labels = self._get_sequences_from_bed_file(
            input_path,
            strand_index=strand_index,
            output_NAs_to_file="{0}.NA".format(os.path.join(output_dir, output_prefix)),
            reference_sequence=self.reference_sequence)

        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            output_format,
            ["index", "chrom", "start", "end", "strand", "contains_unk"],
            output_size=len(labels),
            mode="prediction")[0]
        sequences = None
        batch_ids = []
        for i, (label, coords) in enumerate(zip(labels, seq_coords)):
            encoding, contains_unk = self.reference_sequence.get_encoding_from_coords_check_unk(
                *coords,
                pad=True)
            if sequences is None:
                sequences = np.zeros((self.batch_size, *encoding.shape))
            if i and i % self.batch_size == 0:
                preds = predict(self.model, sequences, use_cuda=self.use_cuda)
                # assume all Model instances have a `predict` method built-in
                # - which is the case for keras.Models and BioNAS.Controller.child
                # preds = self.model.predict(sequences)
                sequences = np.zeros((self.batch_size, *encoding.shape))
                reporter.handle_batch_predictions(preds, batch_ids)
                batch_ids = []
            batch_ids.append(label + (contains_unk,))
            sequences[i % self.batch_size, :, :] = encoding
            if contains_unk:
                warnings.warn("For region {0}, "
                              "reference sequence contains unknown base(s). "
                              "--will be marked `True` in the `contains_unk` column "
                              "of the .tsv or the row_labels .txt file.".format(
                    label))

        if (batch_ids and i == 0) or i % self.batch_size != 0:
            sequences = sequences[:i % self.batch_size + 1, :, :]
            preds = predict(self.model, sequences, use_cuda=self.use_cuda)
            # preds = self.model.predict(sequences)
            reporter.handle_batch_predictions(preds, batch_ids)

        reporter.write_to_file()

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_dir,
                                        mutate_n_bases=1,
                                        use_sequence_name=True,
                                        output_format="tsv",
                                        start_position=0,
                                        end_position=None):
        """
        Apply in silico mutagenesis to all sequences in a FASTA file.

        Please note that we have not parallelized this function yet, so runtime
        increases exponentially when you increase mutate_n_bases.

        Parameters
        ----------
        input_path: str
            The path to the FASTA file of sequences.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: [abs_diffs, diffs, logits, predictions].
        output_dir : str
            The path to the output directory. Directories in the path will be
            created if they do not currently exist.
        mutate_n_bases : int, optional
            Default is 1. The number of bases to mutate at one time in
            in silico mutagenesis.
        use_sequence_name : bool, optional.
            Default is True. If use_sequence_name, output files are prefixed
            by the sequence name/description corresponding to each sequence
            in the FASTA file. Spaces in the sequence name are replaced with
            underscores '_'. If not use_sequence_name, output files are
            prefixed with an index i (starting with 0) corresponding
            to the i-th sequence in the FASTA file.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. The desired output format. Each sequence in
            the FASTA file will have its own set of output files, where
            the number of output files depends on the number of save_data
            predictions/scores specified.
        start_position : int, optional
            Default is 0. The starting position of the subsequence to be
            mutated.
        end_position : int or None, optional
            Default is None. The ending position of the subsequence to be
            mutated. If left as `None`, then `self.sequence_length` will be
            used.


        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to output_dir.
            For HDF5 output and 'predictions' in `save_data`, an additional
            file named `ref_predictions.h5` will be outputted with the
            model prediction for the original input sequence.

        Raises
        ------
        ValueError
            If the value of `start_position` or `end_position` is negative.
        ValueError
            If there are fewer than `mutate_n_bases` between `start_position`
            and `end_position`.
        ValueError
            If `start_position` is greater or equal to `end_position`.
        ValueError
            If `start_position` is not less than `self.sequence_length`.
        ValueError
            If `end_position` is greater than `self.sequence_length`.

        """
        if end_position is None:
            end_position = self.sequence_length
        if start_position >= end_position:
            raise ValueError(("Starting positions must be less than the ending "
                              "positions. Found a starting position of {0} with "
                              "an ending position of {1}.").format(start_position,
                                                                   end_position))
        if start_position < 0:
            raise ValueError("Negative starting positions are not supported.")
        if end_position < 0:
            raise ValueError("Negative ending positions are not supported.")
        if start_position >= self.sequence_length:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(start_position, self.sequence_length))
        if end_position > self.sequence_length:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(end_position, self.sequence_length))
        if (end_position - start_position) < mutate_n_bases:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(end_position - start_position, mutate_n_bases))

        os.makedirs(output_dir, exist_ok=True)

        fasta_file = pyfaidx.Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = str.upper(str(fasta_record))
            if len(cur_sequence) < self.sequence_length:
                cur_sequence = _pad_sequence(cur_sequence,
                                             self.sequence_length,
                                             self.reference_sequence.UNK_BASE)
            elif len(cur_sequence) > self.sequence_length:
                cur_sequence = _truncate_sequence(
                    cur_sequence, self.sequence_length)

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(
                cur_sequence,
                mutate_n_bases=mutate_n_bases,
                reference_sequence=self.reference_sequence,
                start_position=start_position,
                end_position=end_position)
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(
                1, *cur_sequence_encoding.shape)
            base_preds = predict(
                self.model, base_encoding, use_cuda=self.use_cuda)

            file_prefix = None
            if use_sequence_name:
                file_prefix = os.path.join(
                    output_dir, fasta_record.name.replace(' ', '_'))
            else:
                file_prefix = os.path.join(
                    output_dir, str(i))
            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(
                save_data,
                file_prefix,
                output_format,
                ISM_COLS,
                output_size=len(mutated_sequences))

            if "predictions" in save_data and output_format == 'hdf5':
                ref_reporter = self._initialize_reporters(
                    ["predictions"],
                    "{0}_ref".format(file_prefix),
                    output_format, ["name"], output_size=1)[0]
                ref_reporter.handle_batch_predictions(
                    base_preds, [["input_sequence"]])
                ref_reporter.write_to_file()
            elif "predictions" in save_data and output_format == 'tsv':
                reporters[-1].handle_batch_predictions(
                    base_preds, [["input_sequence", "NA", "NA"]])

            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences,
                reporters=reporters)
        fasta_file.close()

    def in_silico_mutagenesis_predict(self,
                                      sequence,
                                      base_preds,
                                      mutations_list,
                                      reporters=[]):
        """
        Get the predictions for all specified mutations applied
        to a given sequence and, if applicable, compute the scores
        ("abs_diffs", "diffs", "logits") for these mutations.

        Parameters
        ----------
        sequence : str
            The sequence to mutate.
        base_preds : numpy.ndarray
            The model's prediction for `sequence`.
        mutations_list : list(list(tuple))
            The mutations to apply to the sequence. Each element in
            `mutations_list` is a list of tuples, where each tuple
            specifies the `int` position in the sequence to mutate and what
            `str` base to which the position is mutated (e.g. (1, 'A')).
        reporters : list(PredictionsHandler)
            The list of reporters, where each reporter handles the predictions
            made for each mutated sequence. Will collect, compute scores
            (e.g. `AbsDiffScoreHandler` computes the absolute difference
            between `base_preds` and the predictions for the mutated
            sequence), and output these as a file at the end.

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.

        """
        current_sequence_encoding = self.reference_sequence.sequence_to_encoding(
            sequence)
        for i in range(0, len(mutations_list), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(mutations_list))

            mutated_sequences = np.zeros(
                (end - start, *current_sequence_encoding.shape))

            batch_ids = []
            for ix, mutation_info in enumerate(mutations_list[start:end]):
                mutated_seq = mutate_sequence(
                    current_sequence_encoding, mutation_info,
                    reference_sequence=self.reference_sequence)
                mutated_sequences[ix, :, :] = mutated_seq
                batch_ids.append(_ism_sample_id(sequence, mutation_info))
            outputs = predict(
                self.model, mutated_sequences, use_cuda=self.use_cuda)

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batch_ids, base_preds)
                else:
                    r.handle_batch_predictions(outputs, batch_ids)

        for r in reporters:
            r.write_to_file()

    def variant_effect_prediction(self,
                                  vcf_file,
                                  save_data,
                                  output_dir=None,
                                  output_format="tsv",
                                  strand_index=None,
                                  require_strand=False):
        """
        Get model predictions and scores for a list of variants.

        Parameters
        ----------
        vcf_file : str
            Path to a VCF file. Must contain the columns
            [#CHROM, POS, ID, REF, ALT], in order. Column header does not need
            to be present.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        output_dir : str or None, optional
            Default is None. Path to the output directory. If no path is
            specified, will save files corresponding to the options in
            `save_data` to the current working directory.
        output_format : {'tsv', 'hdf5'}, optional
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of variants is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (chrom, pos, id, ref, alt) will be
                  output as a separate .txt file.
        strand_index : int or None, optional.
            Default is None. If applicable, specify the column index (0-based)
            in the VCF file that contains strand information for each variant.
        require_strand : bool, optional.
            Default is False. Whether strand can be specified as '.'. If False,
            Selene accepts strand value to be '+', '-', or '.' and automatically
            treats '.' as '+'. If True, Selene skips any variant with strand '.'.
            This parameter assumes that `strand_index` has been set.

        Returns
        -------
        None
            Saves all files to `output_dir`. If any bases in the 'ref' column
            of the VCF do not match those at the specified position in the
            reference genome, the row labels .txt file will mark this variant
            as `ref_match = False`. If most of your variants do not match
            the reference genome, please check that the reference genome
            you specified matches the version with which the variants were
            called. The predictions can used directly if you have verified that
            the 'ref' bases specified for these variants are correct (Selene
            will have substituted these bases for those in the reference
            genome). In addition, if any base in the retrieved reference
            sequence is unknown, the row labels .txt file will mark this variant
            as `contains_unk = True`. Finally, some variants may show up in an
            'NA' file. This is because the surrounding sequence context ended up
            being out of bounds or overlapping with blacklist regions  or the
            chromosome containing the variant did not show up in the reference
            genome FASTA file.

        """
        # TODO: GIVE USER MORE CONTROL OVER PREFIX.
        path, filename = os.path.split(vcf_file)
        output_path_prefix = '.'.join(filename.split('.')[:-1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = path

        output_path_prefix = os.path.join(output_dir, output_path_prefix)
        variants = read_vcf_file(
            vcf_file,
            strand_index=strand_index,
            require_strand=require_strand,
            output_NAs_to_file="{0}.NA".format(output_path_prefix),
            seq_context=(self._start_radius, self._end_radius),
            reference_sequence=self.reference_sequence)
        reporters = self._initialize_reporters(
            save_data,
            output_path_prefix,
            output_format,
            VARIANTEFFECT_COLS,
            output_size=len(variants),
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        t_i = time()
        for ix, (chrom, pos, name, ref, alt, strand) in enumerate(variants):
            # centers the sequence containing the ref allele based on the size
            # of ref
            center = pos + len(ref) // 2
            start = center - self._start_radius
            end = center + self._end_radius
            ref_sequence_encoding, contains_unk = \
                self.reference_sequence.get_encoding_from_coords_check_unk(
                    chrom, start, end)

            ref_encoding = self.reference_sequence.sequence_to_encoding(ref)
            alt_sequence_encoding = _process_alt(
                chrom, pos, ref, alt, start, end,
                ref_sequence_encoding,
                self.reference_sequence)

            match = True
            seq_at_ref = None
            if len(ref) and len(ref) < self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_standard_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self.sequence_length,
                    self.reference_sequence)
            elif len(ref) >= self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_long_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self._start_radius,
                    self._end_radius,
                    self.reference_sequence)

            if contains_unk:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                              "reference sequence contains unknown base(s)"
                              "--will be marked `True` in the `contains_unk` column "
                              "of the .tsv or the row_labels .txt file.".format(
                    chrom, pos, name, ref, alt, strand))
            if not match:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                              "reference does not match the reference genome. "
                              "Reference genome contains {6} instead. "
                              "Predictions/scores associated with this "
                              "variant--where we use '{3}' in the input "
                              "sequence--will be marked `False` in the `ref_match` "
                              "column of the .tsv or the row_labels .txt file".format(
                    chrom, pos, name, ref, alt, strand, seq_at_ref))
            batch_ids.append((chrom, pos, name, ref, alt, strand, match, contains_unk))
            if strand == '-':
                ref_sequence_encoding = get_reverse_complement_encoding(
                    ref_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
                alt_sequence_encoding = get_reverse_complement_encoding(
                    alt_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
            batch_ref_seqs.append(ref_sequence_encoding)
            batch_alt_seqs.append(alt_sequence_encoding)

            if len(batch_ref_seqs) >= self.batch_size:
                _handle_ref_alt_predictions(
                    self.model,
                    batch_ref_seqs,
                    batch_alt_seqs,
                    batch_ids,
                    reporters,
                    use_cuda=self.use_cuda)
                batch_ref_seqs = []
                batch_alt_seqs = []
                batch_ids = []

            if ix and ix % 10000 == 0:
                print("[STEP {0}]: {1} s to process 10000 variants.".format(
                    ix, time() - t_i))
                t_i = time()

        if batch_ref_seqs:
            _handle_ref_alt_predictions(
                self.model,
                batch_ref_seqs,
                batch_alt_seqs,
                batch_ids,
                reporters,
                use_cuda=self.use_cuda)

        for r in reporters:
            r.write_to_file()


######
# END CODE FROM SELENE V0.4.4
######

def read_motif(motif_name, motif_file, is_log_motif):
    motif_dict = motif_fn.load_binding_motif_pssm(motif_file, is_log_motif)
    target_motif = motif_dict[motif_name]
    return target_motif


def scan_motif(seq, motif):
    """ match_score = sum of individual site likelihood * site weight
    site weight = 1 / site entropy

    TODO
    ----
    add weights for each sites, down-weigh un-informative sites
    """
    motif_len = motif.shape[0]
    seq_len = seq.shape[0]
    match_score = np.zeros(seq_len - motif_len + 1)
    motif[np.where(motif == 0)] = 1e-10
    for i in range(seq_len - motif_len + 1):
        this_seq = seq[i:(i + motif_len)]
        # ms = np.sum(this_seq*motif)
        ms = 0
        for s, m in zip(this_seq, motif):
            idx = np.where(s != 0)[0]
            ms += np.sum(s[idx] * np.log10(m[idx]))
        match_score[i] = ms
    return match_score


def normalize_sequence(seq):
    norm_seq = seq / (np.sum(seq) / seq.shape[0])
    normalizer = (np.sum(seq) / seq.shape[0])
    return norm_seq, normalizer


def matrix_to_seq(mat):
    seq = ''
    letter_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    for i in range(mat.shape[0]):
        idx = np.where(mat[i] > 0)[0].flatten()
        if len(idx) == 1:
            seq += letter_dict[idx[0]]
        else:
            seq += 'N'
    return seq


def saturate_permute_pred(seq, model, normalizer, lambda_pred):
    """permute every nucleotide to every letter on every position, and
    record the change in prediction score as vanilla substract perturbed
    """
    vanilla_pred = lambda_pred(model.predict(np.expand_dims(seq * normalizer, axis=0)))
    pseq_pred_change = np.zeros(seq.shape)
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            pseq = seq.copy()
            pseq[i, :] = 0
            pseq[i, j] = 1
            pseq_pred = lambda_pred(model.predict(np.expand_dims(pseq * normalizer, axis=0)))
            pseq_pred_change[i, j] = vanilla_pred - pseq_pred
    return pseq_pred_change


def saturate_permute_motif(seq, motif, normalizer):
    """permute every nucleotide to every letter on every position, and
    record the change in max motif match score as vanilla substract perturbed
    """
    vanilla_match_score = scan_motif(seq, motif)
    vanilla_max = np.max(vanilla_match_score)
    pseq_motif_change = np.zeros(seq.shape)
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            pseq = seq.copy()
            pseq[i, :] = 0
            pseq[i, j] = 1
            pseq_max = np.max(scan_motif(pseq, motif))
            pseq_motif_change[i, j] = vanilla_max - pseq_max
    # pseq_motif_change[np.where(seq>0)] = np.nan
    return pseq_motif_change


def evaluate_permute_acc_single_seq(pred_change, motif_change, seq, disrupt_cutoff=np.log10(10),
                                    nochange_cutoff=np.log10(2), auc_scorer=metrics.roc_auc_score):
    """measure the consistency of model perturbation with motif perturbation
    in a single input sequence, i.e. local prioritization of genome perturbations

    Note:
        for reduce sites, need to reverse the sign when computing AUROC/AUPR
    """
    # disrupt_idx = np.where( motif_change>=disrupt_cutoff )
    # nochange_idx = np.where( (motif_change > -enhance_reduce_cutoff) & (motif_change < enhance_reduce_cutoff) & (seq==0) )
    # nochange_idx = np.where( (motif_change==0) & (seq==0) )
    # enhance_idx = np.where( (motif_change> enhance_reduce_cutoff ) & (motif_change<disrupt_cutoff) )
    # reduce_idx = np.where(motif_change < -enhance_reduce_cutoff )
    disrupt_idx = np.where(motif_change >= disrupt_cutoff)
    nochange_idx = np.where((np.abs(motif_change) <= nochange_cutoff) & (seq == 0))

    try:
        auc_disrupt = auc_scorer(
            y_true=np.concatenate([np.ones(len(disrupt_idx[0])), np.zeros(len(nochange_idx[0]))]),
            y_score=np.concatenate([pred_change[disrupt_idx].flatten(), pred_change[nochange_idx].flatten()])
        )
    except ValueError:
        auc_disrupt = np.nan
    # try:
    #	auc_enhance = auc_scorer(
    #		y_true = np.concatenate([np.ones(len(enhance_idx[0])), np.zeros(len(nochange_idx[0])) ]),
    #		y_score = np.concatenate([pred_change[enhance_idx].flatten(), pred_change[nochange_idx].flatten() ])
    #		)
    # except ValueError:
    #	auc_enhance = np.nan
    # try:
    #	auc_reduce = auc_scorer(
    #		y_true = np.concatenate([np.ones(len(reduce_idx[0])), np.zeros(len(nochange_idx[0])) ]),
    #		y_score = np.concatenate([-pred_change[reduce_idx].flatten(), -pred_change[nochange_idx].flatten() ])
    #		)
    # except ValueError:
    #	auc_reduce = np.nan

    eval_dict = {
        "disrupt": pred_change[disrupt_idx],
        "nochange": pred_change[nochange_idx],
        # "enhance": pred_change[enhance_idx],
        # "reduce": pred_change[reduce_idx],
        "auc_disrupt": auc_disrupt,
        # "auc_enhance": auc_enhance,
        # "auc_reduce": auc_reduce
    }
    return eval_dict


def evaluate_permute_acc_aggregate(model_idx, model_performance_dict, data_idx_list, auc_scorer=metrics.roc_auc_score):
    """DEPRECATED: measure the consistency of model perturbation with motif perturbation
    by aggregating all input sequences, i.e. genome-wide prioritization of
    perturbed DNAs

    Note
    ----
    for reduce sites, need to reverse the sign when computing AUROC/AUPR
    """
    disrupt = []
    nochange = []
    # enhance = []
    # repress = []
    for i in data_idx_list:
        disrupt.extend(model_performance_dict[(model_idx, i)]['disrupt'])
        nochange.extend(model_performance_dict[(model_idx, i)]['nochange'])
    # enhance.extend(model_performance_dict[(model_idx, i)]['enhance'])
    # repress.extend(model_performance_dict[(model_idx, i)]['reduce'])

    auc_disrupt = auc_scorer(
        y_true=np.concatenate([np.ones(len(disrupt)), np.zeros(len(nochange))]),
        y_score=np.concatenate([disrupt, nochange])
    )
    # auc_enhance = auc_scorer(
    #	y_true = np.concatenate([np.ones(len(enhance)), np.zeros(len(nochange)) ]),
    #	y_score = np.concatenate([ enhance, nochange ])
    #	)
    # auc_reduce = auc_scorer(
    #	y_true = np.concatenate([np.ones(len(repress)), np.zeros(len(nochange)) ]),
    #	y_score = -np.concatenate([ repress, nochange ])
    #	)

    # return {'auc_disrupt': auc_disrupt, 'auc_enhance': auc_enhance, 'auc_reduce': auc_reduce}
    return {'auc_disrupt': auc_disrupt}


def get_motif_goldstandard(X_seq, motif):
    """Given a set of sequence inputs, systematically perturb
    every letter every position, and record as gold-standard
    """

    motif_change_dict = {}
    for i in range(X_seq.shape[0]):
        sys.stdout.write("%i " % i)
        sys.stdout.flush()
        x, normalizer = normalize_sequence(X_seq[i])
        motif_change = saturate_permute_motif(x, motif, normalizer)
        motif_change_dict[i] = motif_change
    return motif_change_dict


def models_sensitivity_motif(model_dict, x_seq, motif_change_dict, auc_scorer=metrics.roc_auc_score,
                             lambda_pred=lambda x: x.flatten(), **kwargs):
    """evaluate the sensitivity for a set of models based on input x and motif-change as
    gold-standard.

    Sensitivity defined as responsiveness and robustness of models for perturbations in input
    features.

    Returns
    -------
    defaultdict(dict) : model index -> {eval_attr: eval_val}

    Note
    -----
    for reduce sites, need to reverse the sign for auc_scorer
    """
    agg_auc_scorer = lambda a, b: auc_scorer(
        y_true=np.concatenate([np.ones(len(a)), np.zeros(len(b))]),
        y_score=np.concatenate([a, b])
    )
    # not picklable; do not use
    # model_performance_dict = defaultdict(lambda:defaultdict(list))
    model_performance_dict = {}
    model_count = 0
    total_model_count = len(model_dict)
    for model_idx in model_dict:
        model_count += 1
        sys.stdout.write("%i/%i analyzing model %i." % (model_count, total_model_count, model_idx))
        sys.stdout.flush()
        seq_count = 0
        start_time = time.time()
        model_performance_dict[model_idx] = defaultdict(list)
        for seq_idx in motif_change_dict:
            seq_count += 1
            if seq_count / float(len(motif_change_dict)) > 0.1:
                sys.stdout.write('.')
                sys.stdout.flush()
                seq_count = 0
            model = model_dict[model_idx]
            motif_change = motif_change_dict[seq_idx]
            x1, normalizer = normalize_sequence(x_seq[seq_idx])
            pred_change = saturate_permute_pred(x1, model, normalizer, lambda_pred)
            eval_dict = evaluate_permute_acc_single_seq(pred_change, motif_change, x1, auc_scorer=auc_scorer, **kwargs)
            # extend lists
            model_performance_dict[model_idx]['disrupt'].extend(eval_dict['disrupt'])
            model_performance_dict[model_idx]['nochange'].extend(eval_dict['nochange'])
            # model_performance_dict[model_idx]['enhance'].extend(eval_dict['enhance'])
            # model_performance_dict[model_idx]['reduce'].extend(eval_dict['reduce'])
            # append measurements
            model_performance_dict[model_idx]['auc_disrupt'].append(eval_dict['auc_disrupt'])
        # model_performance_dict[model_idx]['auc_enhance'].append(eval_dict['auc_enhance'])
        # model_performance_dict[model_idx]['auc_reduce'].append(eval_dict['auc_reduce'])

        model_performance_dict[model_idx]['auc_agg_disrupt'] = \
            agg_auc_scorer(
                model_performance_dict[model_idx]['disrupt'],
                model_performance_dict[model_idx]['nochange']
            )
        # model_performance_dict[model_idx]['auc_agg_enhance'] = \
        #	agg_auc_scorer(
        #		model_performance_dict[model_idx]['enhance'],
        #		model_performance_dict[model_idx]['nochange']
        #	)
        # model_performance_dict[model_idx]['auc_agg_reduce'] = \
        #	agg_auc_scorer(
        #		- np.array(model_performance_dict[model_idx]['reduce']),
        #		model_performance_dict[model_idx]['nochange']
        #	)
        elapsed_time = time.time() - start_time
        sys.stdout.write("used %.3fs\n" % elapsed_time)
        sys.stdout.flush()
    return model_performance_dict


def rescore_sensitivity_motif(model_performance_dict, auc_scorer=metrics.roc_auc_score):
    """re-score

    Note
    -----
    for reduce sites, need to reverse the sign for auc_scorer
    """
    agg_auc_scorer = lambda a, b: auc_scorer(
        y_true=np.concatenate([np.ones(len(a)), np.zeros(len(b))]),
        y_score=np.concatenate([a, b])
    )
    # not picklable; do not use
    # model_performance_dict = defaultdict(lambda:defaultdict(list))
    model_count = 0
    total_model_count = len(model_performance_dict)
    for model_idx in model_performance_dict:
        model_count += 1
        sys.stdout.write("%i/%i analyzing model %i." % (model_count, total_model_count, model_idx))
        sys.stdout.flush()
        seq_count = 0
        start_time = time.time()
        model_performance_dict[model_idx]['auc_agg_disrupt'] = \
            agg_auc_scorer(
                model_performance_dict[model_idx]['disrupt'],
                model_performance_dict[model_idx]['nochange']
            )
        model_performance_dict[model_idx]['auc_agg_enhance'] = \
            agg_auc_scorer(
                model_performance_dict[model_idx]['enhance'],
                model_performance_dict[model_idx]['nochange']
            )
        model_performance_dict[model_idx]['auc_agg_reduce'] = \
            agg_auc_scorer(
                - np.array(model_performance_dict[model_idx]['reduce']),
                model_performance_dict[model_idx]['nochange']
            )
        model_performance_dict[model_idx]['auc_agg_disrupt_enhance'] = \
            agg_auc_scorer(
                np.concatenate(
                    [
                        model_performance_dict[model_idx]['disrupt'],
                        model_performance_dict[model_idx]['enhance']
                    ]),
                model_performance_dict[model_idx]['nochange']
            )

        elapsed_time = time.time() - start_time
        sys.stdout.write("used %.3fs\n" % elapsed_time)
        sys.stdout.flush()
    return model_performance_dict


def summarize_sensitivity_motif(model_performance_dict):
    """summarize a set of models performance into global/local
    motif accuracy
    """
    local_disrupt_auc = []
    local_enhance_auc = []
    local_reduce_auc = []
    global_disrupt_auc = []
    global_enhance_auc = []
    global_reduce_auc = []
    for model_idx in model_performance_dict:
        local_disrupt_auc.extend(model_performance_dict[model_idx]['auc_disrupt'])
        local_enhance_auc.extend(model_performance_dict[model_idx]['auc_enhance'])
        local_reduce_auc.extend(model_performance_dict[model_idx]['auc_reduce'])
        global_enhance_auc.append(model_performance_dict[model_idx]['auc_agg_enhance'])
        global_reduce_auc.append(model_performance_dict[model_idx]['auc_agg_reduce'])
        global_disrupt_auc.append(model_performance_dict[model_idx]['auc_agg_disrupt'])

    local_disrupt_auc = np.array(local_disrupt_auc)[~np.isnan(local_disrupt_auc)]
    local_enhance_auc = np.array(local_enhance_auc)[~np.isnan(local_enhance_auc)]
    local_reduce_auc = np.array(local_reduce_auc)[~np.isnan(local_reduce_auc)]
    global_disrupt_auc = np.array(global_disrupt_auc)[~np.isnan(global_disrupt_auc)]
    global_enhance_auc = np.array(global_enhance_auc)[~np.isnan(global_enhance_auc)]
    global_reduce_auc = np.array(global_reduce_auc)[~np.isnan(global_reduce_auc)]

    df = pd.DataFrame({
        'auc': np.concatenate([
            local_disrupt_auc, local_enhance_auc, local_reduce_auc,
            global_disrupt_auc, global_enhance_auc, global_reduce_auc
        ]),
        'type': \
            ['local'] * (local_disrupt_auc.shape[0] + local_enhance_auc.shape[0] + local_reduce_auc.shape[0]) + \
            ['global'] * (global_disrupt_auc.shape[0] + global_enhance_auc.shape[0] + global_reduce_auc.shape[0]),
        'group': \
            ['disrupt'] * local_disrupt_auc.shape[0] + ['enhance'] * local_enhance_auc.shape[0] + ['reduce'] *
            local_reduce_auc.shape[0] + \
            ['disrupt'] * global_disrupt_auc.shape[0] + ['enhance'] * global_enhance_auc.shape[0] + ['reduce'] *
            global_reduce_auc.shape[0]
    })

    # return {'local_disrupt': local_disrupt_auc, 'global_disrupt': global_disrupt_auc,
    #	'local_enhance': local_enhance_auc, 'local_reduce': local_reduce_auc,
    #	'global_enhance': global_enhance_auc, 'global_reduce': global_reduce_auc}
    return df
