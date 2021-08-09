# Author: Evan M. Cofer
# Created on June 7, 2020
"""
This module provides the `Sequence` class, which is an abstract class
that defines the interface for loading biological sequence data.

TODO
----
- Abstract away the PYFAIDX interface, so that inheritence and extension of the
    genome class is more straightforward.
"""
import abc

import h5py
import numpy
try:
    import pyfaidx
except ModuleNotFoundError:
    pass

_DNA_COMP_TABLE = str.maketrans("ATCGN", "TAGCN")
_STORE_TYPE = numpy.half


class Sequence(metaclass=abc.ABCMeta):
    """This class represents a source of sequence data, which can be
    fetched by querying different coordinates.
    """
    @abc.abstractmethod
    def __len__(self):
        """Number of queryable positions in the sequence.

        Returns
        -------
        int
            The number of queryable positions.

        """
        pass

    @abc.abstractmethod
    def coords_are_valid(self, *args, **kwargs):
        """Checks if the queried coordinates are valid.

        Returns
        -------
        bool
            `True` if the coordinates are valid, otherwise `False`.
        """
        pass

    @abc.abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """Fetches a string representation of a sequence at
        the specified coordinates.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates. Behavior is undefined for invalid
            coordinates.
        """
        pass


class Genome(Sequence):
    """This class allows the user to a query a potentially file-backed
    genome by coordinate. It is essentially a wrapper around the
    `pyfaidx.Fasta` class.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : pyfaidx.Fasta or dict
        The FASTA file containing the genome sequence. Alternatively,
        this can be a `dict` object mapping chromosomes to sequences that
        stores the file in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    """
    def __init__(self, input_path, in_memory=False):
        """
        Constructs a new `Genome` object.
        """
        super(Genome, self).__init__()
        self.in_memory = in_memory
        if in_memory is True:
            fasta = pyfaidx.Fasta(input_path)
            self.data = {k: str(fasta[k][:].seq).upper() for k in fasta.keys()}
            fasta.close()
        else:
            self.data = pyfaidx.Fasta(input_path)
        self.chrom_len_dict = {k: len(self.data[k]) for k in self.data.keys()}

    def __len__(self):
        """Number of queryable positions in the genome.

        Returns
        -------
        int
            The number of queryable positions.
        """
        return sum(self.chrom_len_dict.values())

    def coords_are_valid(self, chrom, start, end, strand="+"):
        """Checks if the queried coordinates are valid.

        Parameters
        ----------
        chrom : str
            The chromosome to query from.
        start : int
            The first position in the queried corodinates.
        end : int
            One past the last position in the queried coordinates.
        strand : str
            Strand of sequence to draw from.

        Returns
        -------
        bool
            `True` if the coordinates are valid, otherwise `False`.
        """
        if chrom not in self.chrom_len_dict:
            return False
        elif start < 0 or end <= 0:
            return False
        elif start >= end:
            return False
        elif end > self.chrom_len_dict[chrom]:
            return False
        elif start >= self.chrom_len_dict[chrom]:
            return False
        elif strand not in {"+", "-"}:
            return False
        else:
            return True

    def get_sequence_from_coords(self, chrom, start, end, strand="+"):
        """Fetches a string representation of a sequence at
        the specified coordinates.


        Parameters
        ----------
        chrom : str
            Chromosome to query from.
        start : int
            First position in queried sequence.
        end : int
            One past the last position in the queried sequence.
        strand : str
            The strand to sample from.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates.

        Raises
        ------
        IndexError
            If the coordinates are not valid.
        """
        if self.coords_are_valid(chrom, start, end, strand):
            x = self.data[chrom][start:end]
            if not self.in_memory:
                x = str(x.seq).upper()
            if strand == "-":
                x = x.translate(_DNA_COMP_TABLE)[::-1]
            return x
        else:
            s = "Specified coordinates ({} to {} on \"{}\", strand of \"{}\") are invalid!".format(
                start, end, chrom, strand)
            raise IndexError(s)


class Encoding(metaclass=abc.ABCMeta):
    """This class is a mostly-abstract class used to represent some dataset that
    should be transformed with an encoding.
    """
    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        """
        Method to encode some input.
        """
        pass


class EncodedSequence(Encoding, Sequence):
    """Mixin of `Encoding` and `Sequence` to define the approach for
    encoding biological sequence data.
    """
    @property
    @abc.abstractmethod
    def ALPHABET_TO_ARRAY(self):
        """
        The alphabet used to encode the input sequence.
        """
        pass

    def encode(self, s):
        """Encodes a string with a numpy array.

        Parameters
        ----------
        s : str
            The string to encode.

        Returns
        -------
        numpy.ndarray
            An array with the encoded string.
        """
        ret = list()
        for i in range(len(s)):
            ret.append(self.ALPHABET_TO_ARRAY[s[i]].copy())
        ret = numpy.stack(ret)
        return ret

    def get_sequence_from_coords(self, *args, **kwargs):
        """Fetches an encoded sequence at the specified coordinates.

        Returns
        -------
        numpy.ndarray
            The numpy array encoding the queried sequence.
        """
        return self.encode(
            super(EncodedSequence, self).get_sequence_from_coords(
                *args, **kwargs))


class EncodedGenome(EncodedSequence, Genome):
    """This class allows the user to a query a potentially file-backed
    genome by coordinate. It is essentially a wrapper around the
    `pyfaidx.Fasta` class. The returned values have been encoded as numpy
    arrays.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : pyfaidx.Fasta or dict
        The FASTA file containing the genome sequence. Alternatively,
        this can be a `dict` object mapping chromosomes to sequences that
        stores the file in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    ALPHABET_TO_ARRAY : dict
        A mapping from characters in the genome to their
        `numpy.ndarray` representations.
    """
    ALPHABET_TO_ARRAY = dict(A=numpy.array([1, 0, 0, 0], dtype=_STORE_TYPE),
                             C=numpy.array([0, 1, 0, 0], dtype=_STORE_TYPE),
                             G=numpy.array([0, 0, 1, 0], dtype=_STORE_TYPE),
                             T=numpy.array([0, 0, 0, 1], dtype=_STORE_TYPE),
                             N=numpy.array([.25, .25, .25, .25], dtype=_STORE_TYPE))
    """
    A dictionary mapping possible characters in the genome to
    their `numpy.ndarray` representations.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs a new `EncodedGenome` object.
        """
        super(EncodedGenome, self).__init__(*args, **kwargs)
        if self.in_memory is True: # Pre-encode the genome if storing in memory.
            for k, v in self.data.items():
                print(k, flush=True)
                self.data[k] = numpy.zeros((len(v), 4), dtype=_STORE_TYPE)
                for i in range(self.data[k].shape[0]):
                    self.data[k][i, :] = self.ALPHABET_TO_ARRAY[v[i]]
                #self.data[k] = self.encode(self.data[k])#.astype(_STORE_TYPE)

    def get_sequence_from_coords(self, chrom, start, end, strand="+"):
        """Fetches a string representation of a sequence at
        the specified coordinates.


        Parameters
        ----------
        chrom : str
            Chromosome to query from.
        start : int
            First position in queried sequence.
        end : int
            One past the last position in the queried sequence.
        strand : str
            The strand to sample from.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates.

        Raises
        ------
        IndexError
            If the coordinates are not valid.
        """
        if self.in_memory is True:
            if self.coords_are_valid(chrom, start, end, strand):
                x = self.data[chrom][start:end].copy()
                if strand == "-":
                    x = numpy.flip(numpy.flip(x, 0), 1)
                return x.astype(numpy.float64)
            else:
                s = "Specified coordinates ({} to {} on \"{}\", strand of \"{}\") are invalid!".format(
                     start, end, chrom, strand)
                raise IndexError(s)
        else:
            return self.encode(
                super(EncodedSequence, self).get_sequence_from_coords(
                    chrom=chrom, start=start, end=end, strand=strand)).astype(numpy.float64)


class HDF5Genome(Genome):
    """This class allows the user to query a Genome stored in an HDF5 file.

    Parameters
    ----------
    input_path : str
        Path to an HDF5 file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : h5py.File
        The HDF5 file containing the genome sequence. Alternatively,
        this can be a `dict` object mapping chromosomes to sequences that
        stores the file in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    """
    def __init__(self, input_path, in_memory=False):
        """
        Constructs a new `HDF5Genome` object.
        """
        self.in_memory = in_memory
        if in_memory is True:
            f = h5py.File(input_path, "r")
            self.data = dict()
            for k in f.keys():
                self.data[k] = f[k][:].tostring().decode("UTF-8")
            f.close()
        else:
            self.data = h5py.File(input_path, "r")
        self.chrom_len_dict = {k: len(self.data[k]) for k in self.data.keys()}

    def get_sequence_from_coords(self, chrom, start, end, strand="+"):
        """Fetches a string representation of a sequence at
        the specified coordinates.


        Parameters
        ----------
        chrom : str
            Chromosome to query from.
        start : int
            First position in queried sequence.
        end : int
            One past the last position in the queried sequence.
        strand : str
            The strand to sample from.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates.

        Raises
        ------
        IndexError
            If the coordinates are not valid.
        """
        if self.coords_are_valid(chrom, start, end, strand):
            x = self.data[chrom][start:end]
            if not self.in_memory:
                x = x.tostring().decode("UTF-8")
            if strand == "-":
                x = x.translate(_DNA_COMP_TABLE)[::-1]
            return x
        else:
            s = "Specified coordinates ({} to {} on \"{}\", strand of \"{}\") are invalid!".format(
                start, end, chrom, strand)
            raise IndexError(s)


class EncodedHDF5Genome(EncodedGenome):
    """
    This class allows the user to specify an HDF5-backed encoded genome.

    Parameters
    ----------
    input_path : str
        Path to the HDF5 file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : h5py.File or dict
        The HDF5 file pointer or a dict containing the sequences in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    """
    def __init__(self, input_path, in_memory=False):
        """
        Constructs a new `EncodedHDF5Genome` object.
        """
        self.in_memory = in_memory
        if in_memory is True:
            f = h5py.File(input_path, "r")
            self.data = dict()
            for k in f.keys():
                self.data[k] = f[k][()]
            f.close()
        else:
            self.data = h5py.File(input_path, "r")
        self.chrom_len_dict = {k: self.data[k].shape[0] for k
                               in self.data.keys()}

    def get_sequence_from_coords(self, chrom, start, end, strand="+"):
        """Fetches an array representation of a sequence at
        the specified coordinates.


        Parameters
        ----------
        chrom : str
            Chromosome to query from.
        start : int
            First position in queried sequence.
        end : int
            One past the last position in the queried sequence.
        strand : str
            The strand to sample from.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates.

        Raises
        ------
        IndexError
            If the coordinates are not valid.
        """
        if self.coords_are_valid(chrom, start, end, strand):
            if self.in_memory:
                x = self.data[chrom][start:end].copy()
            else:
                x = numpy.zeros((end - start, 4), dtype=_STORE_TYPE)
                self.data[chrom].read_direct(x, numpy.s_[start:end], numpy.s_[0:x.shape[0]])
            if strand == "-":
                x = numpy.flip(numpy.flip(x, 0), 1)
            return x.astype(numpy.float64)
        else:
            s = "Specified coordinates ({} to {} on \"{}\", strand of \"{}\") are invalid!".format(start, end, chrom, strand)
            raise IndexError(s)

    def close(self):
        """
        Close the file connection to HDF5
        """
        self.data.close()

