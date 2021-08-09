import numpy
import unittest

from amber.utils.sequences import HDF5Genome

class TestHDF5Genome(unittest.TestCase):
    def setUp(self):
        self.bases = ['A', 'C', 'G', 'T']
        self.bases_to_arr = dict(A=numpy.array([1, 0, 0, 0]),
                                 C=numpy.array([0, 1, 0, 0]),
                                 G=numpy.array([0, 0, 1, 0]),
                                 T=numpy.array([0, 0, 0, 1]),
                                 N=numpy.array([.25, .25, .25, .25]))

        self.chrom_to_lens = {"seq0": 10,
                              "seq1": 25,
                              "seq2": 24}
        self.in_memory = False

    def _get_small_genome(self):
        return HDF5Genome("amber/utils/tests/files/small_genome.h5", in_memory=self.in_memory)

    def test_load_mixed_case_sequence(self):
        expected = "AANNCCTTGG"
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq0", 0, 10)
        self.assertEqual(expected, observed)

    def test_load_rc(self):
        expected = "CCAAGGNNTT"
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq0", 0, 10, "-")
        self.assertEqual(expected, observed)

    def test_load_sequence0(self):
        expected = "C"
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq1", 5, 6)
        self.assertEqual(expected, observed)

    def test_load_sequence1(self):
        expected = "A"
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq2", 0, 1)
        self.assertEqual(expected, observed)

    def test_load_sequence2(self):
        expected = ("A" * 5) + ("C" * 5) + ("T" * 5 ) + ("G" * 5) + ("N" * 5)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq1", 0, 25)
        self.assertEqual(expected, observed)

    def test_load_sequence3(self):
        expected = "ACTG" * 6
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq2", 0, 24)
        self.assertEqual(expected, observed)

    def test_load_sequence4(self):
        expected = ("T" * 5 ) + ("G" * 5) + ("N" * 5)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq1", 10, 25)
        self.assertEqual(expected, observed)

    def test_length(self):
        expected = sum(self.chrom_to_lens.values())
        g = self._get_small_genome()
        observed = len(g)
        self.assertEqual(expected, observed)

    def test_coords_flipped(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 2, 1)
        self.assertFalse(observed)

    def test_end_coord_too_large(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 9, 11)
        self.assertFalse(observed)

    def test_both_coords_too_large(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 11, 14)
        self.assertFalse(observed)

    def test_coords_too_large_negative(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", -100, -99)
        self.assertFalse(observed)

    def test_coords_negative_flipped(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", -1, -2)
        self.assertFalse(observed)

    def test_coords_negative(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", -2, -1)
        self.assertFalse(observed)

    def test_bad_chrom(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq-1", 1, 2)
        self.assertFalse(observed)

    def test_too_small(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 1, 1)
        self.assertFalse(observed)

    def test_first_coord(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 0, 1)
        self.assertTrue(observed)

    def test_last_coord(self):
        g = self._get_small_genome()
        observed = g.coords_are_valid("seq0", 9, 10)
        self.assertTrue(observed)

    def test_in_memory(self):
        self.assertFalse(self.in_memory)

    def test_genome_in_memory(self):
        g = self._get_small_genome()
        self.assertFalse(g.in_memory)


class TestHDF5GenomeInMemory(TestHDF5Genome):
    def setUp(self):
        super(TestHDF5GenomeInMemory, self).setUp()
        self.in_memory = True

    def test_in_memory(self):
        self.assertTrue(self.in_memory)

    def test_genome_in_memory(self):
        g = self._get_small_genome()
        self.assertTrue(g.in_memory)


