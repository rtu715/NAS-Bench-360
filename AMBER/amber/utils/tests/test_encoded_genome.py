import numpy
import unittest

from amber.utils.sequences import EncodedGenome


class TestEncodedGenome(unittest.TestCase):
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
        return EncodedGenome("amber/utils/tests/files/small_genome.fa", in_memory=self.in_memory)

    def test_load_mixed_case_sequence(self):
        expected = [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [.25, .25, .25, .25],
                [.25, .25, .25, .25],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0]
            ]
        expected = numpy.array(expected)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq0", 0, 10)
        self.assertSequenceEqual(expected.tolist(), observed.tolist())


    def test_load_sequence0(self):
        expected = [[0, 1, 0, 0]]
        expected = numpy.array(expected)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq0", 5, 6)
        self.assertSequenceEqual(expected.tolist(), observed.tolist())

    def test_load_sequence1(self):
        expected = [[0, 0, 0, 1],
                    [0, 0, 1, 0]]
        expected = numpy.array(expected)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq2", 22, 24)
        self.assertSequenceEqual(expected.tolist(), observed.tolist())

    def test_load_rc(self):
        expected = "CCAAGGNNTT"
        expected = [[0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [.25, .25, .25, .25],
                    [.25, .25, .25, .25],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]]
        expected = numpy.array(expected)
        g = self._get_small_genome()
        observed = g.get_sequence_from_coords("seq0", 0, 10, "-")
        self.assertSequenceEqual(expected.tolist(), observed.tolist())

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


class TestEncodedGenomeInMemory(TestEncodedGenome):
    def setUp(self):
        super(TestEncodedGenomeInMemory, self).setUp()
        self.in_memory = True

    def test_in_memory(self):
        self.assertTrue(self.in_memory)

    def test_genome_in_memory(self):
        g = self._get_small_genome()
        self.assertTrue(g.in_memory)

