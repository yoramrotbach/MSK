"""
Tests for msk.py

Run with:
    python -m pytest test_msk.py -v
or:
    python test_msk.py
"""

import unittest
import numpy as np
from msk import msk_modulate, awgn, msk_demodulate, compute_ber, ber_vs_snr, iq_samples

# Shared default parameters used across tests
BIT_RATE     = 1e3
CARRIER_FREQ = 2e3
SAMPLE_RATE  = 1e5
SPB          = int(SAMPLE_RATE / BIT_RATE)   # samples per bit = 100


class TestMskModulate(unittest.TestCase):

    def test_output_length_matches_data(self):
        data = [1, 0, 1, 1, 0]
        t, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        expected = len(data) * SPB
        self.assertEqual(len(t), expected)
        self.assertEqual(len(sig), expected)

    def test_time_array_starts_at_zero(self):
        t, _ = msk_modulate([1, 0], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertAlmostEqual(t[0], 0.0)

    def test_time_step_equals_inverse_sample_rate(self):
        t, _ = msk_modulate([1, 0, 1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        dt = t[1] - t[0]
        self.assertAlmostEqual(dt, 1 / SAMPLE_RATE, places=10)

    def test_amplitude_bounded(self):
        _, sig = msk_modulate([1, 0, 1, 0, 1, 0, 1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertLessEqual(np.max(np.abs(sig)), 1.0 + 1e-9)

    def test_single_bit_one_is_cosine_at_high_freq(self):
        _, sig = msk_modulate([1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        freq_high = CARRIER_FREQ + BIT_RATE / 2
        t = np.arange(SPB) / SAMPLE_RATE
        expected = np.cos(2 * np.pi * freq_high * t)
        np.testing.assert_allclose(sig, expected, atol=1e-9)

    def test_single_bit_zero_is_cosine_at_low_freq(self):
        _, sig = msk_modulate([0], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        freq_low = CARRIER_FREQ - BIT_RATE / 2
        t = np.arange(SPB) / SAMPLE_RATE
        expected = np.cos(2 * np.pi * freq_low * t)
        np.testing.assert_allclose(sig, expected, atol=1e-9)

    def test_signal_power_is_constant_across_bits(self):
        # A pure sinusoid has mean power = 0.5; each bit should match this.
        data = [1, 0, 1, 0, 1]
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        for i, _ in enumerate(data):
            s, e = i * SPB, (i + 1) * SPB
            power = np.mean(sig[s:e] ** 2)
            self.assertAlmostEqual(power, 0.5, delta=0.01,
                msg=f"Bit {i} power {power:.4f} deviates from expected 0.5")

    def test_empty_data_returns_empty_arrays(self):
        t, sig = msk_modulate([], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertEqual(len(t), 0)
        self.assertEqual(len(sig), 0)

    def test_single_bit_sequence(self):
        t, sig = msk_modulate([1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertEqual(len(sig), SPB)

    def test_all_ones(self):
        data = [1] * 10
        t, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertEqual(len(sig), 10 * SPB)
        self.assertLessEqual(np.max(np.abs(sig)), 1.0 + 1e-9)

    def test_all_zeros(self):
        data = [0] * 10
        t, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertEqual(len(sig), 10 * SPB)
        self.assertLessEqual(np.max(np.abs(sig)), 1.0 + 1e-9)

    def test_different_bit_rates_produce_different_lengths(self):
        data = [1, 0, 1]
        _, sig_slow = msk_modulate(data, 500,  CARRIER_FREQ, SAMPLE_RATE)
        _, sig_fast = msk_modulate(data, 2000, CARRIER_FREQ, SAMPLE_RATE)
        self.assertGreater(len(sig_slow), len(sig_fast))


class TestAwgn(unittest.TestCase):

    def setUp(self):
        _, self.sig = msk_modulate([1, 0, 1, 1, 0, 0, 1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)

    def test_output_length_unchanged(self):
        noisy = awgn(self.sig, snr_db=10)
        self.assertEqual(len(noisy), len(self.sig))

    def test_output_is_different_from_input(self):
        noisy = awgn(self.sig, snr_db=10)
        self.assertFalse(np.allclose(noisy, self.sig))

    def test_high_snr_signal_nearly_unchanged(self):
        noisy = awgn(self.sig, snr_db=60)
        np.testing.assert_allclose(noisy, self.sig, atol=0.01)

    def test_snr_is_approximately_correct(self):
        np.random.seed(0)
        target_snr_db = 10.0
        noisy = awgn(self.sig, snr_db=target_snr_db)
        noise = noisy - self.sig
        measured_snr = 10 * np.log10(np.mean(self.sig ** 2) / np.mean(noise ** 2))
        self.assertAlmostEqual(measured_snr, target_snr_db, delta=1.5)

    def test_low_snr_adds_substantial_noise(self):
        noisy = awgn(self.sig, snr_db=0)
        noise_power = np.mean((noisy - self.sig) ** 2)
        signal_power = np.mean(self.sig ** 2)
        # At 0 dB SNR, noise power ≈ signal power
        self.assertGreater(noise_power, signal_power * 0.5)


class TestMskDemodulate(unittest.TestCase):

    def _roundtrip(self, data):
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        return msk_demodulate(sig, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)

    def test_clean_recovery_all_ones(self):
        data = [1] * 8
        self.assertEqual(self._roundtrip(data), data)

    def test_clean_recovery_all_zeros(self):
        data = [0] * 8
        self.assertEqual(self._roundtrip(data), data)

    def test_clean_recovery_alternating(self):
        data = [1, 0, 1, 0, 1, 0, 1, 0]
        self.assertEqual(self._roundtrip(data), data)

    def test_clean_recovery_standard_sequence(self):
        data = [1, 0, 1, 1, 0, 0, 1]
        self.assertEqual(self._roundtrip(data), data)

    def test_clean_recovery_random_long_sequence(self):
        rng = np.random.default_rng(7)
        data = rng.integers(0, 2, 50).tolist()
        self.assertEqual(self._roundtrip(data), data)

    def test_output_length_matches_input(self):
        data = [1, 0, 1, 1, 0]
        decoded = self._roundtrip(data)
        self.assertEqual(len(decoded), len(data))

    def test_high_snr_near_perfect_recovery(self):
        rng = np.random.default_rng(99)
        data = rng.integers(0, 2, 100).tolist()
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        noisy = awgn(sig, snr_db=20)
        decoded = msk_demodulate(noisy, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        ber = compute_ber(data, decoded)
        self.assertLess(ber, 0.05, f"BER at 20 dB SNR too high: {ber:.3f}")

    def test_output_contains_only_binary_values(self):
        data = [1, 0, 1, 0, 1]
        decoded = self._roundtrip(data)
        for bit in decoded:
            self.assertIn(bit, (0, 1))

    def test_single_bit_one(self):
        self.assertEqual(self._roundtrip([1]), [1])

    def test_single_bit_zero(self):
        self.assertEqual(self._roundtrip([0]), [0])


class TestComputeBer(unittest.TestCase):

    def test_identical_sequences_zero_ber(self):
        bits = [1, 0, 1, 1, 0]
        self.assertEqual(compute_ber(bits, bits), 0.0)

    def test_all_wrong_ber_is_one(self):
        orig    = [1, 1, 1, 1]
        decoded = [0, 0, 0, 0]
        self.assertEqual(compute_ber(orig, decoded), 1.0)

    def test_half_wrong_ber_is_half(self):
        orig    = [1, 1, 0, 0]
        decoded = [1, 1, 1, 1]
        self.assertAlmostEqual(compute_ber(orig, decoded), 0.5)

    def test_one_error_out_of_four(self):
        orig    = [1, 0, 1, 1]
        decoded = [1, 0, 0, 1]
        self.assertAlmostEqual(compute_ber(orig, decoded), 0.25)

    def test_decoded_longer_than_original_is_trimmed(self):
        orig    = [1, 0, 1]
        decoded = [1, 0, 1, 0, 0, 0]
        self.assertEqual(compute_ber(orig, decoded), 0.0)

    def test_returns_float(self):
        result = compute_ber([1, 0], [1, 0])
        self.assertIsInstance(result, float)

    def test_single_bit_match(self):
        self.assertEqual(compute_ber([1], [1]), 0.0)

    def test_single_bit_mismatch(self):
        self.assertEqual(compute_ber([1], [0]), 1.0)


class TestBerVsSnr(unittest.TestCase):

    def test_output_lengths_match_snr_range(self):
        snr_range = np.arange(0, 10, 2)
        snrs, bers = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=200)
        self.assertEqual(len(snrs), len(snr_range))
        self.assertEqual(len(bers), len(snr_range))

    def test_ber_values_in_valid_range(self):
        snr_range = np.arange(0, 15, 5)
        _, bers = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=200)
        for ber in bers:
            self.assertGreaterEqual(ber, 0.0)
            self.assertLessEqual(ber, 1.0)

    def test_high_snr_gives_low_ber(self):
        _, bers = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, [20], n_bits=500)
        self.assertLess(bers[0], 0.05, f"BER at 20 dB should be near 0, got {bers[0]:.3f}")

    def test_ber_generally_decreases_with_snr(self):
        snr_range = np.arange(0, 16, 3)
        _, bers = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=500, seed=0)
        # Average BER in upper half of SNR range should be lower than lower half
        mid = len(bers) // 2
        self.assertLessEqual(np.mean(bers[mid:]), np.mean(bers[:mid]) + 0.1)

    def test_seed_produces_reproducible_results(self):
        snr_range = [10]
        _, bers1 = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=200, seed=123)
        _, bers2 = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=200, seed=123)
        self.assertEqual(bers1, bers2)

    def test_snrs_returned_unchanged(self):
        snr_range = np.array([2.0, 5.0, 8.0])
        snrs, _ = ber_vs_snr(BIT_RATE, CARRIER_FREQ, SAMPLE_RATE, snr_range, n_bits=100)
        np.testing.assert_array_equal(snrs, snr_range)


class TestIqSamples(unittest.TestCase):

    def setUp(self):
        self.data = [1, 0, 1, 1, 0, 0, 1]
        _, self.sig = msk_modulate(self.data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)

    def test_output_length_equals_number_of_bits(self):
        I, Q = iq_samples(self.sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        self.assertEqual(len(I), len(self.data))
        self.assertEqual(len(Q), len(self.data))

    def test_returns_numpy_arrays(self):
        I, Q = iq_samples(self.sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        self.assertIsInstance(I, np.ndarray)
        self.assertIsInstance(Q, np.ndarray)

    def test_i_and_q_have_same_length(self):
        I, Q = iq_samples(self.sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        self.assertEqual(len(I), len(Q))

    def test_clean_signal_iq_points_lie_near_unit_circle(self):
        I, Q = iq_samples(self.sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        magnitudes = np.sqrt(I ** 2 + Q ** 2)
        # All I/Q points from a clean constant-envelope signal should have similar magnitude
        self.assertLess(np.std(magnitudes), np.mean(magnitudes) * 0.5)

    def test_noisy_signal_iq_more_spread_than_clean(self):
        I_clean, Q_clean = iq_samples(self.sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        noisy = awgn(self.sig, snr_db=5)
        I_noisy, Q_noisy = iq_samples(noisy, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        spread_clean = np.std(I_clean) + np.std(Q_clean)
        spread_noisy = np.std(I_noisy) + np.std(Q_noisy)
        self.assertGreater(spread_noisy, spread_clean)

    def test_single_bit_signal(self):
        _, sig1 = msk_modulate([1], BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        I, Q = iq_samples(sig1, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        self.assertEqual(len(I), 1)
        self.assertEqual(len(Q), 1)


class TestEndToEnd(unittest.TestCase):

    def test_modulate_add_noise_demodulate_ber(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, 200).tolist()
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        noisy   = awgn(sig, snr_db=15)
        decoded = msk_demodulate(noisy, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        ber     = compute_ber(data, decoded)
        self.assertLess(ber, 0.05)

    def test_zero_noise_roundtrip_is_perfect(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, 50).tolist()
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        decoded = msk_demodulate(sig, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        self.assertEqual(compute_ber(data, decoded), 0.0)

    def test_iq_samples_consistent_with_modulated_length(self):
        data = [1, 0, 1, 0, 1, 0]
        _, sig = msk_modulate(data, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)
        I, Q = iq_samples(sig, CARRIER_FREQ, SAMPLE_RATE, BIT_RATE)
        self.assertEqual(len(I), len(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
