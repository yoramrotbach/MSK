import numpy as np


def msk_modulate(data, bit_rate, carrier_freq, sample_rate):
    """MSK modulation with continuous phase."""
    bit_duration = 1 / bit_rate
    samples_per_bit = int(sample_rate * bit_duration)
    n_samples = len(data) * samples_per_bit
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)
    freq_high = carrier_freq + 1 / (2 * bit_duration)
    freq_low  = carrier_freq - 1 / (2 * bit_duration)
    phase = 0.0

    for idx, bit in enumerate(data):
        freq = freq_high if bit == 1 else freq_low
        s, e = idx * samples_per_bit, (idx + 1) * samples_per_bit
        signal[s:e] = np.cos(2 * np.pi * freq * t[s:e] + phase)
        phase += 2 * np.pi * freq * bit_duration

    return t, signal


def awgn(signal, snr_db):
    """Add white Gaussian noise at the given SNR (dB)."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = np.mean(signal ** 2) / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def msk_demodulate(signal, bit_rate, carrier_freq, sample_rate):
    """Coherent MSK demodulation with phase tracking."""
    bit_duration = 1 / bit_rate
    samples_per_bit = int(sample_rate * bit_duration)
    n_bits = len(signal) // samples_per_bit
    t = np.arange(len(signal)) / sample_rate
    freq_high = carrier_freq + 1 / (2 * bit_duration)
    freq_low  = carrier_freq - 1 / (2 * bit_duration)
    phase = 0.0
    bits = []

    for i in range(n_bits):
        s, e = i * samples_per_bit, (i + 1) * samples_per_bit
        t_bit = t[s:e]
        seg = signal[s:e]

        ref_high = np.cos(2 * np.pi * freq_high * t_bit + phase)
        ref_low  = np.cos(2 * np.pi * freq_low  * t_bit + phase)

        if np.dot(seg, ref_high) >= np.dot(seg, ref_low):
            bits.append(1)
            phase += 2 * np.pi * freq_high * bit_duration
        else:
            bits.append(0)
            phase += 2 * np.pi * freq_low * bit_duration

    return bits


def compute_ber(original, decoded):
    orig = np.array(original)
    dec  = np.array(decoded[:len(orig)])
    return float(np.mean(orig != dec))


def ber_vs_snr(bit_rate, carrier_freq, sample_rate, snr_range_db, n_bits=1000, seed=42):
    """Monte Carlo BER curve; uses the same random bit sequence at every SNR."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, n_bits).tolist()
    _, clean = msk_modulate(data, bit_rate, carrier_freq, sample_rate)
    bers = []
    for snr_db in snr_range_db:
        decoded = msk_demodulate(awgn(clean, snr_db), bit_rate, carrier_freq, sample_rate)
        bers.append(compute_ber(data, decoded))
    return snr_range_db, bers


def iq_samples(signal, carrier_freq, sample_rate, bit_rate):
    """Return one I/Q point per bit (sampled at bit centre after mixing)."""
    bit_duration = 1 / bit_rate
    samples_per_bit = int(sample_rate * bit_duration)
    n_bits = len(signal) // samples_per_bit
    t = np.arange(len(signal)) / sample_rate
    i_vals, q_vals = [], []

    for k in range(n_bits):
        s, e = k * samples_per_bit, (k + 1) * samples_per_bit
        seg = signal[s:e]
        t_bit = t[s:e]
        i_vals.append(np.mean(seg * np.cos(2 * np.pi * carrier_freq * t_bit)))
        q_vals.append(np.mean(seg * np.sin(2 * np.pi * carrier_freq * t_bit)))

    return np.array(i_vals), np.array(q_vals)
