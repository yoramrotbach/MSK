import numpy as np

# ── Convolutional FEC: rate 1/2, constraint length K=7 ───────────────────────
# Generator polynomials (octal 133 / 171), standard NASA/MIL-STD-188-110 code.
_K  = 7
_G0 = 0b1011011   # octal 133
_G1 = 0b1111001   # octal 171


def conv_encode(bits):
    """Rate 1/2, K=7 convolutional encoder. Appends K-1 flush zeros."""
    sr = 0   # 6-bit shift-register state
    out = []
    for bit in list(bits) + [0] * (_K - 1):
        full_sr = (bit << (_K - 1)) | sr
        c0 = bin(full_sr & _G0).count('1') % 2
        c1 = bin(full_sr & _G1).count('1') % 2
        out += [c0, c1]
        sr = full_sr >> 1
    return out


def viterbi_decode(received, n_data_bits):
    """Hard-decision Viterbi decoder for the rate 1/2, K=7 code."""
    n_states = 1 << (_K - 1)   # 64
    INF      = 10 ** 9
    n_steps  = len(received) // 2

    # Precompute (next_state, c0, c1) for every (state, input) pair
    trans = []
    for s in range(n_states):
        row = []
        for b in range(2):
            full_sr = (b << (_K - 1)) | s
            c0 = bin(full_sr & _G0).count('1') % 2
            c1 = bin(full_sr & _G1).count('1') % 2
            row.append((full_sr >> 1, c0, c1))
        trans.append(row)

    pm = [INF] * n_states
    pm[0] = 0
    tb_state = [[-1] * n_states for _ in range(n_steps)]
    tb_bit   = [[0]  * n_states for _ in range(n_steps)]

    for t in range(n_steps):
        r0, r1  = received[2 * t], received[2 * t + 1]
        new_pm  = [INF] * n_states
        for s in range(n_states):
            if pm[s] == INF:
                continue
            for b in range(2):
                ns, c0, c1 = trans[s][b]
                cost = pm[s] + int(r0 != c0) + int(r1 != c1)
                if cost < new_pm[ns]:
                    new_pm[ns]    = cost
                    tb_state[t][ns] = s
                    tb_bit[t][ns]   = b
        pm = new_pm

    state    = int(np.argmin(pm))
    bits_rev = []
    for t in range(n_steps - 1, -1, -1):
        bits_rev.append(tb_bit[t][state])
        state = tb_state[t][state]
    bits_rev.reverse()
    return bits_rev[:n_data_bits]


# ── LFSR scrambler ────────────────────────────────────────────────────────────
# 9-stage Fibonacci LFSR, polynomial x^9 + x^4 + 1.
# Applying scramble twice with the same seed restores the original bits.

def scramble(bits, seed=0x1FF):
    """XOR bit stream with a 9-bit LFSR sequence (self-inverse)."""
    state = seed & 0x1FF
    out   = []
    for bit in bits:
        out.append(bit ^ (state & 1))
        feedback = ((state >> 8) ^ (state >> 3)) & 1
        state    = ((state >> 1) | (feedback << 8)) & 0x1FF
    return out


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
