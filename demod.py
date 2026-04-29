import numpy as np
import matplotlib.pyplot as plt
from msk import msk_modulate, msk_demodulate, awgn, ber_vs_snr, iq_samples

binary_data  = [1, 0, 1, 1, 0, 0, 1]
bit_rate     = 1e3
carrier_freq = 2e3
sample_rate  = 1e5
snr_demo_db  = 10

t, clean_signal = msk_modulate(binary_data, bit_rate, carrier_freq, sample_rate)
noisy_signal    = awgn(clean_signal, snr_demo_db)
decoded_clean   = msk_demodulate(clean_signal,  bit_rate, carrier_freq, sample_rate)
decoded_noisy   = msk_demodulate(noisy_signal,  bit_rate, carrier_freq, sample_rate)

print("Original:          ", binary_data)
print(f"Decoded (no noise): {decoded_clean}")
print(f"Decoded (SNR={snr_demo_db} dB): {decoded_noisy}")

# BER vs SNR
snr_range = np.arange(0, 16, 1)
snrs, bers = ber_vs_snr(bit_rate, carrier_freq, sample_rate, snr_range)

# I/Q constellation at two SNR levels
_, sig_clean = msk_modulate(binary_data, bit_rate, carrier_freq, sample_rate)
sig_noisy_low  = awgn(sig_clean, 5)
sig_noisy_high = awgn(sig_clean, 15)

I_c, Q_c = iq_samples(sig_clean,     carrier_freq, sample_rate, bit_rate)
I_l, Q_l = iq_samples(sig_noisy_low,  carrier_freq, sample_rate, bit_rate)
I_h, Q_h = iq_samples(sig_noisy_high, carrier_freq, sample_rate, bit_rate)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Time domain: clean vs noisy
ax = axes[0, 0]
ax.plot(t * 1e3, clean_signal, label="Clean",  linewidth=0.8)
ax.plot(t * 1e3, noisy_signal, label=f"Noisy (SNR={snr_demo_db} dB)", alpha=0.7, linewidth=0.8)
ax.set_title("Time Domain")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

# BER vs SNR
ax = axes[0, 1]
ax.semilogy(snrs, np.maximum(bers, 1e-4), marker='o', label="Simulated BER")
ax.set_title("BER vs SNR")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Bit Error Rate")
ax.grid(True, which='both')
ax.legend()

# I/Q constellation — noisy (SNR 5 dB)
ax = axes[1, 0]
ax.scatter(I_l, Q_l, s=60, alpha=0.8, label="SNR = 5 dB")
ax.scatter(I_c, Q_c, s=80, marker='x', color='red', label="Clean")
ax.set_title("I/Q Constellation (SNR = 5 dB)")
ax.set_xlabel("In-phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_aspect('equal')
ax.legend()
ax.grid(True)

# I/Q constellation — noisy (SNR 15 dB)
ax = axes[1, 1]
ax.scatter(I_h, Q_h, s=60, alpha=0.8, label="SNR = 15 dB")
ax.scatter(I_c, Q_c, s=80, marker='x', color='red', label="Clean")
ax.set_title("I/Q Constellation (SNR = 15 dB)")
ax.set_xlabel("In-phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_aspect('equal')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
