import numpy as np
import matplotlib.pyplot as plt
from msk import msk_modulate

binary_data  = [1, 0, 1, 1, 0, 0, 1]
bit_rate     = 1e3
carrier_freq = 2e3
sample_rate  = 1e5

t, signal = msk_modulate(binary_data, bit_rate, carrier_freq, sample_rate)

freqs    = np.fft.rfftfreq(len(signal), 1 / sample_rate)
spectrum = np.abs(np.fft.rfft(signal))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(t * 1e3, signal, linewidth=0.8)
ax1.set_title("MSK Modulated Signal")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")
ax1.grid(True)

ax2.plot(freqs / 1e3, 20 * np.log10(spectrum + 1e-12))
ax2.set_xlim(0, 6)
ax2.set_title("Frequency Spectrum")
ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.grid(True)

plt.tight_layout()
plt.show()
