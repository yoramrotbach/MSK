"""
Decode an MSK-modulated WAV file back into text.

Usage:
    python wav_to_text.py output.wav recovered.txt
    python wav_to_text.py output.wav          # prints decoded text to the terminal

The WAV file must have been produced by text_to_wav.py using the same parameters.
"""

import sys
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from msk import msk_demodulate

CARRIER_FREQ = 1500   # Hz  — must match text_to_wav.py
BIT_RATE     = 300    # bps — must match text_to_wav.py


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        # Accept printable ASCII and common whitespace
        if byte == 0x0A or byte == 0x0D or (0x20 <= byte <= 0x7E):
            chars.append(chr(byte))
    return "".join(chars)


def decode(input_path, output_path=None):
    sample_rate, data = wavfile.read(input_path)

    # Normalise to float [-1, 1]
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    else:
        signal = data.astype(np.float64)

    # Handle stereo by taking the left channel
    if signal.ndim == 2:
        signal = signal[:, 0]

    bits = msk_demodulate(signal, BIT_RATE, CARRIER_FREQ, sample_rate)
    text = bits_to_text(bits)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Decoded {len(bits)} bits → {len(text)} characters → {output_path}")
    else:
        print(text)

    return text


def main():
    parser = argparse.ArgumentParser(description="Decode an MSK WAV file back into text.")
    parser.add_argument("input",  help="Path to the WAV file produced by text_to_wav.py")
    parser.add_argument("output", nargs="?", default=None, help="Path for the recovered text file (default: print to terminal)")
    args = parser.parse_args()
    decode(args.input, args.output)


if __name__ == "__main__":
    main()
