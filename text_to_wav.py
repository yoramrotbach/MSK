"""
Encode a text file as an MSK-modulated WAV audio file.

Usage:
    python text_to_wav.py input.txt output.wav
    python text_to_wav.py input.txt              # writes output.wav by default

The resulting audio file can be played on any media player.
You will hear two alternating tones (like a dial-up modem) encoding your text.

Parameters are chosen so the tones sit comfortably in the audible range:
  carrier : 1500 Hz  (centre tone)
  bit rate: 300 bps  (slow enough to hear the frequency changes)
  f_high  : 1650 Hz  (bit = 1)
  f_low   : 1350 Hz  (bit = 0)
"""

import sys
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from msk import msk_modulate

SAMPLE_RATE  = 44100   # Hz  — standard audio quality
CARRIER_FREQ = 1500    # Hz
BIT_RATE     = 300     # bps — slow enough to hear individual tone shifts


def text_to_bits(text):
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(7, -1, -1):       # MSB first
            bits.append((byte >> i) & 1)
    return bits


def encode(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text:
        print("Input file is empty.")
        return

    bits = text_to_bits(text)
    _, signal = msk_modulate(bits, BIT_RATE, CARRIER_FREQ, SAMPLE_RATE)

    # Normalise to full 16-bit PCM range
    pcm = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(output_path, SAMPLE_RATE, pcm)

    duration = len(bits) / BIT_RATE
    print(f"Input    : {input_path}  ({len(text)} characters, {len(bits)} bits)")
    print(f"Output   : {output_path}")
    print(f"Duration : {duration:.2f} s  at {BIT_RATE} bps")
    print(f"Carrier  : {CARRIER_FREQ} Hz   |  f_high={CARRIER_FREQ + BIT_RATE//2} Hz  f_low={CARRIER_FREQ - BIT_RATE//2} Hz")


def main():
    parser = argparse.ArgumentParser(description="Encode a text file as an MSK audio signal.")
    parser.add_argument("input",  help="Path to the input text file")
    parser.add_argument("output", nargs="?", default="output.wav", help="Path for the output WAV file (default: output.wav)")
    args = parser.parse_args()
    encode(args.input, args.output)


if __name__ == "__main__":
    main()
