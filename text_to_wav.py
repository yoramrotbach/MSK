"""
Encode a text file as an MSK-modulated WAV audio file.

Encoding chain:
  text → bits → [16-bit length header | data bits]
       → LFSR scramble
       → rate 1/2 K=7 convolutional FEC
       → preamble prepended
       → MSK modulate → WAV

Usage:
    python text_to_wav.py input.txt
    python text_to_wav.py input.txt output.wav
    python text_to_wav.py input.txt output.wav --standard STANAG-4285
    python text_to_wav.py input.txt output.wav --standard MIL-STD-188-110 --bit-rate 600
"""

import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from msk import msk_modulate, conv_encode, scramble
from standards import STANDARDS, DEFAULT_STANDARD, VALID_BIT_RATES


def text_to_bits(text):
    bits = []
    for ch in text:
        b = ord(ch)
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def int_to_bits(n, width=16):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]


def encode(input_path, output_path, standard_name, bit_rate):
    std = STANDARDS[standard_name]

    if bit_rate not in std["valid_bit_rates"]:
        raise ValueError(
            f"{bit_rate} bps is not valid for {standard_name}. "
            f"Choose from: {std['valid_bit_rates']}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text:
        print("Input file is empty.")
        return

    data_bits   = text_to_bits(text)
    header_bits = int_to_bits(len(data_bits), 16)   # 16-bit length field
    payload     = header_bits + data_bits

    if std["use_scrambler"]:
        payload = scramble(payload, std["scrambler_seed"])

    if std["use_fec"]:
        payload = conv_encode(payload)                # doubles bit count + K-1 flush

    transmission = std["preamble"] + payload

    carrier     = std["carrier_freq"]
    sample_rate = std["sample_rate"]
    _, signal   = msk_modulate(transmission, bit_rate, carrier, sample_rate)

    pcm = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(output_path, sample_rate, pcm)

    n_fec = len(payload) if std["use_fec"] else 0
    duration = len(transmission) / bit_rate
    f_high = carrier + bit_rate // 2
    f_low  = carrier - bit_rate // 2

    print(f"Standard : {standard_name}")
    print(f"Input    : {input_path}  ({len(text)} chars, {len(data_bits)} data bits)")
    print(f"FEC      : {'on — ' + str(len(data_bits) + 16) + ' payload bits → ' + str(n_fec) + ' encoded bits' if std['use_fec'] else 'off'}")
    print(f"Scrambler: {'on' if std['use_scrambler'] else 'off'}")
    print(f"Preamble : {len(std['preamble'])} bits")
    print(f"Output   : {output_path}")
    print(f"Duration : {duration:.2f} s  at {bit_rate} bps")
    print(f"Carrier  : {carrier} Hz  |  f_high={f_high} Hz  f_low={f_low} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Encode a text file as a standard-compliant MSK audio signal."
    )
    parser.add_argument("input",  help="Input text file")
    parser.add_argument("output", nargs="?", default="output.wav", help="Output WAV file (default: output.wav)")
    parser.add_argument(
        "--standard", default=DEFAULT_STANDARD,
        choices=list(STANDARDS.keys()),
        help=f"Waveform standard (default: {DEFAULT_STANDARD})"
    )
    parser.add_argument(
        "--bit-rate", type=int, default=None,
        help=f"Bit rate in bps. Valid values: {VALID_BIT_RATES} (default: standard's default)"
    )
    args = parser.parse_args()

    bit_rate = args.bit_rate or STANDARDS[args.standard]["default_bit_rate"]
    encode(args.input, args.output, args.standard, bit_rate)


if __name__ == "__main__":
    main()
