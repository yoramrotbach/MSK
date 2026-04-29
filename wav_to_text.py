"""
Decode a standard-compliant MSK WAV file back into text.

Decoding chain:
  WAV → MSK demodulate → strip preamble
      → Viterbi FEC decode
      → LFSR descramble
      → extract 16-bit length header
      → bits → text

Usage:
    python wav_to_text.py output.wav
    python wav_to_text.py output.wav recovered.txt
    python wav_to_text.py output.wav recovered.txt --standard STANAG-4285
    python wav_to_text.py output.wav recovered.txt --standard MIL-STD-188-110 --bit-rate 600
"""

import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from msk import msk_demodulate, viterbi_decode, scramble
from standards import STANDARDS, DEFAULT_STANDARD, VALID_BIT_RATES

_VITERBI_K = 7   # must match conv_encode


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        if byte == 0x0A or byte == 0x0D or (0x20 <= byte <= 0x7E):
            chars.append(chr(byte))
    return "".join(chars)


def bits_to_int(bits):
    n = 0
    for b in bits:
        n = (n << 1) | b
    return n


def decode(input_path, output_path, standard_name, bit_rate):
    std = STANDARDS[standard_name]

    if bit_rate not in std["valid_bit_rates"]:
        raise ValueError(
            f"{bit_rate} bps is not valid for {standard_name}. "
            f"Choose from: {std['valid_bit_rates']}"
        )

    sample_rate, data = wavfile.read(input_path)

    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    else:
        signal = data.astype(np.float64)

    if signal.ndim == 2:
        signal = signal[:, 0]

    all_bits = msk_demodulate(signal, bit_rate, std["carrier_freq"], sample_rate)

    # Strip preamble
    n_preamble = len(std["preamble"])
    payload    = all_bits[n_preamble:]

    # Viterbi FEC decode
    if std["use_fec"]:
        # Each data bit produced 2 encoded bits + K-1 flush pairs
        n_data_bits = len(payload) // 2 - (_VITERBI_K - 1)
        payload = viterbi_decode(payload, n_data_bits)

    # Descramble (self-inverse)
    if std["use_scrambler"]:
        payload = scramble(payload, std["scrambler_seed"])

    # Extract 16-bit length header
    if len(payload) < 16:
        print("Error: too few bits to decode — check that --standard and --bit-rate match the encoder.")
        return ""

    n_text_bits = bits_to_int(payload[:16])
    text_bits   = payload[16: 16 + n_text_bits]
    text        = bits_to_text(text_bits)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Standard : {standard_name}")
        print(f"Decoded  : {len(payload)} bits → {len(text)} characters → {output_path}")
    else:
        print(text)

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Decode a standard-compliant MSK WAV file back into text."
    )
    parser.add_argument("input",  help="Input WAV file produced by text_to_wav.py")
    parser.add_argument("output", nargs="?", default=None, help="Output text file (default: print to terminal)")
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
    decode(args.input, args.output, args.standard, bit_rate)


if __name__ == "__main__":
    main()
