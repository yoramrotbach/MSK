"""
Standard parameter presets for MSK modem operation.

MIL-STD-188-110
    US military HF data modem standard.
    Carrier 1500 Hz, rate 1/2 K=7 FEC, 9-bit LFSR scrambler.
    Valid bit rates: 75 / 150 / 300 / 600 / 1200 / 2400 bps.

STANAG-4285
    NATO single-tone HF modem standard.
    Carrier 1800 Hz, same FEC and scrambler.
    Valid bit rates: 75 / 150 / 300 / 600 / 1200 / 2400 bps.
"""

# 48-bit alternating preamble for MIL-STD, 80-bit for STANAG.
_PREAMBLE_MIL    = [1, 0] * 24          # 48 bits
_PREAMBLE_STANAG = [1, 0] * 40          # 80 bits

STANDARDS = {
    "MIL-STD-188-110": {
        "carrier_freq":    1500,
        "sample_rate":     44100,
        "valid_bit_rates": [75, 150, 300, 600, 1200, 2400],
        "default_bit_rate": 300,
        "preamble":        _PREAMBLE_MIL,
        "use_scrambler":   True,
        "scrambler_seed":  0x1FF,
        "use_fec":         True,
    },
    "STANAG-4285": {
        "carrier_freq":    1800,
        "sample_rate":     44100,
        "valid_bit_rates": [75, 150, 300, 600, 1200, 2400],
        "default_bit_rate": 300,
        "preamble":        _PREAMBLE_STANAG,
        "use_scrambler":   True,
        "scrambler_seed":  0x1FF,
        "use_fec":         True,
    },
}

DEFAULT_STANDARD = "MIL-STD-188-110"
VALID_BIT_RATES  = [75, 150, 300, 600, 1200, 2400]
