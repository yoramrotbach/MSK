"""
MSK Modem — Desktop GUI
Run with: python gui.py
"""

import os
import sys
import threading
import subprocess

import numpy as np
import scipy.io.wavfile as wavfile
import customtkinter as ctk
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msk import msk_modulate, msk_demodulate, conv_encode, viterbi_decode, scramble
from standards import STANDARDS, VALID_BIT_RATES

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

STANDARDS_LIST = list(STANDARDS.keys())
BIT_RATES_LIST  = [str(r) for r in VALID_BIT_RATES]
HERE            = os.path.dirname(os.path.abspath(__file__))


# ── helpers ───────────────────────────────────────────────────────────────────

def text_to_bits(text):
    bits = []
    for ch in text:
        b = ord(ch)
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        byte = sum(bits[i + j] << (7 - j) for j in range(8))
        if byte == 0x0A or byte == 0x0D or (0x20 <= byte <= 0x7E):
            chars.append(chr(byte))
    return "".join(chars)

def int_to_bits(n, width=16):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def bits_to_int(bits):
    n = 0
    for b in bits:
        n = (n << 1) | b
    return n


# ── main application ──────────────────────────────────────────────────────────

class MSKApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("MSK Modem  —  MIL-STD-188-110 / STANAG-4285")
        self.geometry("860x680")
        self.minsize(700, 580)

        # Header
        ctk.CTkLabel(self, text="MSK Modem",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(18, 0))
        ctk.CTkLabel(self, text="MIL-STD-188-110  |  STANAG-4285",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(2, 12))

        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        for name in ("Encode", "Decode", "Analysis"):
            self.tabs.add(name)

        self._build_encode_tab()
        self._build_decode_tab()
        self._build_analysis_tab()

    # ── widget helpers ────────────────────────────────────────────────────────

    def _file_row(self, parent, label, row, save=False, filetypes=None):
        ctk.CTkLabel(parent, text=label, width=120, anchor="e").grid(
            row=row, column=0, padx=(12, 6), pady=8, sticky="e")
        var = ctk.StringVar()
        ctk.CTkEntry(parent, textvariable=var, width=420).grid(
            row=row, column=1, padx=6, pady=8, sticky="ew")
        ft = filetypes or [("All files", "*.*")]
        def browse(v=var, s=save, f=ft):
            p = (filedialog.asksaveasfilename(filetypes=f, initialdir=HERE)
                 if s else filedialog.askopenfilename(filetypes=f, initialdir=HERE))
            if p:
                v.set(p)
        ctk.CTkButton(parent, text="Browse", width=80, command=browse).grid(
            row=row, column=2, padx=(6, 12), pady=8)
        parent.columnconfigure(1, weight=1)
        return var

    def _dropdown_row(self, parent, label, row, values, default):
        ctk.CTkLabel(parent, text=label, width=120, anchor="e").grid(
            row=row, column=0, padx=(12, 6), pady=8, sticky="e")
        var = ctk.StringVar(value=default)
        ctk.CTkOptionMenu(parent, variable=var, values=values, width=260).grid(
            row=row, column=1, padx=6, pady=8, sticky="w")
        return var

    def _log_widget(self, parent):
        box = ctk.CTkTextbox(parent, font=ctk.CTkFont(family="Courier New", size=12),
                              wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        box.configure(state="disabled")
        return box

    def _log(self, box, text):
        def _do():
            box.configure(state="normal")
            box.insert("end", text + "\n")
            box.see("end")
            box.configure(state="disabled")
        self.after(0, _do)

    def _clear(self, box):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.configure(state="disabled")

    def _play(self, path_var_or_str):
        path = path_var_or_str if isinstance(path_var_or_str, str) else path_var_or_str.get()
        path = path.strip()
        if not path:
            messagebox.showwarning("No file", "No WAV file selected.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return
        subprocess.Popen(["afplay", path])

    # ── Encode tab ────────────────────────────────────────────────────────────

    def _build_encode_tab(self):
        tab = self.tabs.tab("Encode")

        form = ctk.CTkFrame(tab)
        form.pack(fill="x", padx=4, pady=(10, 4))

        self.enc_input  = self._file_row(form, "Text File", 0,
                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        self.enc_output = self._file_row(form, "Output WAV", 1, save=True,
                            filetypes=[("WAV files", "*.wav")])
        self.enc_std    = self._dropdown_row(form, "Standard",    2, STANDARDS_LIST, STANDARDS_LIST[0])
        self.enc_rate   = self._dropdown_row(form, "Bit Rate (bps)", 3, BIT_RATES_LIST, "300")

        btns = ctk.CTkFrame(tab, fg_color="transparent")
        btns.pack(pady=8)
        ctk.CTkButton(btns, text="▶  Encode to WAV", width=180,
                      command=self._run_encode).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="🔊  Play WAV", width=130,
                      command=lambda: self._play(self.enc_output)).pack(side="left", padx=8)

        ctk.CTkLabel(tab, text="Status", anchor="w",
                     font=ctk.CTkFont(size=12)).pack(fill="x", padx=14, pady=(4, 0))
        self.enc_log = self._log_widget(tab)

    def _run_encode(self):
        inp = self.enc_input.get().strip()
        out = self.enc_output.get().strip()
        if not inp or not out:
            messagebox.showwarning("Missing fields", "Please select both a text file and an output WAV path.")
            return
        if not os.path.exists(inp):
            messagebox.showerror("Not found", f"Input file not found:\n{inp}")
            return
        self._clear(self.enc_log)
        threading.Thread(target=self._encode_worker, args=(inp, out), daemon=True).start()

    def _encode_worker(self, inp, out):
        try:
            std_name = self.enc_std.get()
            bit_rate = int(self.enc_rate.get())
            std      = STANDARDS[std_name]
            carrier  = std["carrier_freq"]
            sr       = std["sample_rate"]

            self._log(self.enc_log, f"Standard : {std_name}")
            self._log(self.enc_log, f"Bit rate : {bit_rate} bps  |  Carrier: {carrier} Hz")

            with open(inp, "r", encoding="utf-8") as f:
                text = f.read()

            data_bits   = text_to_bits(text)
            header_bits = int_to_bits(len(data_bits))
            payload     = header_bits + data_bits

            if std["use_scrambler"]:
                payload = scramble(payload, std["scrambler_seed"])
            if std["use_fec"]:
                payload = conv_encode(payload)

            transmission = std["preamble"] + payload
            _, signal    = msk_modulate(transmission, bit_rate, carrier, sr)
            pcm          = np.int16(signal / np.max(np.abs(signal)) * 32767)
            wavfile.write(out, sr, pcm)

            duration = len(transmission) / bit_rate
            f_high   = carrier + bit_rate // 2
            f_low    = carrier - bit_rate // 2

            self._log(self.enc_log, f"Input    : {len(text)} chars  →  {len(data_bits)} bits")
            self._log(self.enc_log, f"FEC      : {'on  →  ' + str(len(payload)) + ' encoded bits' if std['use_fec'] else 'off'}")
            self._log(self.enc_log, f"Preamble : {len(std['preamble'])} bits")
            self._log(self.enc_log, f"Duration : {duration:.2f} s")
            self._log(self.enc_log, f"Tones    : {f_low} Hz (0)  /  {f_high} Hz (1)")
            self._log(self.enc_log, f"Saved  → {out}")
            self._log(self.enc_log, "✓ Done.")
        except Exception as e:
            self._log(self.enc_log, f"✗ ERROR: {e}")

    # ── Decode tab ────────────────────────────────────────────────────────────

    def _build_decode_tab(self):
        tab = self.tabs.tab("Decode")

        form = ctk.CTkFrame(tab)
        form.pack(fill="x", padx=4, pady=(10, 4))

        self.dec_input  = self._file_row(form, "WAV File", 0,
                            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        self.dec_output = self._file_row(form, "Output Text", 1, save=True,
                            filetypes=[("Text files", "*.txt")])
        self.dec_std    = self._dropdown_row(form, "Standard",    2, STANDARDS_LIST, STANDARDS_LIST[0])
        self.dec_rate   = self._dropdown_row(form, "Bit Rate (bps)", 3, BIT_RATES_LIST, "300")

        btns = ctk.CTkFrame(tab, fg_color="transparent")
        btns.pack(pady=8)
        ctk.CTkButton(btns, text="▶  Decode WAV", width=180,
                      command=self._run_decode).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="🔊  Play WAV", width=130,
                      command=lambda: self._play(self.dec_input)).pack(side="left", padx=8)

        ctk.CTkLabel(tab, text="Decoded Text", anchor="w",
                     font=ctk.CTkFont(size=12)).pack(fill="x", padx=14, pady=(4, 0))
        self.dec_log = self._log_widget(tab)

    def _run_decode(self):
        inp = self.dec_input.get().strip()
        if not inp:
            messagebox.showwarning("Missing field", "Please select a WAV file.")
            return
        if not os.path.exists(inp):
            messagebox.showerror("Not found", f"File not found:\n{inp}")
            return
        self._clear(self.dec_log)
        threading.Thread(target=self._decode_worker, args=(inp,), daemon=True).start()

    def _decode_worker(self, inp):
        try:
            std_name  = self.dec_std.get()
            bit_rate  = int(self.dec_rate.get())
            std       = STANDARDS[std_name]

            self._log(self.dec_log, f"Standard : {std_name}  |  {bit_rate} bps")
            self._log(self.dec_log, "Demodulating …")

            sr, data = wavfile.read(inp)
            if data.dtype == np.int16:
                signal = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                signal = data.astype(np.float64) / 2147483648.0
            else:
                signal = data.astype(np.float64)
            if signal.ndim == 2:
                signal = signal[:, 0]

            all_bits = msk_demodulate(signal, bit_rate, std["carrier_freq"], sr)
            payload  = all_bits[len(std["preamble"]):]

            if std["use_fec"]:
                self._log(self.dec_log, "Viterbi decoding …")
                n_data  = len(payload) // 2 - 6
                payload = viterbi_decode(payload, n_data)
            if std["use_scrambler"]:
                payload = scramble(payload, std["scrambler_seed"])

            if len(payload) < 16:
                self._log(self.dec_log, "✗ Too few bits — check standard and bit rate match the encoder.")
                return

            n_text   = bits_to_int(payload[:16])
            text     = bits_to_text(payload[16: 16 + n_text])

            out = self.dec_output.get().strip()
            if out:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(text)
                self._log(self.dec_log, f"Saved → {out}")

            self._log(self.dec_log, f"─── Decoded ({len(text)} characters) ───")
            self._log(self.dec_log, text)
            self._log(self.dec_log, "✓ Done.")
        except Exception as e:
            self._log(self.dec_log, f"✗ ERROR: {e}")

    # ── Analysis tab ──────────────────────────────────────────────────────────

    def _build_analysis_tab(self):
        tab = self.tabs.tab("Analysis")

        ctk.CTkLabel(tab,
                     text="Open analysis plots in a separate window.",
                     font=ctk.CTkFont(size=13), text_color="gray").pack(pady=(24, 16))

        btns = ctk.CTkFrame(tab, fg_color="transparent")
        btns.pack(pady=4)

        ctk.CTkButton(btns, text="Modulator Demo\nTime domain + Spectrum",
                      width=230, height=70,
                      command=lambda: self._launch_plot("mod.py")).pack(side="left", padx=16)

        ctk.CTkButton(btns, text="Demodulator Demo\nBER curve + I/Q Constellation",
                      width=230, height=70,
                      command=lambda: self._launch_plot("demod.py")).pack(side="left", padx=16)

        ctk.CTkLabel(tab, text="Console Output", anchor="w",
                     font=ctk.CTkFont(size=12)).pack(fill="x", padx=14, pady=(24, 0))
        self.ana_log = self._log_widget(tab)

    def _launch_plot(self, script):
        path = os.path.join(HERE, script)
        self._log(self.ana_log, f"Launching {script} …")
        def worker():
            result = subprocess.run([sys.executable, path],
                                    capture_output=True, text=True, cwd=HERE)
            if result.stderr.strip():
                self._log(self.ana_log, result.stderr.strip())
            self._log(self.ana_log, f"{script} closed.")
        threading.Thread(target=worker, daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MSKApp()
    app.mainloop()
