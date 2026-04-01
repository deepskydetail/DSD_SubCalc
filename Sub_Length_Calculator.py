# Sub_Length_Calculator.py
# SPDX-License-Identifier: MIT License + Commons Clause
# Author: Deep Sky Detail (Python/Siril port)
#
# Sub-Exposure Length Calculator for Siril
# Requires Siril >= 1.4.0
#
# INSTALLATION:
#   Place this file in your Siril scripts directory, then run it from
#   Scripts menu, or via the Siril command: pyscript Sub_Length_Calculator
#
# USAGE:
#   1. Run the script — a GUI window will open.
#   2. Pick your Bias (or Master Bias/Dark) FITS file.
#   3. Pick your Light FITS file.
#   4. Adjust parameters and click Analyse.

"""
Sub-Exposure Length Calculator.
Estimates the minimum sub-exposure length required to swamp read noise,
using an ex-Gaussian MLE fit to the background pixel distribution.
Ported from the R Shiny app by Deep Sky Detail.
"""

# ── Core imports ────────────────────────────────────────────────────────────
import os
import sys
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# ── Siril module ─────────────────────────────────────────────────────────────
import sirilpy as s
from sirilpy import tksiril, SirilError

# ── Optional deps managed by Siril's venv ────────────────────────────────────
s.ensure_installed("ttkthemes")
s.ensure_installed("matplotlib")

from ttkthemes import ThemedTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

VERSION = "1.0.1"

# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy ex-Gaussian MLE  (no scipy)
# Mirrors R retimes::timefit(x, iter=0)
# ─────────────────────────────────────────────────────────────────────────────

from math import (erfc as _erfc, sqrt as _sqrt, exp as _exp,
                  log as _log, isnan as _isnan, isinf as _isinf)


def _norm_cdf(x: float) -> float:
    return 0.5 * _erfc(-x / _sqrt(2.0))


def _mexgauss(arr: np.ndarray):
    """Method-of-moments starting values — port of R mexgauss()."""
    n   = len(arr)
    k1  = float(np.mean(arr))
    dev = arr - k1
    k2  = float(np.sum(dev ** 2) / (n - 1))
    k3  = float(np.sum(dev ** 3) / (n - 1))

    if k3 > 0:
        tau = (k3 / 2.0) ** (1.0 / 3.0)
    else:
        tau = 0.8 * float(np.std(arr, ddof=1))

    sigma = float(np.sqrt(abs(k2 - tau ** 2)))
    mu    = k1 - tau
    return mu, sigma, tau


def _negloglik(par, x_arr: np.ndarray) -> float:
    """Negative log-likelihood for ex-Gaussian (unconstrained parameterisation)."""
    mu = par[0]
    try:
        sigma = _exp(par[1])
        tau   = _exp(par[2])
    except OverflowError:
        return 1e50

    inv_t = 1.0 / tau
    s2_t  = sigma * sigma / tau
    const = mu * inv_t + sigma * sigma / (2.0 * tau * tau)

    total = 0.0
    for xi in x_arr:
        lp = const - xi * inv_t
        if lp < -700:
            return 1e50
        cdf_val = _norm_cdf((xi - mu - s2_t) / sigma)
        if cdf_val <= 0.0:
            return 1e50
        d = inv_t * _exp(lp) * cdf_val
        if d <= 0.0:
            return 1e50
        total += _log(d)

    val = -total
    return 1e50 if (_isnan(val) or _isinf(val)) else val


def _nelder_mead(func, x0, args=(), max_iter=2000, xatol=1e-6, fatol=1e-6):
    """Pure-Python Nelder-Mead, matching R optim() behaviour."""
    n    = len(x0)
    pts  = [np.array(x0, dtype=float)]
    for i in range(n):
        p     = np.array(x0, dtype=float)
        p[i] += 0.05 if abs(p[i]) > 1e-10 else 0.00025
        pts.append(p)
    fpts = [func(p, *args) for p in pts]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fpts[i])
        pts   = [pts[i]  for i in order]
        fpts  = [fpts[i] for i in order]

        if (max(abs(fpts[i] - fpts[0]) for i in range(1, n + 1)) < fatol
                and np.max(np.abs(np.array(pts[1:]) - pts[0])) < xatol):
            break

        centroid = np.mean(pts[:-1], axis=0)
        worst    = pts[-1]
        fw       = fpts[-1]

        r  = centroid + (centroid - worst)
        fr = func(r, *args)

        if fr < fpts[0]:
            e  = centroid + 2.0 * (r - centroid)
            fe = func(e, *args)
            pts[-1], fpts[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < fpts[-2]:
            pts[-1], fpts[-1] = r, fr
        else:
            c  = centroid + 0.5 * (worst - centroid)
            fc = func(c, *args)
            if fc < fw:
                pts[-1], fpts[-1] = c, fc
            else:
                best = pts[0]
                pts  = [best] + [best + 0.5 * (p - best) for p in pts[1:]]
                fpts = [func(p, *args) for p in pts]

    return pts[0]


def fit_exgaussian(data: np.ndarray):
    """Full MLE ex-Gaussian fit matching R retimes::timefit(x, iter=0)."""
    try:
        arr            = np.asarray(data, dtype=np.float64)
        mu0, s0, tau0  = _mexgauss(arr)
        s0    = max(s0,   1e-8)
        tau0  = max(tau0, 1e-8)
        start = np.array([mu0, math.log(s0), math.log(tau0)], dtype=float)
        res   = _nelder_mead(_negloglik, start, args=(arr,))
        return float(res[0]), float(_exp(res[1])), float(_exp(res[2]))
    except Exception:
        m, sd = float(np.mean(data)), float(np.std(data, ddof=1))
        return m, sd, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FITS reader
# ─────────────────────────────────────────────────────────────────────────────

def read_fits_mono(filepath: str) -> np.ndarray:
    """Read a FITS file and return a 2-D float64 array."""
    s.ensure_installed("astropy")
    from astropy.io import fits as afits
    with afits.open(filepath, ignore_missing_simple=True) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                break
    if data is None:
        raise ValueError(f"No image data found in {filepath}")
    data = data.astype(np.float64)
    if data.ndim == 3:
        data = data.mean(axis=0)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(bias_path, light_path, bit_depth=16,
                 duration=60.0, gain=1.0, swamp=10.0):
    # (logging handled by caller)

    bias_data  = read_fits_mono(bias_path)
    light_data = read_fits_mono(light_path)

    rng = np.random.default_rng(123)

    def sample(arr):
        flat = arr.flatten()
        return rng.choice(flat, size=min(10_000, len(flat)), replace=False)

    bias_s  = sample(bias_data)
    light_s = sample(light_data)

    bd_norm    = 2 ** (16 - int(bit_depth))
    bias_norm  = bias_s  / bd_norm
    light_norm = light_s / bd_norm

    rel_rn = float(np.std(bias_norm, ddof=1))
    bias_m = float(np.median(bias_norm))

    med = float(np.median(light_norm))
    mad = 1.4826 * float(np.median(np.abs(light_norm - med)))
    light_bg = light_norm[np.abs(light_norm - med) < 3 * mad]

    pass  # fitting
    light_m, sigma_eg, tau = fit_exgaussian(light_bg)

    total_noise  = float(np.sqrt(sigma_eg ** 2 + tau ** 2))
    rel_sn       = float(np.sqrt(max(sigma_eg ** 2 - rel_rn ** 2, 0.0)))
    rel_sn2      = np.sqrt((light_m - bias_m)* gain) / gain #float(np.sqrt(max(light_m - bias_m, 0.0)))

    adu_added    = light_m - bias_m
    e_added      = adu_added * gain
    adu_s        = (rel_sn ** 2) / duration if duration else float("nan")
    adu_s2       = adu_added / duration     if duration else float("nan")

    #desired      = swamp * rel_rn ** 2
    #min_sub      = desired / adu_s  if adu_s  else float("nan")
    #min_sub2     = desired / adu_s2 if adu_s2 else float("nan")
    

    true_rn  = rel_rn  * gain
    true_sn  = rel_sn  * gain
    true_sn2 = rel_sn2 * gain
    true_tot = total_noise * gain
    e_s      = adu_s2 * gain
    
    #1.0.1
    desired_e = swamp * true_rn ** 2  # corresponds to DesiredSignal_e in R
    min_sub  = desired_e / (true_sn ** 2 / duration) if duration else float("nan")
    min_sub2 = desired_e / (true_sn2 ** 2 / duration) if duration else float("nan")
    #end 1.0.1
    
    swamp_est  = (rel_sn ** 2) / (rel_rn ** 2) if rel_rn else float("nan")
    ideal_cam  = float("nan")
    ideal_cam2 = float("nan")
    if rel_sn > 0:
        ideal_cam  = (math.sqrt(0.001 + rel_sn ** 2) /
                      math.sqrt(0.001 + rel_sn ** 2 + rel_rn ** 2))
        ideal_cam2 = (1.0 / ideal_cam) ** 2

    seg_x1     = bias_m + 3 * rel_rn
    seg_x2     = float(np.quantile(light_bg, 0.01))
    multfactor = round((seg_x2 - seg_x1) / rel_rn, 2) if rel_rn else 0.0

    pass  # results logged by caller

    return dict(
        light_bg=light_bg, bias_norm=bias_norm,
        light_m=light_m, bias_m=bias_m, sigma_eg=sigma_eg,
        rel_rn=rel_rn, rel_sn=rel_sn, rel_sn2=rel_sn2,
        true_rn=true_rn, true_sn=true_sn, true_sn2=true_sn2, true_tot=true_tot,
        total_noise=total_noise, adu_added=adu_added, e_added=e_added,
        adu_s=adu_s2, e_s=e_s, min_sub=min_sub, min_sub2=min_sub2,
        swamp_est=swamp_est, ideal_cam=ideal_cam, ideal_cam2=ideal_cam2,
        seg_x1=seg_x1, seg_x2=seg_x2, multfactor=multfactor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class SubLengthApp:

    # ── Colours (match Siril dark theme) ────────────────────────────────────
    BG      = "#2b2b2b"
    FG      = "#e0e0e0"
    ENTRY   = "#1e1e1e"
    ACCENT  = "#1B9E77"
    RED     = "#c0392b"
    BTN_FG  = "#ffffff"

    def __init__(self, root: ThemedTk):
        self.root   = root
        self.result = None
        self.style  = tksiril.standard_style()

        # Connect to Siril
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            messagebox.showerror("Connection Error", "Failed to connect to Siril.")
            return

        root.title(f"Sub Length Calculator  v{VERSION}  ·  Deep Sky Detail")
        root.resizable(True, True)
        root.configure(bg=self.BG)

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        tksiril.match_theme_to_siril(self.root, self.siril)

        # ── Top: file pickers ────────────────────────────────────────────────
        pf = ttk.LabelFrame(root, text="FITS Files", padding=8)
        pf.pack(fill="x", padx=12, pady=(10, 4))

        self.bias_var  = tk.StringVar()
        self.light_var = tk.StringVar()

        self._file_row(pf, "Bias / Dark FITS:",  self.bias_var,  0)
        self._file_row(pf, "Light FITS:",         self.light_var, 1)

        # ── Middle: parameters ───────────────────────────────────────────────
        pf2 = ttk.LabelFrame(root, text="Parameters", padding=8)
        pf2.pack(fill="x", padx=12, pady=4)

        self.bit_var   = tk.IntVar(value=16)
        self.swamp_var = tk.IntVar(value=10)
        self.dur_var   = tk.DoubleVar(value=60.0)
        self.gain_var  = tk.DoubleVar(value=1.0)

        self._param_row(pf2, "Bit Depth:",              self.bit_var,   0,
                        is_scale=True, from_=8, to=16, resolution=2)
        self._param_row(pf2, "Desired Swamp Factor:",   self.swamp_var, 1,
                        is_scale=True, from_=2, to=10, resolution=1)
        self._param_row(pf2, "Exposure Duration (sec):", self.dur_var,  2)
        self._param_row(pf2, "Gain (e⁻/ADU):",           self.gain_var, 3)

        # ── Analyse button ───────────────────────────────────────────────────
        bf = ttk.Frame(root)
        bf.pack(fill="x", padx=12, pady=6)
        self.analyse_btn = ttk.Button(bf, text="▶  Analyse",
                                      command=self._on_analyse)
        self.analyse_btn.pack(side="left", ipadx=16, ipady=4)
        self.status_lbl = ttk.Label(bf, text="")
        self.status_lbl.pack(side="left", padx=12)

        # ── Notebook: histogram | tables ─────────────────────────────────────
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        self.hist_frame  = ttk.Frame(nb)
        self.table_frame = ttk.Frame(nb)
        nb.add(self.hist_frame,  text="Histogram")
        nb.add(self.table_frame, text="Results")

        self._build_histogram_placeholder()
        self._build_table_area()

    def _file_row(self, parent, label, var, row):
        ttk.Label(parent, text=label, width=18, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        e = ttk.Entry(parent, textvariable=var, width=55)
        e.grid(row=row, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(parent, text="Browse…",
                   command=lambda v=var: self._browse(v)).grid(
            row=row, column=2, sticky="e")
        parent.columnconfigure(1, weight=1)

    def _param_row(self, parent, label, var, row,
                   is_scale=False, from_=None, to=None, resolution=None):
        ttk.Label(parent, text=label, width=26, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        if is_scale:
            val_lbl = ttk.Label(parent, width=4, anchor="w")
            val_lbl.grid(row=row, column=2, sticky="w", padx=(6, 0))

            def update_lbl(v, lbl=val_lbl, vr=var):
                lbl.config(text=str(int(float(v))))
            sl = ttk.Scale(parent, variable=var, from_=from_, to=to,
                           orient="horizontal",
                           command=update_lbl)
            sl.grid(row=row, column=1, sticky="ew", padx=(0, 4))
            val_lbl.config(text=str(var.get()))
        else:
            ttk.Entry(parent, textvariable=var, width=10).grid(
                row=row, column=1, sticky="w")
        parent.columnconfigure(1, weight=1)

    def _build_histogram_placeholder(self):
        self.fig    = Figure(figsize=(7, 3.5), dpi=96, facecolor="#2b2b2b")
        self.ax     = self.fig.add_subplot(111)
        self._style_ax(self.ax)
        self.ax.text(0.5, 0.5, "Load files and click Analyse",
                     ha="center", va="center",
                     transform=self.ax.transAxes,
                     color="#888", fontsize=11)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _build_table_area(self):
        tf = self.table_frame

        sections = [
            ("ADU Analysis",    self._adu_rows),
            ("e⁻ Analysis",     self._e_rows),
            ("Sub Optimisation",self._opt_rows),
        ]
        self.section_widgets = {}
        for title, _ in sections:
            lf = ttk.LabelFrame(tf, text=title, padding=6)
            lf.pack(fill="x", padx=8, pady=4)
            self.section_widgets[title] = lf

        self._build_definitions_box(tf)

    def _build_definitions_box(self, parent):
        lf = ttk.LabelFrame(parent, text="Definitions", padding=6)
        lf.pack(fill="x", padx=8, pady=4)
        defs = (
            ("Read.Noise",       "Estimated read noise (ADU or e⁻) from the bias std-dev."),
            ("ADU/e Added",      "Sky-glow increase in pixel values during the sub."),
            ("SkyNoise E1",      "Sky-glow noise estimated from the variance. "
                                 "Independent of Gain and Bit-Depth."),
            ("SkyNoise E2",      "Sky-glow noise from signal amplitude. "
                                 "Depends on Gain and Bit-Depth."),
            ("Total Noise",      "Total noise including DSO and stars."),
            ("ADU/e Per Sec",    "Sky-glow rate per second."),
            ("Min Sub E1 (s)",   "Minimum sub length to swamp read noise (variance method, "
                                 "independent of Gain/Bit-Depth)."),
            ("Min Sub E2 (s)",   "Minimum sub length to swamp read noise (signal method, "
                                 "depends on Gain/Bit-Depth)."),
            ("Measured Swamp",   "Ratio of sky-glow variance to read-noise variance "
                                 "in the uploaded frame."),
            ("Sub Efficiency",   "Your SNR as % of a perfect zero-read-noise camera. "
                                 "≥95 % is excellent."),
            ("Extra Integration","Extra integration time vs a perfect camera. "
                                 "E.g. 1.5× means you need 50 % more time."),
        )
        for term, defn in defs:
            row_f = ttk.Frame(lf)
            row_f.pack(fill="x", pady=1)
            ttk.Label(row_f, text=term, width=18, anchor="nw",
                      font=("", 9, "bold")).pack(side="left")
            ttk.Label(row_f, text=defn, anchor="w",
                      wraplength=520, justify="left",
                      font=("", 9)).pack(side="left", fill="x", expand=True)

    # ── Event handlers ───────────────────────────────────────────────────────

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Select FITS file",
            filetypes=[("FITS files", "*.fit *.fits *.FIT *.FITS"),
                       ("All files",  "*.*")],
        )
        if path:
            var.set(path)

    def _on_analyse(self):
        bias_path  = self.bias_var.get().strip()
        light_path = self.light_var.get().strip()

        if not bias_path or not os.path.isfile(bias_path):
            messagebox.showerror("Error", "Please select a valid Bias FITS file.")
            return
        if not light_path or not os.path.isfile(light_path):
            messagebox.showerror("Error", "Please select a valid Light FITS file.")
            return

        self.analyse_btn.state(["disabled"])
        self.status_lbl.config(text="Analysing…")
        self.root.update_idletasks()

        try:
            self.siril.log("Sub Length Calculator: starting analysis...")
            self.result = run_analysis(
                bias_path, light_path,
                bit_depth = int(self.bit_var.get()),
                duration  = float(self.dur_var.get()),
                gain      = float(self.gain_var.get()),
                swamp     = float(self.swamp_var.get()),
            )
            a = self.result
            self.siril.log(
                f"Sub Length Calculator: min sub ≈ {a['min_sub']:.1f}s (E1) / "
                f"{a['min_sub2']:.1f}s (E2)  |  swamp = {a['swamp_est']:.1f}×"
            )
            self._update_histogram(self.result)
            self._update_tables(self.result)
            self.status_lbl.config(text="✓ Done")
        except Exception as exc:
            messagebox.showerror("Analysis Error", str(exc))
            self.status_lbl.config(text="Error")
        finally:
            self.analyse_btn.state(["!disabled"])

    # ── Result rendering ─────────────────────────────────────────────────────

    def _update_histogram(self, a):
        self.ax.clear()
        self._style_ax(self.ax)

        self.ax.hist(a["light_bg"],   bins=100, density=True,
                     color="gold",      alpha=0.6, label="Light")
        self.ax.hist(a["bias_norm"],  bins=100, density=True,
                     color="tomato",    alpha=0.6, label="Bias")

        self.ax.axvline(a["light_m"], color="white", lw=1.2)

        mu, sigma = a["light_m"], a["sigma_eg"]
        self.ax.hlines(0, mu - sigma, mu + sigma,
                       colors="limegreen", lw=2.5, label="±1σ")
        self.ax.hlines(0.1, a["seg_x1"], a["seg_x2"],
                       colors="orange", lw=2, label="RN→sky gap")
        self.ax.annotate("", xy=(a["seg_x2"], 0.1),
                         xytext=(a["seg_x2"] - 0.5, 0.12),
                         arrowprops=dict(arrowstyle="->",
                                         color="orange", lw=1.5))

        bias_mean = float(np.mean(a["bias_norm"]))
        bias_std  = float(np.std(a["bias_norm"], ddof=1))
        x_min     = bias_mean - 3 * bias_std
        x_max     = float(np.quantile(a["light_bg"], 0.95))
        self.ax.set_xlim(x_min, x_max)

        self.ax.set_xlabel(
            f"Skyglow is ≈ {a['multfactor']}× RN-σ from read noise",
            color=self.FG, fontsize=9)
        self.ax.legend(fontsize=8, facecolor="#3a3a3a",
                       labelcolor=self.FG, framealpha=0.7)
        self.canvas.draw()

    def _style_ax(self, ax):
        ax.set_facecolor("#404040")
        ax.tick_params(colors=self.FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#666")
        ax.yaxis.label.set_color(self.FG)
        ax.xaxis.label.set_color(self.FG)

    def _update_tables(self, a):
        def fmt(v, d=4):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.{d}f}"

        # Clear old content
        for lf in self.section_widgets.values():
            for w in lf.winfo_children():
                w.destroy()

        self._fill_table(
            self.section_widgets["ADU Analysis"],
            headers=["Read Noise", "ADU Added", "SkyNoise E1", "SkyNoise E2",
                     "Total Noise", "ADU/Sec", "Min Sub E1 (s)", "Min Sub E2 (s)"],
            values=[fmt(a["rel_rn"]), fmt(a["adu_added"]),
                    fmt(a["rel_sn"]),  fmt(a["rel_sn2"]),
                    fmt(a["total_noise"]), fmt(a["adu_s"]),
                    fmt(a["min_sub"], 1),  fmt(a["min_sub2"], 1)],
        )
        self._fill_table(
            self.section_widgets["e⁻ Analysis"],
            headers=["Read Noise", "e Added", "SkyNoise E1", "SkyNoise E2",
                     "Total Noise", "e/Sec", "Min Sub E1 (s)", "Min Sub E2 (s)"],
            values=[fmt(a["true_rn"]), fmt(a["e_added"]),
                    fmt(a["true_sn"]),  fmt(a["true_sn2"]),
                    fmt(a["true_tot"]), fmt(a["e_s"]),
                    fmt(a["min_sub"], 1),  fmt(a["min_sub2"], 1)],
        )

        ic  = a["ideal_cam"]
        ic2 = a["ideal_cam2"]
        self._fill_table(
            self.section_widgets["Sub Optimisation"],
            headers=["Measured Swamp", "Sub Efficiency", "Extra Integration"],
            values=[fmt(a["swamp_est"], 2),
                    f"{round(ic  * 100, 2)} %" if not math.isnan(ic)  else "—",
                    f"{round(ic2,      2)} ×"  if not math.isnan(ic2) else "—"],
        )

    def _fill_table(self, parent, headers, values):
        style = {"relief": "flat", "padx": 6, "pady": 4}
        for col, (h, v) in enumerate(zip(headers, values)):
            tk.Label(parent, text=h,  bg="#3a3a3a", fg="#aaa",
                     font=("", 9, "bold"), **style).grid(
                row=0, column=col, sticky="ew", padx=1, pady=1)
            tk.Label(parent, text=v,  bg=self.BG,    fg=self.FG,
                     font=("Courier", 9), **style).grid(
                row=1, column=col, sticky="ew", padx=1, pady=1)
            parent.columnconfigure(col, weight=1)

    # ── ADU / e row helpers (unused directly but kept for clarity) ───────────
    def _adu_rows(self): pass
    def _e_rows(self):   pass
    def _opt_rows(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = ThemedTk(theme="equilux")   # dark Tk theme
    app  = SubLengthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
