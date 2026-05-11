"""pulse_plot.py – PulsePlotPanel: single pulse or PRN sequence."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from utils.dsp_utils import safe_eval, fmt, FREQ_SUFFIX
from utils.gui_utils  import (FREQ_UNITS, AMP_UNIT, AMP_COLOR,
                               TEXT, N_SAMPLES, BTN_STYLE)
from gui.widgets      import SliderRow
from classes.base_plot import BasePlotPanel

DELAY_COLOR     = "#fb923c"   # orange
WIDTH_COLOR     = "#a3e635"   # lime
CHIP_RATE_COLOR = "#c084fc"   # purple

# Fixed view window: show 1000 samples worth of signal
VIEW_SAMPLES = N_SAMPLES
# PRN sequence length (maximal-length, 127 chips via 7-bit LFSR)
PRN_LENGTH   = 127


def _single_pulse(n_samples: int, amplitude: float,
                  width_frac: float, delay_frac: float) -> np.ndarray:
    """
    Single rectangular pulse over n_samples.
    width_frac : pulse width as fraction of total window [0–1]
    delay_frac : delay (start position) as fraction of total window [0–1]
    """
    y     = np.zeros(n_samples)
    start = int(delay_frac * n_samples)
    end   = int((delay_frac + width_frac) * n_samples)
    end   = min(end, n_samples)
    if start < end:
        y[start:end] = amplitude
    return y


def _lfsr_prn(length: int = PRN_LENGTH) -> np.ndarray:
    """Generate a maximal-length binary PRN sequence (+1/−1) via 7-bit LFSR."""
    state = [1, 0, 0, 0, 0, 0, 0]   # initial state (non-zero)
    seq   = []
    for _ in range(length):
        bit = state[-1]
        seq.append(1 if bit == 0 else -1)
        feedback = state[6] ^ state[5]   # taps: x^7 + x^6 + 1
        state    = [feedback] + state[:-1]
    return np.array(seq, dtype=float)


_PRN_BASE = _lfsr_prn(PRN_LENGTH)   # compute once


def _prn_signal(n_samples: int, amplitude: float,
                chip_rate_sps: float, delay_samples: int) -> np.ndarray:
    """
    PRN signal: tile the 127-chip sequence at chip_rate_sps chips per sample-window.
    chip_rate_sps : number of samples per chip
    delay_samples : integer sample delay
    """
    if chip_rate_sps < 1:
        chip_rate_sps = 1.0
    samples_per_chip = max(1, int(round(n_samples / chip_rate_sps)))
    # build one full tile
    tile = np.repeat(_PRN_BASE, samples_per_chip)
    # repeat to cover n_samples + delay
    needed = n_samples + delay_samples + len(tile)
    reps   = int(np.ceil(needed / len(tile))) + 1
    full   = np.tile(tile, reps)
    y      = full[delay_samples: delay_samples + n_samples]
    return amplitude * y[:n_samples]


class PulsePlotPanel(BasePlotPanel):
    """
    Signal panel with two modes toggled by a button:

    Pulse mode:
      - Amplitude slider
      - Width slider  (% of view window)
      - Delay slider  (% of view window)

    PRN mode:
      - Amplitude slider
      - Chip Rate slider  (chips displayed across the window, 1–127)
      - Delay slider  (samples)
    """

    LINE_COLOR = "#fb923c"

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(sample_rate_hz, parent)
        self._mode  = "pulse"   # "pulse" | "prn"
        self._guard = False

        # ADC sample overlay
        self._sample_dots  = self._pw.plot(pen=None, symbol="o", symbolSize=6,
                                           symbolBrush=pg.mkBrush("#ef4444"),
                                           symbolPen=pg.mkPen(None))
        self._sample_curve = self._pw.plot(pen=pg.mkPen("#ef4444", width=1.2,
                                                        style=Qt.PenStyle.SolidLine))

        # ── mode toggle row ───────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        mode_row.addWidget(mode_lbl)
        mode_row.addStretch()

        self._mode_btn = QPushButton("⎍ Pulse")
        self._mode_btn.setFixedWidth(100)
        self._mode_btn.setStyleSheet(BTN_STYLE)
        self._mode_btn.clicked.connect(self._toggle_mode)
        mode_row.addWidget(self._mode_btn)
        self._outer.addLayout(mode_row)

        # ── shared: amplitude ─────────────────────────────────────────────────
        self._amp_row = SliderRow("Amplitude", 0.0, 2.0, 1.0, AMP_COLOR, AMP_UNIT)
        self._amp_row.unit_btn.setEnabled(False)
        self._amp_row.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")
        self._amp_row.box.setText(fmt(1.0))
        self._outer.addWidget(self._amp_row)

        # ── pulse-mode rows ───────────────────────────────────────────────────
        self._width_row = SliderRow("Width %",  1.0,  99.0,  20.0, WIDTH_COLOR, "%")
        self._delay_row = SliderRow("Delay %",  0.0,  80.0,   0.0, DELAY_COLOR, "%")
        for r in (self._width_row, self._delay_row):
            r.unit_btn.setEnabled(False)
            r.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")
        self._width_row.box.setText(fmt(20.0))
        self._delay_row.box.setText(fmt(0.0))
        self._outer.addWidget(self._width_row)
        self._outer.addWidget(self._delay_row)

        # ── PRN-mode rows (hidden until PRN selected) ─────────────────────────
        # chip_rate: how many chips are displayed across the window (1–PRN_LENGTH)
        self._chip_row  = SliderRow("Chips shown", 1.0, float(PRN_LENGTH), 31.0,
                                    CHIP_RATE_COLOR, "chips")
        self._pdelay_row= SliderRow("Delay (smp)", 0.0, float(N_SAMPLES - 1), 0.0,
                                    DELAY_COLOR, "smp")
        for r in (self._chip_row, self._pdelay_row):
            r.unit_btn.setEnabled(False)
            r.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")
        self._chip_row.box.setText(fmt(31.0))
        self._pdelay_row.box.setText(fmt(0.0))
        self._outer.addWidget(self._chip_row)
        self._outer.addWidget(self._pdelay_row)
        self._chip_row.setVisible(False)
        self._pdelay_row.setVisible(False)

        # ── connect all ───────────────────────────────────────────────────────
        self._amp_row.slider.valueChanged.connect(self._on_amp_sl)
        self._amp_row.box.returnPressed.connect(self._on_amp_box)

        for row, lo, hi in [
            (self._width_row,  1.0,  99.0),
            (self._delay_row,  0.0,  80.0),
            (self._chip_row,   1.0,  float(PRN_LENGTH)),
            (self._pdelay_row, 0.0,  float(N_SAMPLES - 1)),
        ]:
            row.slider.valueChanged.connect(
                lambda _, r=row: self._on_simple_sl(r))
            row.box.returnPressed.connect(
                lambda _, r=row, l=lo, h=hi: self._on_simple_box(r, l, h))

        self._redraw()

    # ── public interface ──────────────────────────────────────────────────────
    def freq_hz(self) -> float:
        return 0.0   # no frequency — pulse/PRN is aperiodic in this panel

    def label(self) -> str:
        if self._mode == "pulse":
            return (f"Pulse  W={fmt(self._width_row.value())}%  "
                    f"D={fmt(self._delay_row.value())}%")
        return (f"PRN  chips={fmt(self._chip_row.value())}  "
                f"delay={fmt(self._pdelay_row.value())} smp")

    def _compute_signal(self, t_sec: np.ndarray) -> np.ndarray:
        n = len(t_sec)
        amp = self._amp_row.value()
        if self._mode == "pulse":
            w = self._width_row.value() / 100.0
            d = self._delay_row.value() / 100.0
            # clamp so pulse doesn't overflow window
            d = min(d, 1.0 - w)
            return _single_pulse(n, amp, w, d)
        else:
            chip_rate_sps = self._chip_row.value()
            delay_smp     = int(round(self._pdelay_row.value()))
            return _prn_signal(n, amp, chip_rate_sps, delay_smp)

    # ── internals ─────────────────────────────────────────────────────────────
    def _toggle_mode(self):
        self._mode = "prn" if self._mode == "pulse" else "pulse"
        is_pulse   = self._mode == "pulse"
        self._mode_btn.setText("⎍ Pulse" if is_pulse else "PRN")
        self._width_row.setVisible(is_pulse)
        self._delay_row.setVisible(is_pulse)
        self._chip_row.setVisible(not is_pulse)
        self._pdelay_row.setVisible(not is_pulse)
        # switch line colour
        color = "#fb923c" if is_pulse else "#c084fc"
        self._curve.setPen(pg.mkPen(color, width=1.5))
        self._redraw()

    def _redraw(self):
        t = np.linspace(0, 1.0, N_SAMPLES, endpoint=False)   # 1-second window, N points
        y = self._compute_signal(t)

        self._curve.setData(t, y)
        self._pw.setXRange(0, 1.0, padding=0)
        amp = self._amp_row.value()
        if self._mode == "pulse":
            self._pw.setYRange(-0.15, amp * 1.3 + 0.1)
        else:
            self._pw.setYRange(-amp * 1.3 - 0.1, amp * 1.3 + 0.1)
        self._pw.setLabel("bottom", "Time (s)", color=TEXT)

        if self._mode == "pulse":
            self._title_lbl.setText(
                f"Pulse   A={fmt(amp)} V   "
                f"Width={fmt(self._width_row.value())}%   "
                f"Delay={fmt(self._delay_row.value())}%")
        else:
            self._title_lbl.setText(
                f"PRN (127-chip LFSR)   A={fmt(amp)} V   "
                f"Chips shown={fmt(self._chip_row.value())}   "
                f"Delay={fmt(int(self._pdelay_row.value()))} smp")

        # ADC overlay — use N_SAMPLES points, same density
        sr = self._sample_rate_hz
        if sr > 0:
            n_s = max(2, int(round(sr)))   # sr samples per second, window = 1s
            t_s = np.linspace(0, 1.0, n_s, endpoint=False)
            y_s = self._compute_signal(t_s)
            self._sample_dots.setData(t_s, y_s)
            if len(t_s) > 1:
                t_h = np.repeat(t_s, 2)[1:]; y_h = np.repeat(y_s, 2)[:-1]
                self._sample_curve.setData(t_h, y_h)
            else:
                self._sample_curve.setData([], [])

        self.signal_changed.emit()

    # ── generic handlers ──────────────────────────────────────────────────────
    def _on_simple_sl(self, row):
        if self._guard: return
        self._guard = True; row.box.setText(fmt(row.value())); self._guard = False
        self._redraw()

    def _on_simple_box(self, row, lo, hi):
        try:    v = float(safe_eval(row.box.text()))
        except: row.box.setText(fmt(row.value())); return
        row.set_value(max(lo, min(hi, v)))
        row.box.setText(fmt(row.value()))
        self._redraw()

    def _on_amp_sl(self):
        if self._guard: return
        self._guard = True; self._amp_row.box.setText(fmt(self._amp_row.value())); self._guard = False
        self._redraw()

    def _on_amp_box(self):
        try:    v = float(safe_eval(self._amp_row.box.text()))
        except: self._amp_row.box.setText(fmt(self._amp_row.value())); return
        self._amp_row.set_value(v); self._redraw()
