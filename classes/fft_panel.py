"""fft_panel.py – FftPanel: right-side widget with signal + FFT plots."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame

from utils.dsp_utils import (compute_continuous_fft, compute_dft,
                              zero_order_hold, fft_view_range)
from utils.gui_utils import (BG, PANEL_BG, EDGE, TEXT,
                              MIX_SAMPLES, make_signal_plot, make_fft_plot,
                              add_crosshair)
from gui.widgets import SamplingRateRow


class FftPanel(QWidget):
    """
    Right-side panel containing:
      - Continuous superimposed signal plot  (blue)
      - Continuous FFT plot                  (blue)
      - Sampled superimposed signal plot     (red)
      - Sampled FFT / DFT plot               (red)
      - Sampling rate slider with Nyquist marker
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── continuous signal (blue) ──────────────────────────────────────────
        self._pw_sig_c, self._curve_sig_c = make_signal_plot(
            "Superimposed Signal (Continuous)", "#3b82f6")
        lay.addWidget(self._pw_sig_c, stretch=1)

        # ── continuous FFT (blue) ─────────────────────────────────────────────
        self._pw_fft_c, self._curve_fft_c = make_fft_plot(
            "X(f) = ∫ x(t) · e^(−j2πft) dt", "#3b82f6")
        lay.addWidget(self._pw_fft_c, stretch=1)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"color:{EDGE};"); lay.addWidget(sep1)

        # ── sampled signal (red) ──────────────────────────────────────────────
        self._pw_sig_s, self._curve_sig_s = make_signal_plot(
            "Superimposed Signal (Sampled / ADC)", "#ef4444")
        self._dots_s = self._pw_sig_s.plot(
            pen=None, symbol="o", symbolSize=5,
            symbolBrush=pg.mkBrush("#ef4444"), symbolPen=pg.mkPen(None))
        lay.addWidget(self._pw_sig_s, stretch=1)

        # ── sampled FFT / DFT (red) ───────────────────────────────────────────
        self._pw_fft_s, self._curve_fft_s = make_fft_plot(
            "X[k] = Σ(n=0→N-1) x[n] · e^(−j2πkn/N)", "#ef4444")
        self._stems_fft_s = self._pw_fft_s.plot(
            pen=None, symbol="o", symbolSize=4,
            symbolBrush=pg.mkBrush("#ef4444"), symbolPen=pg.mkPen(None))
        lay.addWidget(self._pw_fft_s, stretch=1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{EDGE};"); lay.addWidget(sep2)

        # ── sampling rate slider ──────────────────────────────────────────────
        self._sr = SamplingRateRow()
        lay.addWidget(self._sr)

        # ── crosshairs on FFT plots only ──────────────────────────────────────
        self._ch_c, self._chs_c = add_crosshair(self._pw_fft_c, "#3b82f6")
        self._ch_s, self._chs_s = add_crosshair(self._pw_fft_s, "#ef4444")
        self._chs_c["curve"] = self._curve_fft_c
        self._chs_s["curve"] = self._curve_fft_s

    # ── public ────────────────────────────────────────────────────────────────
    def sample_rate_hz(self) -> float:
        return self._sr.value_hz()

    def connect_sr_changed(self, cb):
        self._sr.value_changed = cb

    def update(self, panels: list):
        all_curves = (self._curve_sig_c, self._curve_fft_c,
                      self._curve_sig_s, self._dots_s,
                      self._curve_fft_s, self._stems_fft_s)
        if not panels:
            for c in all_curves: c.setData([], [])
            return

        sr_hz     = self._sr.value_hz()
        freqs_hz  = [p.freq_hz() for p in panels]
        lowest_hz = max(min((f for f in freqs_hz if f > 0), default=1.0), 0.01)
        x_window  = 3.0 / lowest_hz          # 3 periods of lowest freq
        n_samp    = max(2, int(round(sr_hz * x_window)))
        actual_sr = n_samp / x_window         # true sample rate for this window

        t_cont = np.linspace(0, x_window, MIX_SAMPLES, endpoint=False)
        t_samp = np.linspace(0, x_window, n_samp,      endpoint=False)

        # ── continuous superimposed ───────────────────────────────────────────
        sig_c = sum(p.get_signal(t_cont) for p in panels)
        self._curve_sig_c.setData(t_cont, sig_c)
        self._pw_sig_c.setXRange(0, x_window, padding=0)
        self._pw_sig_c.setYRange(-float(np.max(np.abs(sig_c))) * 1.2 or -1,
                                  float(np.max(np.abs(sig_c))) * 1.2 or  1)

        # ── continuous FFT ────────────────────────────────────────────────────
        fa, ma = compute_continuous_fft(sig_c, t_cont)
        res = fft_view_range(fa, ma)
        if res is not None:
            fa_v, ma_v, xs, xe, peak = res
            self._curve_fft_c.setData(fa_v, ma_v)
            self._pw_fft_c.setXRange(xs, xe, padding=0)
            self._pw_fft_c.setYRange(peak - 60, peak + 6)
        else:
            self._curve_fft_c.setData([], [])

        # ── sampled superimposed ──────────────────────────────────────────────
        sig_s = sum(p.get_signal(t_samp) for p in panels)
        self._dots_s.setData(t_samp, sig_s)
        t_h, y_h = zero_order_hold(t_samp, sig_s)
        self._curve_sig_s.setData(t_h, y_h)
        self._pw_sig_s.setXRange(0, x_window, padding=0)
        self._pw_sig_s.setYRange(-float(np.max(np.abs(sig_s))) * 1.2 or -1,
                                  float(np.max(np.abs(sig_s))) * 1.2 or  1)

        # ── DFT ──────────────────────────────────────────────────────────────
        fs, ms  = compute_dft(sig_s, actual_sr)
        nyquist = actual_sr / 2
        res_s   = fft_view_range(fs, ms, nyquist=nyquist)
        if res_s is not None:
            fs_v, ms_v, xs_s, xe_s, peak_s = res_s
            self._curve_fft_s.setData(fs_v, ms_v)
            self._stems_fft_s.setData(fs_v, ms_v)
            self._pw_fft_s.setXRange(xs_s, xe_s, padding=0)
            self._pw_fft_s.setYRange(peak_s - 60, peak_s + 6)
        else:
            self._curve_fft_s.setData([], [])
            self._stems_fft_s.setData([], [])

        # ── Nyquist marker ────────────────────────────────────────────────────
        if any(f > 0 for f in freqs_hz):
            self._sr.set_nyquist(max(f for f in freqs_hz if f > 0) * 2.0)
