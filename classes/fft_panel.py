"""fft_panel.py – FftPanel with mix mode, spike annotations, leakage, aliasing."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QFrame, QPushButton, QLabel)

from utils.dsp_utils import (compute_continuous_fft, compute_dft,
                              zero_order_hold, fft_view_range,
                              find_fft_peaks, aliased_frequency, fmt_hz)
from utils.gui_utils  import (BG, PANEL_BG, EDGE, TEXT, BTN_STYLE,
                               MIX_SAMPLES, make_signal_plot, make_fft_plot,
                               add_crosshair)
from gui.widgets      import SamplingRateRow


# ── small toolbar button ──────────────────────────────────────────────────────
def _tb_btn(label: str, active: bool = False) -> QPushButton:
    b = QPushButton(label)
    b.setCheckable(True)
    b.setChecked(active)
    b.setFixedHeight(26)
    _set_tb_style(b, active)
    b.toggled.connect(lambda on, btn=b: _set_tb_style(btn, on))
    return b

def _set_tb_style(btn: QPushButton, on: bool):
    if on:
        btn.setStyleSheet(
            "QPushButton{background:#1d4ed8;color:white;font-size:10px;"
            "border:none;border-radius:4px;padding:2px 10px;}"
            "QPushButton:hover{background:#2563eb;}")
    else:
        btn.setStyleSheet(
            "QPushButton{background:#2a2a2a;color:#9ca3af;font-size:10px;"
            "border:1px solid #444;border-radius:4px;padding:2px 10px;}"
            "QPushButton:hover{background:#3a3a3a;}")


class FftPanel(QWidget):
    """
    Right-side panel:
      - Toolbar: Sum/Product mix mode | Show Leakage | Show Aliasing
      - Continuous signal + FFT (blue) with spike labels
      - Sampled signal + DFT (red) with spike labels + aliasing overlay
      - Sampling rate slider with Nyquist marker
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = QWidget()
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.setSpacing(6)

        mode_lbl = QLabel("Mix:")
        mode_lbl.setStyleSheet(f"color:{TEXT}; font-size:10px; font-weight:bold;")
        tb_lay.addWidget(mode_lbl)

        self._btn_sum  = _tb_btn("∑ Sum",     active=True)
        self._btn_prod = _tb_btn("× Product", active=False)
        self._btn_sum.toggled.connect(lambda on: self._btn_prod.setChecked(not on) if on else None)
        self._btn_prod.toggled.connect(lambda on: self._btn_sum.setChecked(not on) if on else None)
        tb_lay.addWidget(self._btn_sum)
        tb_lay.addWidget(self._btn_prod)

        tb_lay.addSpacing(16)
        self._btn_leakage = _tb_btn("◈ Leakage",  active=False)
        self._btn_alias   = _tb_btn("⚡ Aliasing", active=False)
        tb_lay.addWidget(self._btn_leakage)
        tb_lay.addWidget(self._btn_alias)
        tb_lay.addStretch()

        for btn in (self._btn_sum, self._btn_prod, self._btn_leakage, self._btn_alias):
            btn.toggled.connect(lambda _: self._request_update())

        lay.addWidget(tb)

        # ── continuous signal (blue) ──────────────────────────────────────────
        self._pw_sig_c, self._curve_sig_c = make_signal_plot(
            "Superimposed Signal (Continuous)", "#3b82f6")
        lay.addWidget(self._pw_sig_c, stretch=1)

        # ── continuous FFT (blue) ─────────────────────────────────────────────
        self._pw_fft_c, self._curve_fft_c = make_fft_plot(
            "X(f) = ∫ x(t) · e^(−j2πft) dt", "#3b82f6")
        # leakage reference (dimmer blue)
        self._curve_leak = self._pw_fft_c.plot(
            pen=pg.mkPen("#1e40af", width=1, style=Qt.PenStyle.DashLine))
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
        # aliasing markers (yellow)
        self._alias_stems = self._pw_fft_s.plot(
            pen=None, symbol="d", symbolSize=10,
            symbolBrush=pg.mkBrush("#facc15"), symbolPen=pg.mkPen("#facc15", width=1))
        self._alias_vlines: list[pg.InfiniteLine] = []
        lay.addWidget(self._pw_fft_s, stretch=1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{EDGE};"); lay.addWidget(sep2)

        # ── sampling rate slider ──────────────────────────────────────────────
        self._sr = SamplingRateRow()
        lay.addWidget(self._sr)

        # ── crosshairs ────────────────────────────────────────────────────────
        self._ch_c, self._chs_c = add_crosshair(self._pw_fft_c, "#3b82f6")
        self._ch_s, self._chs_s = add_crosshair(self._pw_fft_s, "#ef4444")
        self._chs_c["curve"] = self._curve_fft_c
        self._chs_s["curve"] = self._curve_fft_s

        # spike annotation text items
        self._spike_labels_c: list[pg.TextItem] = []
        self._spike_labels_s: list[pg.TextItem] = []

        # store panels ref for toolbar-triggered redraw
        self._last_panels: list = []

    # ── public ────────────────────────────────────────────────────────────────
    def sample_rate_hz(self) -> float:
        return self._sr.value_hz()

    def connect_sr_changed(self, cb):
        self._sr.value_changed = cb

    def _request_update(self):
        if self._last_panels:
            self.update(self._last_panels)

    def update(self, panels: list):
        self._last_panels = panels
        active = [p for p in panels if p.enabled]

        all_curves = (self._curve_sig_c, self._curve_fft_c, self._curve_leak,
                      self._curve_sig_s, self._dots_s,
                      self._curve_fft_s, self._stems_fft_s, self._alias_stems)
        if not active:
            for c in all_curves: c.setData([], [])
            self._clear_spike_labels()
            return

        sr_hz      = self._sr.value_hz()
        freqs_hz   = [p.freq_hz() for p in active]
        pos_freqs  = [f for f in freqs_hz if f > 0]
        lowest_hz  = max(min(pos_freqs, default=1.0), 0.01)
        x_window   = 3.0 / lowest_hz
        n_samp     = max(2, int(round(sr_hz * x_window)))
        actual_sr  = n_samp / x_window

        t_cont = np.linspace(0, x_window, MIX_SAMPLES, endpoint=False)
        t_samp = np.linspace(0, x_window, n_samp,      endpoint=False)

        mix_sum  = self._btn_sum.isChecked()

        # ── compose signals ───────────────────────────────────────────────────
        if mix_sum:
            sig_c = sum(p.get_signal(t_cont) for p in active)
            sig_s = sum(p.get_signal(t_samp) for p in active)
        else:
            sig_c = np.ones(len(t_cont))
            sig_s = np.ones(len(t_samp))
            for p in active:
                sig_c *= p.get_signal(t_cont)
                sig_s *= p.get_signal(t_samp)

        # ── continuous signal plot ────────────────────────────────────────────
        self._curve_sig_c.setData(t_cont, sig_c)
        self._pw_sig_c.setXRange(0, x_window, padding=0)
        ymax_c = float(np.max(np.abs(sig_c))) * 1.2 or 1.0
        self._pw_sig_c.setYRange(-ymax_c, ymax_c)
        title_c = "Modulated Signal" if not mix_sum else "Superimposed Signal (Continuous)"
        self._pw_sig_c.setTitle(title_c, color=TEXT, size="10pt")

        # ── continuous FFT ────────────────────────────────────────────────────
        fa, ma = compute_continuous_fft(sig_c, t_cont)
        res = fft_view_range(fa, ma)
        if res:
            fa_v, ma_v, xs, xe, peak = res
            self._curve_fft_c.setData(fa_v, ma_v)
            self._pw_fft_c.setXRange(xs, xe, padding=0)
            self._pw_fft_c.setYRange(peak - 60, peak + 6)
            self._annotate_spikes(fa_v, ma_v, self._pw_fft_c, self._spike_labels_c, "#93c5fd")

            # leakage: show rectangular-window FFT (no windowing = leakage visible)
            if self._btn_leakage.isChecked():
                # apply Hann window and show difference as the leakage reference
                hann    = np.hanning(len(sig_c))
                fa_h, ma_h = compute_continuous_fft(sig_c * hann, t_cont)
                mask_h  = (fa_h > 0) & np.isfinite(ma_h) & (fa_h >= xs) & (fa_h <= xe)
                self._curve_leak.setData(fa_h[mask_h], ma_h[mask_h])
            else:
                self._curve_leak.setData([], [])
        else:
            self._curve_fft_c.setData([], [])
            self._curve_leak.setData([], [])
            self._clear_spike_labels(which="c")

        # ── sampled signal plot ───────────────────────────────────────────────
        self._dots_s.setData(t_samp, sig_s)
        t_h, y_h = zero_order_hold(t_samp, sig_s)
        self._curve_sig_s.setData(t_h, y_h)
        self._pw_sig_s.setXRange(0, x_window, padding=0)
        ymax_s = float(np.max(np.abs(sig_s))) * 1.2 or 1.0
        self._pw_sig_s.setYRange(-ymax_s, ymax_s)
        title_s = "Modulated Signal (Sampled)" if not mix_sum else "Superimposed Signal (Sampled / ADC)"
        self._pw_sig_s.setTitle(title_s, color=TEXT, size="10pt")

        # ── DFT ──────────────────────────────────────────────────────────────
        fs, ms  = compute_dft(sig_s, actual_sr)
        nyquist = actual_sr / 2
        res_s   = fft_view_range(fs, ms, nyquist=nyquist)
        if res_s:
            fs_v, ms_v, xs_s, xe_s, peak_s = res_s
            self._curve_fft_s.setData(fs_v, ms_v)
            self._stems_fft_s.setData(fs_v, ms_v)
            self._pw_fft_s.setXRange(xs_s, xe_s, padding=0)
            self._pw_fft_s.setYRange(peak_s - 60, peak_s + 6)
            self._annotate_spikes(fs_v, ms_v, self._pw_fft_s, self._spike_labels_s, "#fca5a5")

            # aliasing overlay
            self._clear_alias_lines()
            if self._btn_alias.isChecked() and pos_freqs:
                alias_fs, alias_ms = [], []
                for f_sig in pos_freqs:
                    f_al = aliased_frequency(f_sig, actual_sr)
                    if abs(f_al - f_sig) > 1.0:   # only show if different from original
                        # find nearest DFT bin for amplitude
                        idx = int(np.argmin(np.abs(fs_v - f_al)))
                        alias_fs.append(f_al)
                        alias_ms.append(float(ms_v[idx]) if len(ms_v) else peak_s - 10)
                        # draw vertical yellow line
                        vl = pg.InfiniteLine(
                            pos=f_al, angle=90, movable=False,
                            pen=pg.mkPen("#facc15", width=1.5, style=Qt.PenStyle.DashLine),
                            label=f"alias\n{fmt_hz(f_al)}",
                            labelOpts={"color": "#facc15", "position": 0.85,
                                       "fill": pg.mkBrush("#1c1c1c")})
                        self._pw_fft_s.addItem(vl)
                        self._alias_vlines.append(vl)
                self._alias_stems.setData(alias_fs, alias_ms)
            else:
                self._alias_stems.setData([], [])
        else:
            self._curve_fft_s.setData([], [])
            self._stems_fft_s.setData([], [])
            self._alias_stems.setData([], [])
            self._clear_spike_labels(which="s")

        # ── Nyquist marker ────────────────────────────────────────────────────
        if pos_freqs:
            self._sr.set_nyquist(max(pos_freqs) * 2.0)

    # ── spike annotations ─────────────────────────────────────────────────────
    def _annotate_spikes(self, freqs, mags, pw, label_list, color):
        # remove old labels
        for lbl in label_list:
            pw.removeItem(lbl)
        label_list.clear()

        peaks = find_fft_peaks(freqs, mags, threshold_db=20.0)
        for f_hz, m_dbm in peaks:
            txt = pg.TextItem(
                text=f"{fmt_hz(f_hz)}\n{m_dbm:.1f} dBm",
                anchor=(0.5, 1.0), color=color)
            font = QFont(); font.setPixelSize(9)
            txt.setFont(font)
            txt.setPos(f_hz, m_dbm)
            pw.addItem(txt)
            label_list.append(txt)

    def _clear_spike_labels(self, which: str = "both"):
        if which in ("c", "both"):
            for lbl in self._spike_labels_c:
                self._pw_fft_c.removeItem(lbl)
            self._spike_labels_c.clear()
        if which in ("s", "both"):
            for lbl in self._spike_labels_s:
                self._pw_fft_s.removeItem(lbl)
            self._spike_labels_s.clear()

    def _clear_alias_lines(self):
        for vl in self._alias_vlines:
            self._pw_fft_s.removeItem(vl)
        self._alias_vlines.clear()
