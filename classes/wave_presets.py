"""wave_presets.py – SquareWave, SawtoothWave, TriangleWave preset panels."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from utils.dsp_utils import (square_wave, sawtooth_wave, triangle_wave,
                              safe_eval, fmt, FREQ_SUFFIX)
from utils.gui_utils  import (FREQ_UNITS, AMP_UNIT, AMP_COLOR, FREQ_COLOR,
                               TEXT, N_SAMPLES, VIEW_PERIODS, LOW_FREQ_FLOOR, BTN_STYLE)
from gui.widgets      import SliderRow
from classes.base_plot import BasePlotPanel


class _PresetWavePanel(BasePlotPanel):
    """Shared base for all Fourier-series preset panels."""

    WAVE_FN    = staticmethod(square_wave)   # override in subclass
    WAVE_LABEL = "Wave"
    LINE_COLOR = "#a78bfa"

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(sample_rate_hz, parent)
        self._freq_unit_idx = 0
        self._guard         = False

        # ADC sample overlay
        self._sample_dots  = self._pw.plot(pen=None, symbol="o", symbolSize=6,
                                           symbolBrush=pg.mkBrush("#ef4444"),
                                           symbolPen=pg.mkPen(None))
        self._sample_curve = self._pw.plot(pen=pg.mkPen("#ef4444", width=1.2,
                                                        style=Qt.PenStyle.SolidLine))

        self._amp_row  = SliderRow("Amplitude", 0.0,   2.0,   1.0, AMP_COLOR,  AMP_UNIT)
        self._freq_row = SliderRow("Frequency", 0.0, 1000.0,  1.0, FREQ_COLOR, FREQ_UNITS[0][0])

        self._amp_row.unit_btn.setEnabled(False)
        self._amp_row.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")

        for row in (self._amp_row, self._freq_row):
            self._outer.addWidget(row)

        self._amp_row.box.setText(fmt(1.0))
        self._freq_row.box.setText(fmt(1.0))

        self._amp_row.slider.valueChanged.connect(self._on_amp_sl)
        self._amp_row.box.returnPressed.connect(self._on_amp_box)
        self._freq_row.slider.valueChanged.connect(self._on_freq_sl)
        self._freq_row.box.returnPressed.connect(self._on_freq_box)
        self._freq_row.unit_btn.clicked.connect(self._cycle_freq)

        self._redraw()

    # ── public interface ──────────────────────────────────────────────────────
    def freq_hz(self) -> float:
        return self._freq_row.value() * FREQ_UNITS[self._freq_unit_idx][1]

    def label(self) -> str:
        fu = FREQ_UNITS[self._freq_unit_idx][0]
        return f"{self.WAVE_LABEL} {fmt(self._freq_row.value())} {fu}"

    def _compute_signal(self, t_sec: np.ndarray) -> np.ndarray:
        return self.WAVE_FN(t_sec, self.freq_hz(), self._amp_row.value())

    # ── internals ─────────────────────────────────────────────────────────────
    def _redraw(self):
        fhz   = self.freq_hz()
        f_win = max(fhz, LOW_FREQ_FLOOR * FREQ_UNITS[self._freq_unit_idx][1])
        dur   = VIEW_PERIODS / f_win
        t     = np.linspace(0, dur, N_SAMPLES, endpoint=False)
        y     = self._compute_signal(t)
        x     = t * FREQ_UNITS[self._freq_unit_idx][1]

        self._curve.setData(x, y)
        self._pw.setXRange(0, float(x[-1]) if x[-1] > 0 else 1.0, padding=0)
        self._pw.setLabel("bottom", f"Time ({FREQ_UNITS[self._freq_unit_idx][2]})", color=TEXT)

        fu = FREQ_UNITS[self._freq_unit_idx][0]
        self._title_lbl.setText(
            f"{self.WAVE_LABEL}   A={fmt(self._amp_row.value())} V   "
            f"f={fmt(self._freq_row.value())} {fu}")

        # ADC overlay
        sr = self._sample_rate_hz
        if sr > 0:
            n_s = max(2, int(round(sr * dur)))
            t_s = np.linspace(0, dur, n_s, endpoint=False)
            y_s = self._compute_signal(t_s)
            x_s = t_s * FREQ_UNITS[self._freq_unit_idx][1]
            self._sample_dots.setData(x_s, y_s)
            if len(t_s) > 1:
                t_h = np.repeat(t_s, 2)[1:]; y_h = np.repeat(y_s, 2)[:-1]
                self._sample_curve.setData(t_h * FREQ_UNITS[self._freq_unit_idx][1], y_h)
            else:
                self._sample_curve.setData([], [])

        self.signal_changed.emit()

    def _on_amp_sl(self):
        self._amp_row.box.setText(fmt(self._amp_row.value())); self._redraw()

    def _on_amp_box(self):
        try:    v = float(safe_eval(self._amp_row.box.text()))
        except: self._amp_row.box.setText(fmt(self._amp_row.value())); return
        self._amp_row.set_value(v); self._redraw()

    def _on_freq_sl(self):
        if self._guard: return
        self._guard = True; self._freq_row.box.setText(fmt(self._freq_row.value())); self._guard = False
        self._redraw()

    def _on_freq_box(self):
        text = self._freq_row.box.text().strip(); target = None; num = text
        if text and text[-1] in FREQ_SUFFIX:
            target = FREQ_SUFFIX[text[-1]]; num = text[:-1]
        try:    value = float(safe_eval(num))
        except: self._freq_row.box.setText(fmt(self._freq_row.value())); return
        if target is not None and target != self._freq_unit_idx:
            fhz = value * FREQ_UNITS[target][1]
            self._freq_unit_idx = target
            self._freq_row.unit_btn.setText(FREQ_UNITS[target][0])
            value = fhz / FREQ_UNITS[target][1]
        value = max(0.0, min(1000.0, value))
        self._guard = True; self._freq_row.set_value(value); self._freq_row.box.setText(fmt(value)); self._guard = False
        self._redraw()

    def _cycle_freq(self):
        self._freq_unit_idx = (self._freq_unit_idx + 1) % len(FREQ_UNITS)
        self._freq_row.unit_btn.setText(FREQ_UNITS[self._freq_unit_idx][0])
        self._redraw()


class SquareWavePlot(_PresetWavePanel):
    WAVE_FN    = staticmethod(square_wave)
    WAVE_LABEL = "Square Wave"
    LINE_COLOR = "#f59e0b"


class SawtoothWavePlot(_PresetWavePanel):
    WAVE_FN    = staticmethod(sawtooth_wave)
    WAVE_LABEL = "Sawtooth Wave"
    LINE_COLOR = "#34d399"


class TriangleWavePlot(_PresetWavePanel):
    WAVE_FN    = staticmethod(triangle_wave)
    WAVE_LABEL = "Triangle Wave"
    LINE_COLOR = "#c084fc"
