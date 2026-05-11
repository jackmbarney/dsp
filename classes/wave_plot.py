"""wave_plot.py – WavePlotPanel: sine-wave panel."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from utils.dsp_utils import safe_eval, fmt, fmt_phase, FREQ_SUFFIX
from utils.gui_utils  import (FREQ_UNITS, PHASE_UNITS, AMP_UNIT,
                               AMP_COLOR, FREQ_COLOR, PHASE_COLOR,
                               TEXT, N_SAMPLES, VIEW_PERIODS, LOW_FREQ_FLOOR, BTN_STYLE)
from gui.widgets      import SliderRow
from classes.base_plot import BasePlotPanel


class WavePlotPanel(BasePlotPanel):
    """Sine-wave panel with amplitude / frequency / phase controls."""

    LINE_COLOR = "#3b82f6"

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(sample_rate_hz, parent)
        self._freq_unit_idx  = 0
        self._phase_unit_idx = 0
        self._guard          = False

        # ADC sample overlay curves
        self._sample_dots  = self._pw.plot(pen=None, symbol="o", symbolSize=6,
                                           symbolBrush=pg.mkBrush("#ef4444"),
                                           symbolPen=pg.mkPen(None))
        self._sample_curve = self._pw.plot(pen=pg.mkPen("#ef4444", width=1.5,
                                                        style=Qt.PenStyle.SolidLine))

        # control rows
        self._amp_row   = SliderRow("Amplitude", 0.0,   2.0,             1.0, AMP_COLOR,   AMP_UNIT)
        self._freq_row  = SliderRow("Frequency", 0.0, 1000.0,            1.0, FREQ_COLOR,  FREQ_UNITS[0][0])
        self._phase_row = SliderRow("Phase",     0.0, PHASE_UNITS[0][1], 0.0, PHASE_COLOR, PHASE_UNITS[0][0])

        self._amp_row.unit_btn.setEnabled(False)
        self._amp_row.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")

        for row in (self._amp_row, self._freq_row, self._phase_row):
            self._outer.addWidget(row)

        self._amp_row.box.setText(fmt(1.0))
        self._freq_row.box.setText(fmt(1.0))
        self._phase_row.box.setText("0")

        self._amp_row.slider.valueChanged.connect(self._on_amp_sl)
        self._amp_row.box.returnPressed.connect(self._on_amp_box)
        self._freq_row.slider.valueChanged.connect(self._on_freq_sl)
        self._freq_row.box.returnPressed.connect(self._on_freq_box)
        self._freq_row.unit_btn.clicked.connect(self._cycle_freq)
        self._phase_row.slider.valueChanged.connect(self._on_phase_sl)
        self._phase_row.box.returnPressed.connect(self._on_phase_box)
        self._phase_row.unit_btn.clicked.connect(self._cycle_phase)

        self._redraw()

    # ── public interface ──────────────────────────────────────────────────────
    def freq_hz(self) -> float:
        return self._freq_row.value() * FREQ_UNITS[self._freq_unit_idx][1]

    def label(self) -> str:
        fu = FREQ_UNITS[self._freq_unit_idx][0]
        return f"Sine {fmt(self._freq_row.value())} {fu}"

    def _compute_signal(self, t_sec: np.ndarray) -> np.ndarray:
        return self._amp_row.value() * np.sin(
            2 * np.pi * self.freq_hz() * t_sec
            + self._phase_to_rad(self._phase_row.value()))

    def get_sampled_signal(self, t_sec: np.ndarray):
        if self._sample_rate_hz <= 0:
            return np.array([]), np.array([])
        fhz = self.freq_hz()
        if fhz > 0:
            spp = max(4, int(round(self._sample_rate_hz / fhz)))
            n   = max(N_SAMPLES, spp * 3)
        else:
            n   = max(2, int(round(self._sample_rate_hz * (t_sec[-1] - t_sec[0]))))
        t_s = np.linspace(t_sec[0], t_sec[-1], n, endpoint=False)
        return t_s, self._compute_signal(t_s)

    # ── internals ─────────────────────────────────────────────────────────────
    def _phase_to_rad(self, v): return v * np.pi / PHASE_UNITS[self._phase_unit_idx][2]
    def _fmt_ph(self, v):       return fmt_phase(v, self._phase_unit_idx, PHASE_UNITS)

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
        pu = PHASE_UNITS[self._phase_unit_idx][0]
        self._title_lbl.setText(
            f"y = A·sin(2πft+φ)   A={fmt(self._amp_row.value())} V  "
            f"f={fmt(self._freq_row.value())} {fu}  φ={self._fmt_ph(self._phase_row.value())} {pu}")

        t_s, y_s = self.get_sampled_signal(t)
        x_s = t_s * FREQ_UNITS[self._freq_unit_idx][1]
        self._sample_dots.setData(x_s, y_s)
        if len(t_s) > 1:
            t_h = np.repeat(t_s, 2)[1:]; y_h = np.repeat(y_s, 2)[:-1]
            self._sample_curve.setData(t_h * FREQ_UNITS[self._freq_unit_idx][1], y_h)
        else:
            self._sample_curve.setData([], [])

        self.signal_changed.emit()

    # ── amplitude ─────────────────────────────────────────────────────────────
    def _on_amp_sl(self):
        if self._guard: return
        self._guard = True; self._amp_row.box.setText(fmt(self._amp_row.value())); self._guard = False
        self._redraw()

    def _on_amp_box(self):
        try:    v = float(safe_eval(self._amp_row.box.text()))
        except: self._amp_row.box.setText(fmt(self._amp_row.value())); return
        self._guard = True; self._amp_row.set_value(v); self._guard = False
        self._redraw()

    # ── frequency ─────────────────────────────────────────────────────────────
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

    # ── phase ─────────────────────────────────────────────────────────────────
    def _on_phase_sl(self):
        if self._guard: return
        self._guard = True; self._phase_row.box.setText(self._fmt_ph(self._phase_row.value())); self._guard = False
        self._redraw()

    def _on_phase_box(self):
        try:    v = float(safe_eval(self._phase_row.box.text()))
        except: self._phase_row.box.setText(self._fmt_ph(self._phase_row.value())); return
        pu = PHASE_UNITS[self._phase_unit_idx]
        self._guard = True; self._phase_row.set_value(max(0.0, min(pu[1], v))); self._guard = False
        self._redraw()

    def _cycle_phase(self):
        cur_rad = self._phase_to_rad(self._phase_row.value())
        self._phase_unit_idx = (self._phase_unit_idx + 1) % len(PHASE_UNITS)
        pu = PHASE_UNITS[self._phase_unit_idx]
        nv = max(0.0, min(pu[1], cur_rad * pu[2] / np.pi))
        self._phase_row.set_max(pu[1])
        self._phase_row.unit_btn.setText(pu[0])
        self._guard = True; self._phase_row.set_value(nv); self._phase_row.box.setText(self._fmt_ph(nv)); self._guard = False
        self._redraw()
