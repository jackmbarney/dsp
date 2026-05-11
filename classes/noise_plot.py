"""noise_plot.py – NoisePlotPanel: noise signal panel (white/pink/Gaussian)."""

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from utils.dsp_utils import generate_noise, fmt
from utils.gui_utils  import (TEXT, N_SAMPLES, VIEW_PERIODS,
                               AMP_COLOR, BTN_STYLE, BOX_STYLE, FREQ_UNITS)
from gui.widgets      import SliderRow
from classes.base_plot import BasePlotPanel

NOISE_TYPES = ["White", "Pink", "Gaussian"]
NOISE_COLORS = {"White": "#e5e5e5", "Pink": "#f9a8d4", "Gaussian": "#86efac"}


class NoisePlotPanel(BasePlotPanel):
    """Noise signal panel — amplitude slider + noise-type cycle button."""

    LINE_COLOR = "#e5e5e5"

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(sample_rate_hz, parent)
        self._noise_idx = 0
        self._cached    = np.zeros(N_SAMPLES)

        # amplitude row
        self._amp_row = SliderRow("Amplitude", 0.0, 2.0, 0.3, AMP_COLOR, "V")
        self._amp_row.unit_btn.setEnabled(False)
        self._amp_row.unit_btn.setStyleSheet(BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")
        self._outer.addWidget(self._amp_row)

        # noise type button
        type_row = QHBoxLayout()
        type_lbl = QLabel("Noise type")
        type_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        type_row.addWidget(type_lbl)
        type_row.addStretch()
        self._type_btn = QPushButton(NOISE_TYPES[0])
        self._type_btn.setFixedWidth(90)
        self._type_btn.setStyleSheet(BTN_STYLE)
        self._type_btn.clicked.connect(self._cycle_type)
        type_row.addWidget(self._type_btn)
        self._outer.addLayout(type_row)

        self._amp_row.slider.valueChanged.connect(self._on_amp)
        self._amp_row.box.returnPressed.connect(self._on_amp_box)

        self._redraw()

    # ── public interface ──────────────────────────────────────────────────────
    def freq_hz(self) -> float:
        return 0.0   # noise has no single frequency

    def label(self) -> str:
        return f"{NOISE_TYPES[self._noise_idx]} Noise"

    def _compute_signal(self, t_sec: np.ndarray) -> np.ndarray:
        noise_type = NOISE_TYPES[self._noise_idx].lower()
        return generate_noise(len(t_sec), noise_type, self._amp_row.value())

    # ── internals ─────────────────────────────────────────────────────────────
    def _redraw(self):
        t = np.linspace(0, VIEW_PERIODS, N_SAMPLES, endpoint=False)
        self._cached = generate_noise(N_SAMPLES, NOISE_TYPES[self._noise_idx].lower(),
                                      self._amp_row.value())
        color = NOISE_COLORS[NOISE_TYPES[self._noise_idx]]
        self._curve.setPen(color, width=1.2)
        self._curve.setData(t, self._cached)
        self._pw.setXRange(0, VIEW_PERIODS, padding=0)
        amp = self._amp_row.value()
        self._pw.setYRange(-amp * 1.5 - 0.1, amp * 1.5 + 0.1)
        self._title_lbl.setText(
            f"{NOISE_TYPES[self._noise_idx]} Noise   A={fmt(amp)} V")
        self.signal_changed.emit()

    def _cycle_type(self):
        self._noise_idx = (self._noise_idx + 1) % len(NOISE_TYPES)
        self._type_btn.setText(NOISE_TYPES[self._noise_idx])
        self._redraw()

    def _on_amp(self):
        self._amp_row.box.setText(fmt(self._amp_row.value()))
        self._redraw()

    def _on_amp_box(self):
        try:    v = float(self._amp_row.box.text())
        except: self._amp_row.box.setText(fmt(self._amp_row.value())); return
        self._amp_row.set_value(v)
        self._redraw()
