"""widgets.py – reusable Qt widgets: SliderRow and SamplingRateRow."""

import numpy as np
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui  import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLineEdit, QPushButton, QLabel,
)

from utils.dsp_utils import safe_eval, fmt, FREQ_SUFFIX
from utils.gui_utils import (FREQ_UNITS, TEXT, BTN_STYLE, BOX_STYLE,
                         slider_style, SR_COLOR)

SR_MIN   = 2.0
SR_MAX   = 1000.0
SR_INIT  = 100.0
SR_STEPS = 2000


class SliderRow(QWidget):
    """
    Generic  label | slider | textbox | unit-button  row.

    Subclass or instantiate directly; connect slider.valueChanged and
    box.returnPressed as needed.
    """

    def __init__(self, label: str, sl_min: float, sl_max: float, sl_init: float,
                 color: str, unit_text: str, steps: int = 2000, parent=None):
        super().__init__(parent)
        self._min   = sl_min
        self._max   = sl_max
        self._steps = steps

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        lay.addWidget(lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._to_int(sl_init))
        self.slider.setStyleSheet(slider_style(color))
        self.slider.setFixedHeight(22)
        lay.addWidget(self.slider, stretch=1)

        self.box = QLineEdit()
        self.box.setFixedWidth(72)
        self.box.setStyleSheet(BOX_STYLE)
        lay.addWidget(self.box)

        self.unit_btn = QPushButton(unit_text)
        self.unit_btn.setFixedWidth(52)
        self.unit_btn.setStyleSheet(BTN_STYLE)
        lay.addWidget(self.unit_btn)

    def _to_int(self, v: float) -> int:
        return int((v - self._min) / (self._max - self._min) * self._steps)

    def value(self) -> float:
        return self._min + self.slider.value() / self._steps * (self._max - self._min)

    def set_value(self, v: float):
        self.slider.setValue(self._to_int(max(self._min, min(self._max, v))))

    def set_max(self, new_max: float):
        self._max = new_max

    def set_color(self, color: str):
        self.slider.setStyleSheet(slider_style(color))


class SamplingRateRow(QWidget):
    """
    Pink sampling-rate slider with:
    - k/M/G suffix parsing + safe_eval in the text box
    - Nyquist marker (yellow line) drawn over the slider
    - Slider turns green ≥ Nyquist, red < Nyquist
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._unit_idx   = 0
        self._min        = SR_MIN
        self._max        = SR_MAX
        self._steps      = SR_STEPS
        self._guard      = False
        self._nyquist_hz = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        lbl = QLabel("Sample Rate")
        lbl.setFixedWidth(86)
        lbl.setStyleSheet(f"color:{TEXT}; font-size:11px; font-weight:bold;")
        row.addWidget(lbl)

        # slider lives inside a wrapper so we can paint the Nyquist tick on top
        self._sw = QWidget()
        self._sw.setFixedHeight(30)
        row.addWidget(self._sw, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal, self._sw)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._to_int(SR_INIT))
        self.slider.setGeometry(0, 4, 200, 22)
        self._update_color()

        self.box = QLineEdit()
        self.box.setFixedWidth(72)
        self.box.setStyleSheet(BOX_STYLE)
        self.box.setText(fmt(SR_INIT))
        row.addWidget(self.box)

        self.unit_btn = QPushButton(FREQ_UNITS[0][0])
        self.unit_btn.setFixedWidth(52)
        self.unit_btn.setStyleSheet(BTN_STYLE)
        row.addWidget(self.unit_btn)

        outer.addLayout(row)

        self.slider.valueChanged.connect(self._on_slider)
        self.box.returnPressed.connect(self._on_box)
        self.unit_btn.clicked.connect(self._cycle_unit)
        self._sw.resizeEvent = lambda e: self.slider.setGeometry(0, 4, e.size().width(), 22)
        self._sw.paintEvent  = lambda e: self._paint_nyquist()

    # ── public ────────────────────────────────────────────────────────────────
    def value_hz(self) -> float:
        return self._raw_val() * FREQ_UNITS[self._unit_idx][1]

    def set_nyquist(self, nyquist_hz: float):
        self._nyquist_hz = nyquist_hz
        self._update_color()
        self._sw.update()

    # ── internals ─────────────────────────────────────────────────────────────
    def _to_int(self, v: float) -> int:
        return int((v - self._min) / (self._max - self._min) * self._steps)

    def _raw_val(self) -> float:
        return self._min + self.slider.value() / self._steps * (self._max - self._min)

    def _update_color(self):
        if self._nyquist_hz is None:
            color = "#22c55e"
        else:
            color = "#22c55e" if self.value_hz() >= self._nyquist_hz else "#ef4444"
        self.slider.setStyleSheet(slider_style(color))

    def _paint_nyquist(self):
        if self._nyquist_hz is None: return
        nyq_raw = self._nyquist_hz / FREQ_UNITS[self._unit_idx][1]
        ratio   = max(0.0, min(1.0, (nyq_raw - self._min) / (self._max - self._min)))
        x       = int(ratio * self._sw.width())
        p = QPainter(self._sw)
        p.setPen(QPen(QColor("#facc15"), 2))
        p.drawLine(x, 0, x, 28)
        f = QFont(); f.setPixelSize(8)
        p.setFont(f)
        p.setPen(QColor("#facc15"))
        p.drawText(x + 2, 26, "Nyq")
        p.end()

    def _on_slider(self):
        if self._guard: return
        self._guard = True
        self.box.setText(fmt(self._raw_val()))
        self._guard = False
        self._update_color(); self._sw.update()
        self.value_changed()

    def _on_box(self):
        text = self.box.text().strip(); target = None; num = text
        if text and text[-1] in FREQ_SUFFIX:
            target = FREQ_SUFFIX[text[-1]]; num = text[:-1]
        try:   value = float(safe_eval(num))
        except: self.box.setText(fmt(self._raw_val())); return
        if target is not None and target != self._unit_idx:
            hz = value * FREQ_UNITS[target][1]
            self._unit_idx = target
            self.unit_btn.setText(FREQ_UNITS[target][0])
            value = hz / FREQ_UNITS[target][1]
        value = max(self._min, min(self._max, value))
        self._guard = True
        self.slider.setValue(self._to_int(value))
        self.box.setText(fmt(value))
        self._guard = False
        self._update_color(); self._sw.update()
        self.value_changed()

    def _cycle_unit(self):
        self._unit_idx = (self._unit_idx + 1) % len(FREQ_UNITS)
        self.unit_btn.setText(FREQ_UNITS[self._unit_idx][0])
        self._update_color(); self._sw.update()
        self.value_changed()

    def value_changed(self):
        """Override or monkey-patch to react to value changes."""
        pass
