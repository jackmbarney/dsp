"""SinWavePanel – one sine-wave panel built with PyQt6 + pyqtgraph."""

import ast
import operator

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLineEdit, QPushButton, QLabel, QFrame,
)

# ── constants ─────────────────────────────────────────────────────────────────
AMP_UNIT    = "V"
FREQ_UNITS  = [("Hz", 1.0, "s"), ("kHz", 1e3, "ms"), ("MHz", 1e6, "µs"), ("GHz", 1e9, "ns")]
PHASE_UNITS = [
    ("rad", 2 * np.pi, np.pi,  "Phase (0-2pi)"),
    ("deg", 360.0,     180.0,  "Phase (0-360 deg)"),
]
AMP_COLOR   = "#22d3ee"
FREQ_COLOR  = "#fbbf24"
PHASE_COLOR = "#f472b6"
BG          = "#0a0a0a"
PANEL_BG    = "#111111"
EDGE        = "#333333"
TEXT        = "#e5e5e5"
N_SAMPLES   = 2000
VIEW_PERIODS= 3.0
LOW_FREQ_FLOOR = 0.1

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED_NAMES = {"pi": np.pi}
_FREQ_SUFFIX   = {"k": 1, "M": 2, "G": 3}

_BTN_STYLE = """
    QPushButton {
        background: #2a2a2a; color: #e5e5e5;
        border: 1px solid #444; border-radius: 4px;
        padding: 2px 8px; font-size: 11px;
    }
    QPushButton:hover { background: #3a3a3a; }
    QPushButton:pressed { background: #1a1a1a; }
"""
_BOX_STYLE = """
    QLineEdit {
        background: #1a1a1a; color: #e5e5e5;
        border: 1px solid #444; border-radius: 3px;
        padding: 1px 4px; font-size: 11px;
    }
"""


def _slider_style(color):
    return f"""
    QSlider::groove:horizontal {{
        height: 4px; background: #333; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 14px; height: 14px; margin: -5px 0;
        border-radius: 7px; background: {color};
    }}
    QSlider::sub-page:horizontal {{
        background: {color}; border-radius: 2px;
    }}
"""


def _safe_eval(expr):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        raise ValueError("disallowed")
    return visit(ast.parse(expr.strip(), mode="eval"))


def _fmt(v):
    return f"{v:.3g}"


def _fmt_phase(val, phase_unit_idx):
    if PHASE_UNITS[phase_unit_idx][0] != "rad":
        return _fmt(val)
    if val == 0:
        return "0"
    pm = val / np.pi
    if abs(pm - round(pm)) < 0.01:
        n = int(round(pm))
        return "pi" if n == 1 else f"{n}pi"
    return f"{pm:.3g}pi"


class _ControlRow(QWidget):
    """label | slider | textbox | unit-button"""

    def __init__(self, label, sl_min, sl_max, sl_init, color, unit_text, parent=None):
        super().__init__(parent)
        self._min = sl_min
        self._max = sl_max
        self._steps = 2000

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
        self.slider.setStyleSheet(_slider_style(color))
        self.slider.setFixedHeight(22)
        lay.addWidget(self.slider, stretch=1)

        self.box = QLineEdit()
        self.box.setFixedWidth(72)
        self.box.setStyleSheet(_BOX_STYLE)
        lay.addWidget(self.box)

        self.unit_btn = QPushButton(unit_text)
        self.unit_btn.setFixedWidth(52)
        self.unit_btn.setStyleSheet(_BTN_STYLE)
        lay.addWidget(self.unit_btn)

    def _to_int(self, v):
        return int((v - self._min) / (self._max - self._min) * self._steps)

    def value(self):
        return self._min + self.slider.value() / self._steps * (self._max - self._min)

    def set_value(self, v):
        self.slider.setValue(self._to_int(max(self._min, min(self._max, v))))

    def set_max(self, new_max):
        self._max = new_max


class SinWavePanel(QWidget):
    """Self-contained sine-wave panel: plot + amplitude/frequency/phase controls."""

    signal_changed = pyqtSignal()

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(parent)
        self._freq_unit_idx  = 0
        self._phase_unit_idx = 0
        self._guard          = False
        self._sample_rate_hz = sample_rate_hz

        self.setStyleSheet(f"background:{PANEL_BG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)

        # title
        self._title = QLabel()
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        outer.addWidget(self._title)

        # plot
        self._pw = pg.PlotWidget(background=BG)
        self._pw.showGrid(x=True, y=True, alpha=0.25)
        self._pw.setLabel("left",   f"Amplitude ({AMP_UNIT})", color=TEXT)
        self._pw.setLabel("bottom", "Time (s)",                color=TEXT)
        self._pw.setYRange(-2.5, 2.5)
        self._pw.getAxis("left").setTextPen(TEXT)
        self._pw.getAxis("bottom").setTextPen(TEXT)
        self._pw.setMouseEnabled(x=False, y=False)
        self._pw.getPlotItem().getViewBox().setMenuEnabled(False)
        self._pw.getPlotItem().setMenuEnabled(False)
        # continuous blue line
        self._curve = self._pw.plot(pen=pg.mkPen("#3b82f6", width=1.5))
        # red sample dots + step-hold approximation
        self._sample_dots  = self._pw.plot(pen=None,
                                            symbol='o', symbolSize=6,
                                            symbolBrush=pg.mkBrush("#ef4444"),
                                            symbolPen=pg.mkPen(None))
        self._sample_curve = self._pw.plot(pen=pg.mkPen("#ef4444", width=1.5,
                                                         style=Qt.PenStyle.SolidLine))
        outer.addWidget(self._pw, stretch=1)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{EDGE};")
        outer.addWidget(sep)

        # control rows
        self._amp_row   = _ControlRow("Amplitude", 0.0,   2.0,              1.0, AMP_COLOR,   AMP_UNIT)
        self._freq_row  = _ControlRow("Frequency", 0.0, 100.0,              1.0, FREQ_COLOR,  FREQ_UNITS[0][0])
        self._phase_row = _ControlRow("Phase",     0.0, PHASE_UNITS[0][1],  0.0, PHASE_COLOR, PHASE_UNITS[0][0])

        self._amp_row.unit_btn.setEnabled(False)
        self._amp_row.unit_btn.setStyleSheet(_BTN_STYLE + "QPushButton{color:#666;border-color:#333;}")

        for row in (self._amp_row, self._freq_row, self._phase_row):
            outer.addWidget(row)

        self._amp_row.box.setText(_fmt(1.0))
        self._freq_row.box.setText(_fmt(1.0))
        self._phase_row.box.setText("0")

        # signals
        self._amp_row.slider.valueChanged.connect(self._on_amp_sl)
        self._amp_row.box.returnPressed.connect(self._on_amp_box)
        self._freq_row.slider.valueChanged.connect(self._on_freq_sl)
        self._freq_row.box.returnPressed.connect(self._on_freq_box)
        self._freq_row.unit_btn.clicked.connect(self._cycle_freq)
        self._phase_row.slider.valueChanged.connect(self._on_phase_sl)
        self._phase_row.box.returnPressed.connect(self._on_phase_box)
        self._phase_row.unit_btn.clicked.connect(self._cycle_phase)

        self._redraw()

    # ── public API ────────────────────────────────────────────────────────────
    def set_sample_rate(self, hz: float):
        self._sample_rate_hz = hz
        self._redraw()

    def get_signal(self, t_sec: np.ndarray) -> np.ndarray:
        freq_hz = self._freq_row.value() * self._freq_factor()
        phase   = self._phase_to_rad(self._phase_row.value())
        return self._amp_row.value() * np.sin(2 * np.pi * freq_hz * t_sec + phase)

    def get_sampled_signal(self, t_sec: np.ndarray):
        """Return (t_samples, y_samples) of the ADC-sampled version."""
        if self._sample_rate_hz <= 0:
            return np.array([]), np.array([])
        duration = float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else 1.0
        n_samp   = max(2, int(round(self._sample_rate_hz * duration)))
        t_s      = np.linspace(t_sec[0], t_sec[-1], n_samp, endpoint=False)
        y_s      = self.get_signal(t_s)
        return t_s, y_s

    # ── helpers ───────────────────────────────────────────────────────────────
    def _freq_factor(self): return FREQ_UNITS[self._freq_unit_idx][1]
    def _phase_half(self):  return PHASE_UNITS[self._phase_unit_idx][2]
    def _phase_to_rad(self, v): return v * np.pi / self._phase_half()
    def _fmt_ph(self, v): return _fmt_phase(v, self._phase_unit_idx)

    def _build_title(self):
        f_unit = FREQ_UNITS[self._freq_unit_idx][0]
        p_unit = PHASE_UNITS[self._phase_unit_idx][0]
        return (f"y = A * sin(2*pi*f*t + phi)     "
                f"A = {_fmt(self._amp_row.value())} V,   "
                f"f = {_fmt(self._freq_row.value())} {f_unit},   "
                f"phi = {self._fmt_ph(self._phase_row.value())} {p_unit}")

    def _redraw(self):
        freq_hz  = self._freq_row.value() * self._freq_factor()
        f_window = max(freq_hz, LOW_FREQ_FLOOR * self._freq_factor())
        dur      = VIEW_PERIODS / f_window
        t        = np.linspace(0, dur, N_SAMPLES, endpoint=False)
        y        = self._amp_row.value() * np.sin(
                       2*np.pi*freq_hz*t + self._phase_to_rad(self._phase_row.value()))
        x = t * self._freq_factor()
        self._curve.setData(x, y)
        self._pw.setXRange(0, float(x[-1]) if x[-1] > 0 else 1.0, padding=0)
        self._pw.setLabel("bottom", f"Time ({FREQ_UNITS[self._freq_unit_idx][2]})", color=TEXT)
        self._title.setText(self._build_title())

        # ADC samples on this panel's time axis
        t_s, y_s = self.get_sampled_signal(t)
        x_s = t_s * self._freq_factor()
        self._sample_dots.setData(x_s, y_s)

        # zero-order hold: step from each sample to the next
        if len(t_s) > 1:
            t_hold = np.repeat(t_s, 2)[1:]
            y_hold = np.repeat(y_s, 2)[:-1]
            self._sample_curve.setData(t_hold * self._freq_factor(), y_hold)
        else:
            self._sample_curve.setData([], [])

        self.signal_changed.emit()

    # ── amplitude ─────────────────────────────────────────────────────────────
    def _on_amp_sl(self):
        if self._guard: return
        self._guard = True
        self._amp_row.box.setText(_fmt(self._amp_row.value()))
        self._guard = False
        self._redraw()

    def _on_amp_box(self):
        try:   v = float(self._amp_row.box.text())
        except: self._amp_row.box.setText(_fmt(self._amp_row.value())); return
        self._guard = True
        self._amp_row.set_value(v)
        self._guard = False
        self._redraw()

    # ── frequency ─────────────────────────────────────────────────────────────
    def _on_freq_sl(self):
        if self._guard: return
        self._guard = True
        self._freq_row.box.setText(_fmt(self._freq_row.value()))
        self._guard = False
        self._redraw()

    def _on_freq_box(self):
        text = self._freq_row.box.text().strip()
        target = None; num = text
        if text and text[-1] in _FREQ_SUFFIX:
            target = _FREQ_SUFFIX[text[-1]]; num = text[:-1]
        try:   value = float(num)
        except: self._freq_row.box.setText(_fmt(self._freq_row.value())); return
        if target is not None and target != self._freq_unit_idx:
            freq_hz = value * FREQ_UNITS[target][1]
            self._freq_unit_idx = target
            self._freq_row.unit_btn.setText(FREQ_UNITS[target][0])
            value = freq_hz / FREQ_UNITS[target][1]
        value = max(0.0, min(100.0, value))
        self._guard = True
        self._freq_row.set_value(value)
        self._freq_row.box.setText(_fmt(value))
        self._guard = False
        self._redraw()

    def _cycle_freq(self):
        self._freq_unit_idx = (self._freq_unit_idx + 1) % len(FREQ_UNITS)
        self._freq_row.unit_btn.setText(FREQ_UNITS[self._freq_unit_idx][0])
        self._redraw()

    # ── phase ─────────────────────────────────────────────────────────────────
    def _on_phase_sl(self):
        if self._guard: return
        self._guard = True
        self._phase_row.box.setText(self._fmt_ph(self._phase_row.value()))
        self._guard = False
        self._redraw()

    def _on_phase_box(self):
        text = self._phase_row.box.text()
        try:
            if PHASE_UNITS[self._phase_unit_idx][0] == "rad":
                v = float(_safe_eval(text))
            else:
                v = float(text)
        except:
            self._phase_row.box.setText(self._fmt_ph(self._phase_row.value())); return
        pu = PHASE_UNITS[self._phase_unit_idx]
        self._guard = True
        self._phase_row.set_value(max(0.0, min(pu[1], v)))
        self._guard = False
        self._redraw()

    def _cycle_phase(self):
        cur_rad = self._phase_to_rad(self._phase_row.value())
        self._phase_unit_idx = (self._phase_unit_idx + 1) % len(PHASE_UNITS)
        pu = PHASE_UNITS[self._phase_unit_idx]
        new_val = max(0.0, min(pu[1], cur_rad * pu[2] / np.pi))
        self._phase_row.set_max(pu[1])
        self._phase_row.unit_btn.setText(pu[0])
        self._guard = True
        self._phase_row.set_value(new_val)
        self._phase_row.box.setText(self._fmt_ph(new_val))
        self._guard = False
        self._redraw()
