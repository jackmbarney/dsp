"""visualize_fourier.py – left column of sine panels, right column with mixed + sampled signals."""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QLabel, QSizePolicy, QSplitter, QFrame,
    QSlider, QLineEdit,
)

from sin_wave import (SinWavePanel, PANEL_BG, BG, TEXT, EDGE,
                      N_SAMPLES, FREQ_UNITS, _slider_style, _BTN_STYLE, _BOX_STYLE, _fmt)

MAX_WAVES    = 8
MIX_SAMPLES  = N_SAMPLES
T_WINDOW     = 1.0          # seconds shown in right-side plots
SR_MIN       = 2.0
SR_MAX       = 1000.0
SR_INIT      = 100.0
SR_STEPS     = 2000
SR_UNITS     = FREQ_UNITS   # reuse same Hz/kHz/MHz/GHz list
SR_COLOR     = "#f472b6"    # pink
_FREQ_SUFFIX = {"k": 1, "M": 2, "G": 3}

_ADD_STYLE = """
    QPushButton {
        background: #16a34a; color: white; font-size: 18px; font-weight: bold;
        border: none; border-radius: 5px; padding: 2px 12px;
    }
    QPushButton:hover  { background: #22c55e; }
    QPushButton:pressed{ background: #15803d; }
"""
_REM_STYLE = """
    QPushButton {
        background: #b91c1c; color: white; font-size: 18px; font-weight: bold;
        border: none; border-radius: 5px; padding: 2px 12px;
    }
    QPushButton:hover  { background: #ef4444; }
    QPushButton:pressed{ background: #991b1b; }
"""


def _make_plot(title, line_color, line_width=2):
    pw = pg.PlotWidget(background=BG)
    pw.setTitle(title, color=TEXT, size="11pt")
    pw.showGrid(x=True, y=True, alpha=0.25)
    pw.setLabel("left",   "Amplitude (V)", color=TEXT)
    pw.setLabel("bottom", "Time (s)",       color=TEXT)
    pw.getAxis("left").setTextPen(TEXT)
    pw.getAxis("bottom").setTextPen(TEXT)
    pw.setMouseEnabled(x=False, y=False)
    pw.getPlotItem().getViewBox().setMenuEnabled(False)
    pw.getPlotItem().setMenuEnabled(False)
    curve = pw.plot(pen=pg.mkPen(line_color, width=line_width))
    return pw, curve


class SamplingRateRow(QWidget):
    """Pink sampling-rate control: label | slider | textbox | unit-button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._unit_idx = 0
        self._min   = SR_MIN
        self._max   = SR_MAX
        self._steps = SR_STEPS
        self._guard = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        lbl = QLabel("Sample Rate")
        lbl.setFixedWidth(86)
        lbl.setStyleSheet(f"color:{TEXT}; font-size:11px; font-weight:bold;")
        lay.addWidget(lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._to_int(SR_INIT))
        self.slider.setStyleSheet(_slider_style(SR_COLOR))
        self.slider.setFixedHeight(22)
        lay.addWidget(self.slider, stretch=1)

        self.box = QLineEdit()
        self.box.setFixedWidth(72)
        self.box.setStyleSheet(_BOX_STYLE)
        self.box.setText(_fmt(SR_INIT))
        lay.addWidget(self.box)

        self.unit_btn = QPushButton(SR_UNITS[0][0])
        self.unit_btn.setFixedWidth(52)
        self.unit_btn.setStyleSheet(_BTN_STYLE)
        lay.addWidget(self.unit_btn)

        self.slider.valueChanged.connect(self._on_slider)
        self.box.returnPressed.connect(self._on_box)
        self.unit_btn.clicked.connect(self._cycle_unit)

    def _to_int(self, v):
        return int((v - self._min) / (self._max - self._min) * self._steps)

    def value_hz(self):
        """Current sample rate in Hz."""
        return (self._min + self.slider.value() / self._steps * (self._max - self._min)) \
               * SR_UNITS[self._unit_idx][1]

    def _slider_val(self):
        return self._min + self.slider.value() / self._steps * (self._max - self._min)

    def _on_slider(self):
        if self._guard: return
        self._guard = True
        self.box.setText(_fmt(self._slider_val()))
        self._guard = False
        self.valueChanged()

    def _on_box(self):
        text = self.box.text().strip()
        target = None; num = text
        if text and text[-1] in _FREQ_SUFFIX:
            target = _FREQ_SUFFIX[text[-1]]; num = text[:-1]
        try:   value = float(num)
        except: self.box.setText(_fmt(self._slider_val())); return
        if target is not None and target != self._unit_idx:
            hz = value * SR_UNITS[target][1]
            self._unit_idx = target
            self.unit_btn.setText(SR_UNITS[target][0])
            value = hz / SR_UNITS[target][1]
        value = max(self._min, min(self._max, value))
        self._guard = True
        self.slider.setValue(self._to_int(value))
        self.box.setText(_fmt(value))
        self._guard = False
        self.valueChanged()

    def _cycle_unit(self):
        self._unit_idx = (self._unit_idx + 1) % len(SR_UNITS)
        self.unit_btn.setText(SR_UNITS[self._unit_idx][0])
        self.valueChanged()

    # override to connect
    def valueChanged(self):
        pass   # replaced by caller


class RightPanel(QWidget):
    """Right side: analog mixed plot (blue), sampled mixed plot (red), sampling rate slider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # analog mixed – blue
        self._pw_analog, self._curve_analog = _make_plot(
            "Actual mixed signal", "#3b82f6", line_width=2)
        lay.addWidget(self._pw_analog, stretch=1)

        # sampled mixed – red
        self._pw_sampled, self._curve_sampled = _make_plot(
            "Sampled mixed signal (ADC)", "#ef4444", line_width=1.5)
        # also add dots on sampled plot
        self._dots_sampled = self._pw_sampled.plot(
            pen=None, symbol='o', symbolSize=5,
            symbolBrush=pg.mkBrush("#ef4444"),
            symbolPen=pg.mkPen(None))
        lay.addWidget(self._pw_sampled, stretch=1)

        # sampling rate row
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{EDGE};")
        lay.addWidget(sep)

        self._sr_row = SamplingRateRow()
        lay.addWidget(self._sr_row)

    def sample_rate_hz(self):
        return self._sr_row.value_hz()

    def connect_sr_changed(self, cb):
        self._sr_row.valueChanged = cb

    def update(self, panels: list):
        if not panels:
            for c in (self._curve_analog, self._curve_sampled, self._dots_sampled):
                c.setData([], [])
            return

        sr_hz = self._sr_row.value_hz()
        t_cont = np.linspace(0, T_WINDOW, MIX_SAMPLES, endpoint=False)

        # ── analog: multiply all continuous signals ───────────────────────────
        mixed = np.ones(MIX_SAMPLES)
        for p in panels:
            mixed *= p.get_signal(t_cont)
        self._curve_analog.setData(t_cont, mixed)
        self._pw_analog.setXRange(0, T_WINDOW, padding=0)
        y_max = float(np.max(np.abs(mixed))) * 1.2 or 1.0
        self._pw_analog.setYRange(-y_max, y_max)

        # ── sampled: multiply sampled signals, zero-order hold ────────────────
        n_samp  = max(2, int(round(sr_hz * T_WINDOW)))
        t_samp  = np.linspace(0, T_WINDOW, n_samp, endpoint=False)
        mixed_s = np.ones(n_samp)
        for p in panels:
            mixed_s *= p.get_signal(t_samp)

        # dots at sample points
        self._dots_sampled.setData(t_samp, mixed_s)

        # zero-order hold reconstruction
        if len(t_samp) > 1:
            t_hold = np.repeat(t_samp, 2)[1:]
            y_hold = np.repeat(mixed_s, 2)[:-1]
            self._curve_sampled.setData(t_hold, y_hold)
        else:
            self._curve_sampled.setData([], [])

        self._pw_sampled.setXRange(0, T_WINDOW, padding=0)
        y_max_s = float(np.max(np.abs(mixed_s))) * 1.2 or 1.0
        self._pw_sampled.setYRange(-y_max_s, y_max_s)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sine Wave Visualizer")
        self.resize(1500, 900)
        self.setStyleSheet(f"background:{PANEL_BG};")

        self._panels: list[SinWavePanel] = []

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QHBoxLayout(root)
        root_lay.setContentsMargins(6, 6, 6, 6)
        root_lay.setSpacing(0)

        # ── left column ───────────────────────────────────────────────────────
        left_outer = QWidget()
        left_outer.setStyleSheet(f"background:{PANEL_BG};")
        left_outer.setFixedWidth(660)
        left_lay = QVBoxLayout(left_outer)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        # +/- toolbar above panel list
        tb = QWidget()
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(4, 4, 4, 2)
        tb_lay.setSpacing(6)

        title_lbl = QLabel("Sine Waves")
        title_lbl.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:bold;")
        tb_lay.addWidget(title_lbl)
        tb_lay.addStretch()

        self._count_lbl = QLabel("1 / 8")
        self._count_lbl.setStyleSheet("color:#888; font-size:11px;")
        tb_lay.addWidget(self._count_lbl)

        self._btn_add = QPushButton("+")
        self._btn_add.setFixedSize(40, 30)
        self._btn_add.setStyleSheet(_ADD_STYLE)
        self._btn_add.clicked.connect(self._add_wave)
        tb_lay.addWidget(self._btn_add)

        self._btn_rem = QPushButton("−")
        self._btn_rem.setFixedSize(40, 30)
        self._btn_rem.setStyleSheet(_REM_STYLE)
        self._btn_rem.clicked.connect(self._remove_wave)
        tb_lay.addWidget(self._btn_rem)

        left_lay.addWidget(tb)

        # scrollable panel list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"QScrollArea{{border:none; background:{PANEL_BG};}}")
        left_lay.addWidget(self._scroll, stretch=1)

        self._container = QWidget()
        self._container.setStyleSheet(f"background:{PANEL_BG};")
        self._vlay = QVBoxLayout(self._container)
        self._vlay.setContentsMargins(0, 0, 0, 0)
        self._vlay.setSpacing(4)
        self._vlay.addStretch()
        self._scroll.setWidget(self._container)

        root_lay.addWidget(left_outer)

        # vertical divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet(f"color:{EDGE};")
        root_lay.addWidget(div)

        # ── right column ──────────────────────────────────────────────────────
        self._right = RightPanel()
        self._right.connect_sr_changed(self._on_sr_changed)
        root_lay.addWidget(self._right, stretch=1)

        self._add_wave()

    # ── wave management ───────────────────────────────────────────────────────
    def _current_sr_hz(self):
        return self._right.sample_rate_hz()

    def _add_wave(self):
        if len(self._panels) >= MAX_WAVES:
            return
        panel = SinWavePanel(sample_rate_hz=self._current_sr_hz())
        panel.setMinimumHeight(220)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.signal_changed.connect(self._update_right)
        self._vlay.insertWidget(self._vlay.count() - 1, panel)
        self._panels.append(panel)
        self._refresh_buttons()
        self._update_right()

    def _remove_wave(self):
        if not self._panels:
            return
        panel = self._panels.pop()
        panel.signal_changed.disconnect(self._update_right)
        self._vlay.removeWidget(panel)
        panel.deleteLater()
        self._refresh_buttons()
        self._update_right()

    def _refresh_buttons(self):
        n = len(self._panels)
        self._count_lbl.setText(f"{n} / {MAX_WAVES}")
        self._btn_add.setEnabled(n < MAX_WAVES)
        self._btn_rem.setEnabled(n > 0)

    def _on_sr_changed(self):
        sr = self._current_sr_hz()
        for p in self._panels:
            p.set_sample_rate(sr)
        self._update_right()

    def _update_right(self):
        self._right.update(self._panels)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
