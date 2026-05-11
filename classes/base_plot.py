"""base_plot.py – BasePlotPanel: shared enable/disable + signal interface."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton

from utils.gui_utils import BG, PANEL_BG, EDGE, TEXT, AMP_UNIT, BTN_STYLE


def _make_panel_plot(line_color: str):
    pw = pg.PlotWidget(background=BG)
    pw.showGrid(x=True, y=True, alpha=0.25)
    pw.setLabel("left",   f"Amplitude ({AMP_UNIT})", color=TEXT)
    pw.setLabel("bottom", "Time (s)",                color=TEXT)
    pw.setYRange(-2.5, 2.5)
    pw.getAxis("left").setTextPen(TEXT)
    pw.getAxis("bottom").setTextPen(TEXT)
    pw.setMouseEnabled(x=False, y=False)
    pw.getPlotItem().getViewBox().setMenuEnabled(False)
    pw.getPlotItem().setMenuEnabled(False)
    curve = pw.plot(pen=pg.mkPen(line_color, width=1.5))
    return pw, curve


class BasePlotPanel(QWidget):
    """
    Base class for all signal panels.
    Provides:
      - title bar with enable/disable toggle button
      - pyqtgraph plot widget
      - signal_changed pyqtSignal
      - enabled property
      - abstract interface: freq_hz(), get_signal(t), label()
    """

    signal_changed = pyqtSignal()

    LINE_COLOR = "#3b82f6"   # override in subclass

    def __init__(self, sample_rate_hz: float = 100.0, parent=None):
        super().__init__(parent)
        self._enabled         = True
        self._sample_rate_hz  = sample_rate_hz

        self.setStyleSheet(f"background:{PANEL_BG};")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(6, 4, 6, 4)
        self._outer.setSpacing(3)

        # ── title bar (label + enable/disable button) ─────────────────────────
        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(4)

        self._title_lbl = QLabel()
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        hdr_lay.addWidget(self._title_lbl, stretch=1)

        self._en_btn = QPushButton("■ Enabled")
        self._en_btn.setFixedWidth(90)
        self._en_btn.setStyleSheet(self._enabled_style(True))
        self._en_btn.clicked.connect(self._toggle_enabled)
        hdr_lay.addWidget(self._en_btn)

        self._outer.addWidget(hdr)

        # ── plot ──────────────────────────────────────────────────────────────
        self._pw, self._curve = _make_panel_plot(self.LINE_COLOR)
        self._outer.addWidget(self._pw, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{EDGE};")
        self._outer.addWidget(sep)

    # ── enable / disable ──────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    def _toggle_enabled(self):
        self._enabled = not self._enabled
        self._en_btn.setText("■ Enabled" if self._enabled else "□ Disabled")
        self._en_btn.setStyleSheet(self._enabled_style(self._enabled))
        self._pw.setVisible(self._enabled)
        self.signal_changed.emit()

    @staticmethod
    def _enabled_style(on: bool) -> str:
        if on:
            return ("QPushButton{background:#16a34a;color:white;font-size:10px;"
                    "border:none;border-radius:4px;padding:2px 6px;}"
                    "QPushButton:hover{background:#22c55e;}")
        return ("QPushButton{background:#374151;color:#9ca3af;font-size:10px;"
                "border:1px solid #4b5563;border-radius:4px;padding:2px 6px;}"
                "QPushButton:hover{background:#4b5563;}")

    # ── public interface (override in subclasses) ─────────────────────────────
    def set_sample_rate(self, hz: float):
        self._sample_rate_hz = hz
        if self._enabled:
            self._redraw()

    def freq_hz(self) -> float:
        """Return the primary frequency in Hz (for Nyquist calculation)."""
        return 0.0

    def get_signal(self, t_sec: np.ndarray) -> np.ndarray:
        """Return signal sampled at t_sec. Return zeros if disabled."""
        if not self._enabled:
            return np.zeros_like(t_sec)
        return self._compute_signal(t_sec)

    def _compute_signal(self, t_sec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def label(self) -> str:
        return ""

    def _redraw(self):
        raise NotImplementedError
