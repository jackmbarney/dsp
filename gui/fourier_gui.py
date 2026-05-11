"""fourier_gui.py – MainWindow: header + left wave panels + right FFT panel."""

import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QLabel, QSizePolicy, QFrame,
)

from utils.gui_utils import PANEL_BG, EDGE, TEXT, MAX_WAVES, ADD_STYLE, REM_STYLE
from classes.wave_plot import WavePlotPanel
from classes.fft_panel import FftPanel


class HeaderBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:#0d0d0d; border-bottom:1px solid {EDGE};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(0)
        lay.addStretch()

        def _block(title, title_color, eq, eq_color):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t = QLabel(title)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(f"color:{title_color}; font-size:15px; font-weight:bold;")
            col.addWidget(t)
            e = QLabel(eq)
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setStyleSheet(f"color:{eq_color}; font-size:14px; font-family:monospace;")
            col.addWidget(e)
            return col

        lay.addLayout(_block(
            "Continuous Fourier Transform", "#3b82f6",
            "X(f)  =  ∫ x(t) · e⁻ʲ²πᶠᵗ dt ,   −∞ < f < ∞", "#93c5fd"))

        lay.addSpacing(60)
        vd = QFrame(); vd.setFrameShape(QFrame.Shape.VLine)
        vd.setFixedHeight(50); vd.setStyleSheet(f"color:{EDGE};")
        lay.addWidget(vd, alignment=Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(60)

        lay.addLayout(_block(
            "Discrete Fourier Transform (DFT)", "#ef4444",
            "X[k]  =  Σⁿ₌₀ᴺ⁻¹ x[n] · e⁻ʲ²πᵏⁿ/ᴺ ,   k = 0 … N−1", "#fca5a5"))

        lay.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Transform Visualizer")
        self.resize(1600, 960)
        self.setStyleSheet(f"background:{PANEL_BG};")
        self._panels: list[WavePlotPanel] = []

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        root_lay.addWidget(HeaderBar())

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(6, 6, 6, 6)
        body_lay.setSpacing(0)

        # ── left: wave panel list ─────────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{PANEL_BG};")
        left.setFixedWidth(660)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        # toolbar
        tb = QWidget()
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(4, 4, 4, 2); tb_lay.setSpacing(6)
        tb_lay.addWidget(QLabel("Sine Waves",
                                styleSheet=f"color:{TEXT};font-size:13px;font-weight:bold;"))
        tb_lay.addStretch()
        self._count_lbl = QLabel("1 / 8", styleSheet="color:#888;font-size:11px;")
        tb_lay.addWidget(self._count_lbl)
        self._btn_add = QPushButton("+")
        self._btn_add.setFixedSize(40, 30); self._btn_add.setStyleSheet(ADD_STYLE)
        self._btn_add.clicked.connect(self._add_wave); tb_lay.addWidget(self._btn_add)
        self._btn_rem = QPushButton("−")
        self._btn_rem.setFixedSize(40, 30); self._btn_rem.setStyleSheet(REM_STYLE)
        self._btn_rem.clicked.connect(self._remove_wave); tb_lay.addWidget(self._btn_rem)
        left_lay.addWidget(tb)

        # scrollable panel container
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"QScrollArea{{border:none;background:{PANEL_BG};}}")
        self._container = QWidget()
        self._container.setStyleSheet(f"background:{PANEL_BG};")
        self._vlay = QVBoxLayout(self._container)
        self._vlay.setContentsMargins(0, 0, 0, 0)
        self._vlay.setSpacing(4)
        self._vlay.addStretch()
        self._scroll.setWidget(self._container)
        left_lay.addWidget(self._scroll, stretch=1)
        body_lay.addWidget(left)

        div = QFrame(); div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet(f"color:{EDGE};"); body_lay.addWidget(div)

        # ── right: FFT panel ──────────────────────────────────────────────────
        self._fft_panel = FftPanel()
        self._fft_panel.connect_sr_changed(self._on_sr_changed)
        body_lay.addWidget(self._fft_panel, stretch=1)

        root_lay.addWidget(body, stretch=1)
        self._add_wave()

    def _add_wave(self):
        if len(self._panels) >= MAX_WAVES: return
        p = WavePlotPanel(sample_rate_hz=self._fft_panel.sample_rate_hz())
        p.setMinimumHeight(220)
        p.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        p.signal_changed.connect(self._update_fft)
        self._vlay.insertWidget(self._vlay.count() - 1, p)
        self._panels.append(p)
        self._refresh(); self._update_fft()

    def _remove_wave(self):
        if not self._panels: return
        p = self._panels.pop()
        p.signal_changed.disconnect(self._update_fft)
        self._vlay.removeWidget(p); p.deleteLater()
        self._refresh(); self._update_fft()

    def _refresh(self):
        n = len(self._panels)
        self._count_lbl.setText(f"{n} / {MAX_WAVES}")
        self._btn_add.setEnabled(n < MAX_WAVES)
        self._btn_rem.setEnabled(n > 0)

    def _on_sr_changed(self):
        sr = self._fft_panel.sample_rate_hz()
        for p in self._panels: p.set_sample_rate(sr)
        self._update_fft()

    def _update_fft(self):
        self._fft_panel.update(self._panels)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
