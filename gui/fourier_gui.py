"""fourier_gui.py – MainWindow with preset library and all panel types."""

import sys
from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QLabel, QSizePolicy, QFrame, QMenu,
)
from PyQt6.QtGui import QAction

from utils.gui_utils        import PANEL_BG, EDGE, TEXT, MAX_WAVES, ADD_STYLE, REM_STYLE
from classes.wave_plot      import WavePlotPanel
from classes.noise_plot     import NoisePlotPanel
from classes.wave_presets   import SquareWavePlot, SawtoothWavePlot, TriangleWavePlot
from classes.fft_panel      import FftPanel
from classes.base_plot      import BasePlotPanel


_PRESET_BTN = """QPushButton{background:#1e293b;color:#94a3b8;font-size:10px;
    border:1px solid #334155;border-radius:4px;padding:2px 10px;}
QPushButton:hover{background:#334155;color:#e2e8f0;}
QPushButton::menu-indicator{width:0;}"""


class HeaderBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:#0d0d0d; border-bottom:1px solid {EDGE};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(0)
        lay.addStretch()

        def _block(title, tc, eq, ec):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t = QLabel(title); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(f"color:{tc}; font-size:15px; font-weight:bold;")
            col.addWidget(t)
            e = QLabel(eq); e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setStyleSheet(f"color:{ec}; font-size:14px; font-family:monospace;")
            col.addWidget(e)
            return col

        lay.addLayout(_block("Continuous Fourier Transform", "#3b82f6",
                             "X(f)  =  ∫ x(t) · e⁻ʲ²πᶠᵗ dt ,   −∞ < f < ∞", "#93c5fd"))
        lay.addSpacing(60)
        vd = QFrame(); vd.setFrameShape(QFrame.Shape.VLine)
        vd.setFixedHeight(50); vd.setStyleSheet(f"color:{EDGE};")
        lay.addWidget(vd, alignment=Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(60)
        lay.addLayout(_block("Discrete Fourier Transform (DFT)", "#ef4444",
                             "X[k]  =  Σⁿ₌₀ᴺ⁻¹ x[n] · e⁻ʲ²πᵏⁿ/ᴺ ,   k = 0 … N−1", "#fca5a5"))
        lay.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Transform Visualizer")
        self.resize(1600, 960)
        self.setStyleSheet(f"background:{PANEL_BG};")
        self._panels: list[BasePlotPanel] = []

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

        # ── left column ───────────────────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{PANEL_BG};")
        left.setFixedWidth(660)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        # toolbar: + dropdown | − | count
        tb = QWidget()
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(4, 4, 4, 2); tb_lay.setSpacing(6)
        tb_lay.addWidget(QLabel("Signals",
                                styleSheet=f"color:{TEXT};font-size:13px;font-weight:bold;"))
        tb_lay.addStretch()

        self._count_lbl = QLabel("1 / 8", styleSheet="color:#888;font-size:11px;")
        tb_lay.addWidget(self._count_lbl)

        # + button with dropdown menu for signal type
        self._btn_add = QPushButton("+ Add")
        self._btn_add.setFixedHeight(30)
        self._btn_add.setStyleSheet(ADD_STYLE)
        add_menu = QMenu(self._btn_add)
        add_menu.setStyleSheet(
            "QMenu{background:#1e1e1e;color:#e5e5e5;border:1px solid #444;}"
            "QMenu::item:selected{background:#3b82f6;}")
        for label, fn in [
            ("∿  Sine Wave",     lambda: self._add_panel(WavePlotPanel)),
            ("⊓  Square Wave",   lambda: self._add_panel(SquareWavePlot)),
            ("⊿  Sawtooth Wave", lambda: self._add_panel(SawtoothWavePlot)),
            ("△  Triangle Wave", lambda: self._add_panel(TriangleWavePlot)),
            ("≋  Noise",         lambda: self._add_panel(NoisePlotPanel)),
        ]:
            act = QAction(label, self)
            act.triggered.connect(fn)
            add_menu.addAction(act)
        self._btn_add.setMenu(add_menu)
        tb_lay.addWidget(self._btn_add)

        self._btn_rem = QPushButton("−")
        self._btn_rem.setFixedSize(40, 30); self._btn_rem.setStyleSheet(REM_STYLE)
        self._btn_rem.clicked.connect(self._remove_panel)
        tb_lay.addWidget(self._btn_rem)
        left_lay.addWidget(tb)

        # scrollable panel list
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
        self._fft = FftPanel()
        self._fft.connect_sr_changed(self._on_sr_changed)
        body_lay.addWidget(self._fft, stretch=1)

        root_lay.addWidget(body, stretch=1)

        # start with one sine wave
        self._add_panel(WavePlotPanel)

    # ── panel management ──────────────────────────────────────────────────────
    def _add_panel(self, cls):
        if len(self._panels) >= MAX_WAVES: return
        p = cls(sample_rate_hz=self._fft.sample_rate_hz())
        p.setMinimumHeight(220)
        p.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        p.signal_changed.connect(self._update_fft)
        self._vlay.insertWidget(self._vlay.count() - 1, p)
        self._panels.append(p)
        self._refresh(); self._update_fft()

    def _remove_panel(self):
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
        sr = self._fft.sample_rate_hz()
        for p in self._panels: p.set_sample_rate(sr)
        self._update_fft()

    def _update_fft(self):
        self._fft.update(self._panels)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
