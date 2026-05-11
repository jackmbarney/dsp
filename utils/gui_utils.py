"""gui_utils.py – shared Qt/pyqtgraph helpers and theme constants."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

# ── theme ─────────────────────────────────────────────────────────────────────
AMP_UNIT    = "V"
FREQ_UNITS  = [("Hz", 1.0, "s"), ("kHz", 1e3, "ms"), ("MHz", 1e6, "µs"), ("GHz", 1e9, "ns")]
PHASE_UNITS = [
    ("rad", 2 * np.pi, np.pi,  "Phase (0-2pi)"),
    ("deg", 360.0,     180.0,  "Phase (0-360 deg)"),
]

AMP_COLOR   = "#22d3ee"
FREQ_COLOR  = "#fbbf24"
PHASE_COLOR = "#f472b6"
SR_COLOR    = "#f472b6"
BG          = "#0a0a0a"
PANEL_BG    = "#111111"
EDGE        = "#333333"
TEXT        = "#e5e5e5"

N_SAMPLES      = 2000
VIEW_PERIODS   = 3.0
LOW_FREQ_FLOOR = 0.1
MIX_SAMPLES    = N_SAMPLES
T_WINDOW       = 1.0
MAX_WAVES      = 8

# ── stylesheet helpers ────────────────────────────────────────────────────────
BTN_STYLE = """
    QPushButton {
        background: #2a2a2a; color: #e5e5e5;
        border: 1px solid #444; border-radius: 4px;
        padding: 2px 8px; font-size: 11px;
    }
    QPushButton:hover  { background: #3a3a3a; }
    QPushButton:pressed{ background: #1a1a1a; }
"""
BOX_STYLE = """
    QLineEdit {
        background: #1a1a1a; color: #e5e5e5;
        border: 1px solid #444; border-radius: 3px;
        padding: 1px 4px; font-size: 11px;
    }
"""
ADD_STYLE = """QPushButton{background:#16a34a;color:white;font-size:18px;font-weight:bold;
    border:none;border-radius:5px;padding:2px 12px;}
QPushButton:hover{background:#22c55e;}QPushButton:pressed{background:#15803d;}"""
REM_STYLE = """QPushButton{background:#b91c1c;color:white;font-size:18px;font-weight:bold;
    border:none;border-radius:5px;padding:2px 12px;}
QPushButton:hover{background:#ef4444;}QPushButton:pressed{background:#991b1b;}"""


def slider_style(color: str) -> str:
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


# ── pyqtgraph helpers ─────────────────────────────────────────────────────────
def lock_plot(pw: pg.PlotWidget) -> None:
    pw.setMouseEnabled(x=False, y=False)
    pw.getPlotItem().getViewBox().setMenuEnabled(False)
    pw.getPlotItem().setMenuEnabled(False)


def make_signal_plot(title: str, color: str,
                     y_label: str = "Amplitude (V)",
                     x_label: str = "Time (s)"):
    pw = pg.PlotWidget(background=BG)
    pw.setTitle(title, color=TEXT, size="10pt")
    pw.showGrid(x=True, y=True, alpha=0.25)
    pw.setLabel("left",   y_label, color=TEXT)
    pw.setLabel("bottom", x_label, color=TEXT)
    pw.getAxis("left").setTextPen(TEXT)
    pw.getAxis("bottom").setTextPen(TEXT)
    lock_plot(pw)
    return pw, pw.plot(pen=pg.mkPen(color, width=1.8))


class HzAxis(pg.AxisItem):
    """AxisItem that auto-formats ticks as Hz / kHz / MHz / GHz."""
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            av = abs(v)
            if   av == 0:    out.append("0")
            elif av >= 1e9:  out.append(f"{v/1e9:.3g}G")
            elif av >= 1e6:  out.append(f"{v/1e6:.3g}M")
            elif av >= 1e3:  out.append(f"{v/1e3:.3g}k")
            else:            out.append(f"{v:.3g}")
        return out


def make_fft_plot(title: str, color: str):
    x_axis = HzAxis(orientation="bottom")
    x_axis.setTextPen(TEXT)
    pw = pg.PlotWidget(background=BG, axisItems={"bottom": x_axis})
    pw.setTitle(title, color=color, size="9pt")
    pw.showGrid(x=True, y=True, alpha=0.25)
    pw.setLabel("left",   "Power (dBm)", color=TEXT)
    pw.setLabel("bottom", "Frequency",   color=TEXT)
    pw.getAxis("left").setTextPen(TEXT)
    lock_plot(pw)
    return pw, pw.plot(pen=pg.mkPen(color, width=1.5))


def add_crosshair(pw: pg.PlotWidget, color: str):
    """
    Attach a snapping crosshair to a PlotWidget.
    Returns (proxy, state) — caller must keep proxy alive and set state["curve"].
    """
    vline = pg.InfiniteLine(angle=90, movable=False,
                            pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
    hline = pg.InfiniteLine(angle=0,  movable=False,
                            pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
    pw.addItem(vline, ignoreBounds=True)
    pw.addItem(hline, ignoreBounds=True)
    label = pg.TextItem("", anchor=(0, 1), color=color)
    label.setZValue(10)
    pw.addItem(label, ignoreBounds=True)
    state = {"curve": None}

    def _moved(evt):
        pos = evt[0]
        if not pw.sceneBoundingRect().contains(pos):
            label.setText(""); return
        mp    = pw.getPlotItem().vb.mapSceneToView(pos)
        curve = state["curve"]
        if curve is None: label.setText(""); return
        xd, yd = curve.getData()
        if xd is None or len(xd) == 0: label.setText(""); return
        idx = int(np.argmin(np.abs(xd - mp.x())))
        sx, sy = float(xd[idx]), float(yd[idx])
        vline.setPos(sx); hline.setPos(sy); label.setPos(sx, sy)
        ax = abs(sx)
        if   ax >= 1e9: disp = f"{sx/1e9:.4g} GHz"
        elif ax >= 1e6: disp = f"{sx/1e6:.4g} MHz"
        elif ax >= 1e3: disp = f"{sx/1e3:.4g} kHz"
        else:           disp = f"{sx:.4g} Hz"
        label.setText(f"  {disp}  {sy:.2f} dBm")

    proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60, slot=_moved)
    return proxy, state
