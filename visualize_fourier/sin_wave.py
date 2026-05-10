"""SinWave: one sine wave plot panel with sliders, text boxes, unit buttons."""

import ast
import operator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

# ── constants ────────────────────────────────────────────────────────────────
AMP_UNIT   = "V"
FREQ_UNITS  = [("Hz", 1.0, "s"), ("kHz", 1e3, "ms"), ("MHz", 1e6, "µs"), ("GHz", 1e9, "ns")]
PHASE_UNITS = [
    ("rad", 2 * np.pi, np.pi, "Phase (0–2π)"),
    ("deg", 360.0,     180.0, "Phase (0–360°)"),
]
AMP_COLOR   = "#22d3ee"
FREQ_COLOR  = "#fbbf24"
PHASE_COLOR = "#f472b6"
WIDGET_BG   = "black"
WIDGET_HOVER= "#1a1a1a"
WIDGET_EDGE = "white"
WIDGET_TEXT = "white"
N_SAMPLES   = 2000
VIEW_PERIODS= 3.0
LOW_FREQ_FLOOR = 0.1

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED_NAMES = {"pi": np.pi, "π": np.pi}
_FREQ_SUFFIX_MAP = {"k": 1, "M": 2, "G": 3}


def _safe_eval(expr):
    def visit(node):
        if isinstance(node, ast.Expression):           return visit(node.body)
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


def _outline(an_ax):
    an_ax.set_facecolor(WIDGET_BG)
    for sp in an_ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(WIDGET_EDGE); sp.set_linewidth(1.2)


# ── class ─────────────────────────────────────────────────────────────────────
class SinWave:
    """
    A self-contained sine-wave panel.

    Parameters
    ----------
    fig        : matplotlib Figure
    plot_rect  : [left, bottom, width, height]  – axes for the wave plot
    widget_rect: [left, bottom, width, height]  – bounding box for the 3 widget rows
    on_change  : optional callback called after every redraw
    """

    def __init__(self, fig, plot_rect, widget_rect, on_change=None):
        self.fig = fig
        self.on_change = on_change

        # ── plot axes ────────────────────────────────────────────────────────
        self.ax = fig.add_axes(plot_rect)
        self.ax.set_facecolor("black")
        self.ax.set_ylabel(f"Amplitude ({AMP_UNIT})")
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.grid(True)
        (self.line,) = self.ax.plot([], [], lw=1.5)

        # ── internal state ───────────────────────────────────────────────────
        self._freq_unit_idx  = 0
        self._phase_unit_idx = 0

        # ── widget rows ───────────────────────────────────────────────────────
        L, B, W, H = widget_rect          # bounding box
        row_h   = 0.028
        row_gap = 0.012
        rows = [B + 2*(row_h + row_gap), B + (row_h + row_gap), B]  # amp, freq, phase

        def make_row(y):
            sl_ax  = fig.add_axes([L,          y, W*0.50, row_h])
            box_ax = fig.add_axes([L+W*0.52,   y, W*0.13, row_h])
            btn_ax = fig.add_axes([L+W*0.67,   y, W*0.15, row_h])
            return sl_ax, box_ax, btn_ax

        amp_sl_ax,   amp_box_ax,   amp_unit_ax   = make_row(rows[0])
        freq_sl_ax,  freq_box_ax,  freq_unit_ax  = make_row(rows[1])
        phase_sl_ax, phase_box_ax, phase_unit_ax = make_row(rows[2])

        # sliders
        self.sl_amp = Slider(amp_sl_ax,   "Amplitude", 0.0,   2.0,   valinit=1.0,
                             color=AMP_COLOR,   track_color=WIDGET_BG, initcolor="none")
        self.sl_freq= Slider(freq_sl_ax,  "Frequency", 0.0, 100.0,   valinit=1.0,
                             color=FREQ_COLOR,  track_color=WIDGET_BG, initcolor="none")
        self.sl_phase=Slider(phase_sl_ax, PHASE_UNITS[0][3], 0.0, PHASE_UNITS[0][1],
                             valinit=0.0,
                             color=PHASE_COLOR, track_color=WIDGET_BG, initcolor="none")
        for s in (self.sl_amp, self.sl_freq, self.sl_phase):
            s.valtext.set_visible(False); _outline(s.ax)

        # text boxes
        self.box_amp  = TextBox(amp_box_ax,   "", initial=_fmt(1.0), color=WIDGET_BG, hovercolor=WIDGET_HOVER)
        self.box_freq = TextBox(freq_box_ax,  "", initial=_fmt(1.0), color=WIDGET_BG, hovercolor=WIDGET_HOVER)
        self.box_phase= TextBox(phase_box_ax, "", initial="0",        color=WIDGET_BG, hovercolor=WIDGET_HOVER)
        for b in (self.box_amp, self.box_freq, self.box_phase):
            for sp in b.ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(WIDGET_EDGE); sp.set_linewidth(1.2)
            b.text_disp.set_color(WIDGET_TEXT); b.cursor.set_color(WIDGET_TEXT)

        # unit labels / buttons
        amp_unit_ax.axis("off")
        amp_unit_ax.text(0.05, 0.5, AMP_UNIT, ha="left", va="center",
                         transform=amp_unit_ax.transAxes, color=WIDGET_TEXT)

        self.btn_freq  = Button(freq_unit_ax,  FREQ_UNITS[0][0],  color="#444", hovercolor="#666")
        self.btn_phase = Button(phase_unit_ax, PHASE_UNITS[0][0], color="#444", hovercolor="#666")

        # ── wire up ───────────────────────────────────────────────────────────
        self._link_amp()
        self._link_phase()
        self._link_freq()
        self.btn_freq.on_clicked(self._cycle_freq_unit)
        self.btn_phase.on_clicked(self._cycle_phase_unit)

        self._redraw()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _freq_factor(self):   return FREQ_UNITS[self._freq_unit_idx][1]
    def _time_label(self):    return f"Time ({FREQ_UNITS[self._freq_unit_idx][2]})"
    def _phase_to_rad(self, v):
        return v * np.pi / PHASE_UNITS[self._phase_unit_idx][2]

    def _fmt_phase(self, val):
        if PHASE_UNITS[self._phase_unit_idx][0] != "rad":
            return _fmt(val)
        if val == 0: return "0"
        pm = val / np.pi
        if abs(pm - round(pm)) < 0.01:
            n = int(round(pm))
            return "π" if n == 1 else f"{n}π"
        return f"{pm:.3g}π"

    def _parse_phase(self, text):
        if PHASE_UNITS[self._phase_unit_idx][0] == "rad":
            return float(_safe_eval(text))
        return float(text)

    def _build_title(self):
        f_unit = FREQ_UNITS[self._freq_unit_idx][0]
        p_unit = PHASE_UNITS[self._phase_unit_idx][0]
        phi_str = self._fmt_phase(self.sl_phase.val)
        return (
            r"$y = A\sin(2\pi f\,t + \phi)$"
            f"        $A={_fmt(self.sl_amp.val)}$ V,"
            f"   $f={_fmt(self.sl_freq.val)}$ {f_unit},"
            f"   $\\phi={phi_str}$ {p_unit}"
        )

    def _redraw(self):
        freq_hz  = self.sl_freq.val * self._freq_factor()
        f_window = max(freq_hz, LOW_FREQ_FLOOR * self._freq_factor())
        dur      = VIEW_PERIODS / f_window
        t        = np.linspace(0, dur, N_SAMPLES, endpoint=False)
        y        = self.sl_amp.val * np.sin(2*np.pi*freq_hz*t + self._phase_to_rad(self.sl_phase.val))
        x        = t * self._freq_factor()
        self.line.set_data(x, y)
        self.ax.set_xlim(0, x[-1] if x[-1] > 0 else 1)
        self.ax.set_xlabel(self._time_label())
        self.ax.set_title(self._build_title(), fontsize=9)
        self.fig.canvas.draw_idle()
        if self.on_change:
            self.on_change()

    # ── wiring ────────────────────────────────────────────────────────────────
    def _link_amp(self):
        def on_sl(_):
            self.box_amp.set_val(_fmt(self.sl_amp.val)); self._redraw()
        def on_box(text):
            try:   v = float(text)
            except: self.box_amp.set_val(_fmt(self.sl_amp.val)); return
            self.sl_amp.set_val(max(self.sl_amp.valmin, min(self.sl_amp.valmax, v)))
        self.sl_amp.on_changed(on_sl); self.box_amp.on_submit(on_box)

    def _link_phase(self):
        def on_sl(_):
            self.box_phase.set_val(self._fmt_phase(self.sl_phase.val)); self._redraw()
        def on_box(text):
            try:   v = self._parse_phase(text)
            except: self.box_phase.set_val(self._fmt_phase(self.sl_phase.val)); return
            self.sl_phase.set_val(max(self.sl_phase.valmin, min(self.sl_phase.valmax, v)))
        self.sl_phase.on_changed(on_sl); self.box_phase.on_submit(on_box)

    def _link_freq(self):
        def on_sl(_):
            self.box_freq.set_val(_fmt(self.sl_freq.val)); self._redraw()

        def on_box(text):
            text = text.strip(); target = None; num = text
            if text and text[-1] in _FREQ_SUFFIX_MAP:
                target = _FREQ_SUFFIX_MAP[text[-1]]; num = text[:-1]
            try:   value = float(num)
            except: self.box_freq.set_val(_fmt(self.sl_freq.val)); return
            if target is not None and target != self._freq_unit_idx:
                new_factor = FREQ_UNITS[target][1]
                freq_hz    = value * new_factor
                self._freq_unit_idx = target
                self.btn_freq.label.set_text(FREQ_UNITS[target][0])
                value = freq_hz / new_factor
            value = max(self.sl_freq.valmin, min(self.sl_freq.valmax, value))
            self.sl_freq.set_val(value); self.box_freq.set_val(_fmt(value)); self._redraw()

        self.sl_freq.on_changed(on_sl); self.box_freq.on_submit(on_box)

    def _cycle_freq_unit(self, _):
        self._freq_unit_idx = (self._freq_unit_idx + 1) % len(FREQ_UNITS)
        self.btn_freq.label.set_text(FREQ_UNITS[self._freq_unit_idx][0])
        self._redraw()

    def _cycle_phase_unit(self, _):
        cur_rad = self._phase_to_rad(self.sl_phase.val)
        self._phase_unit_idx = (self._phase_unit_idx + 1) % len(PHASE_UNITS)
        pu = PHASE_UNITS[self._phase_unit_idx]
        new_val = max(0.0, min(pu[1], cur_rad * pu[2] / np.pi))
        self.sl_phase.valmax = pu[1]
        self.sl_phase.ax.set_xlim(0.0, pu[1])
        self.sl_phase.label.set_text(pu[3])
        self.sl_phase.set_val(new_val)
        self.btn_phase.label.set_text(pu[0])
