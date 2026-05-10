"""Interactive sine wave visualizer with sliders, text boxes, and unit selectors."""

import ast
import operator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

plt.style.use("dark_background")

AMP_UNIT = "V"
FREQ_UNITS = [("Hz", 1.0, "s"), ("kHz", 1e3, "ms"), ("MHz", 1e6, "µs"), ("GHz", 1e9, "ns")]
PHASE_UNITS = [
    ("rad", 2 * np.pi, np.pi, "Phase (0–2π)"),
    ("deg", 360.0, 180.0, "Phase (0–360°)"),
]

AMP_COLOR = "#22d3ee"
FREQ_COLOR = "#fbbf24"
PHASE_COLOR = "#f472b6"
WIDGET_BG = "black"
WIDGET_HOVER = "#1a1a1a"
WIDGET_EDGE = "white"
WIDGET_TEXT = "white"

INIT_AMPLITUDE = 1.0
INIT_FREQ = 1.0
INIT_PHASE = 0.0
N_SAMPLES = 2000
VIEW_PERIODS = 3.0
LOW_FREQ_FLOOR = 0.1

freq_unit_idx = [0]
phase_unit_idx = [0]


_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED_NAMES = {"pi": np.pi, "π": np.pi}


def safe_eval(expr):
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
        raise ValueError("disallowed expression")

    return visit(ast.parse(expr.strip(), mode="eval"))


def fmt(value):
    return f"{value:.3g}"


def fmt_phase(val):
    if PHASE_UNITS[phase_unit_idx[0]][0] != "rad":
        return fmt(val)
    if val == 0:
        return "0"
    pi_mult = val / np.pi
    if abs(pi_mult - round(pi_mult)) < 0.01:
        n = int(round(pi_mult))
        return "π" if n == 1 else f"{n}π"
    return f"{pi_mult:.3g}π"


def parse_phase(text):
    if PHASE_UNITS[phase_unit_idx[0]][0] == "rad":
        return float(safe_eval(text))
    return float(text)


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.30, right=0.95)
ax.set_ylabel(f"Amplitude ({AMP_UNIT})")
ax.set_ylim(-2.5, 2.5)
ax.grid(True)
(line,) = ax.plot([], [])


def row(y):
    return (
        plt.axes([0.13, y, 0.40, 0.03]),
        plt.axes([0.56, y, 0.08, 0.03]),
        plt.axes([0.66, y, 0.10, 0.03]),
    )


amp_slider_ax, amp_box_ax, amp_unit_ax = row(0.18)
freq_slider_ax, freq_box_ax, freq_unit_ax = row(0.11)
phase_slider_ax, phase_box_ax, phase_unit_ax = row(0.04)

slider_amp = Slider(
    amp_slider_ax, "Amplitude", 0.0, 2.0,
    valinit=INIT_AMPLITUDE, color=AMP_COLOR, track_color=WIDGET_BG, initcolor="none",
)
slider_freq = Slider(
    freq_slider_ax, "Frequency", 0.0, 100.0,
    valinit=INIT_FREQ, color=FREQ_COLOR, track_color=WIDGET_BG, initcolor="none",
)
slider_phase = Slider(
    phase_slider_ax, PHASE_UNITS[0][3], 0.0, PHASE_UNITS[0][1],
    valinit=INIT_PHASE, color=PHASE_COLOR, track_color=WIDGET_BG, initcolor="none",
)


def outline_axes(an_ax, edge=WIDGET_EDGE, lw=1.2):
    an_ax.set_facecolor(WIDGET_BG)
    for spine in an_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(edge)
        spine.set_linewidth(lw)


for s in (slider_amp, slider_freq, slider_phase):
    s.valtext.set_visible(False)
    outline_axes(s.ax)


box_amp = TextBox(amp_box_ax, "", initial=fmt(INIT_AMPLITUDE),
                  color=WIDGET_BG, hovercolor=WIDGET_HOVER)
box_freq = TextBox(freq_box_ax, "", initial=fmt(INIT_FREQ),
                   color=WIDGET_BG, hovercolor=WIDGET_HOVER)
box_phase = TextBox(phase_box_ax, "", initial=fmt_phase(INIT_PHASE),
                    color=WIDGET_BG, hovercolor=WIDGET_HOVER)

for b in (box_amp, box_freq, box_phase):
    for spine in b.ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(WIDGET_EDGE)
        spine.set_linewidth(1.2)
    b.text_disp.set_color(WIDGET_TEXT)
    b.cursor.set_color(WIDGET_TEXT)

amp_unit_ax.axis("off")
amp_unit_ax.text(0.05, 0.5, AMP_UNIT, ha="left", va="center", transform=amp_unit_ax.transAxes)

button_freq = Button(freq_unit_ax, FREQ_UNITS[0][0], color="#444", hovercolor="#666")
button_phase = Button(phase_unit_ax, PHASE_UNITS[0][0], color="#444", hovercolor="#666")


def freq_factor():
    return FREQ_UNITS[freq_unit_idx[0]][1]


def time_label():
    return f"Time ({FREQ_UNITS[freq_unit_idx[0]][2]})"


def phase_to_rad(phase_val):
    half_turn = PHASE_UNITS[phase_unit_idx[0]][2]
    return phase_val * np.pi / half_turn


def build_title():
    A = slider_amp.val
    f_val = slider_freq.val
    f_unit = FREQ_UNITS[freq_unit_idx[0]][0]
    phi_val = slider_phase.val
    phi_str = fmt_phase(phi_val)
    p_unit = PHASE_UNITS[phase_unit_idx[0]][0]
    return (
        r"$y = A\sin(2\pi f\,t + \phi)$"
        f"        $A={fmt(A)}$ V,   $f={fmt(f_val)}$ {f_unit},   $\\phi={phi_str}$ {p_unit}"
    )


def redraw():
    freq_hz = slider_freq.val * freq_factor()
    freq_for_window = max(freq_hz, LOW_FREQ_FLOOR * freq_factor())
    duration_sec = VIEW_PERIODS / freq_for_window
    t = np.linspace(0, duration_sec, N_SAMPLES, endpoint=False)
    y = slider_amp.val * np.sin(2 * np.pi * freq_hz * t + phase_to_rad(slider_phase.val))
    x = t * freq_factor()
    line.set_data(x, y)
    ax.set_xlim(0, x[-1] if x[-1] > 0 else 1)
    ax.set_xlabel(time_label())
    ax.set_title(build_title(), fontsize=11)
    fig.canvas.draw_idle()


redraw()


def link(slider, box, formatter=fmt, parser=float):
    def on_slider_change(_):
        box.set_val(formatter(slider.val))
        redraw()

    def on_box_submit(text):
        try:
            value = parser(text)
        except Exception:
            box.set_val(formatter(slider.val))
            return
        slider.set_val(max(slider.valmin, min(slider.valmax, value)))

    slider.on_changed(on_slider_change)
    box.on_submit(on_box_submit)


link(slider_amp, box_amp)
link(slider_phase, box_phase, formatter=fmt_phase, parser=parse_phase)

# Frequency box: custom handler that switches units on k / M / G suffix
_FREQ_SUFFIX_MAP = {"k": 1, "M": 2, "G": 3}  # index into FREQ_UNITS


def _on_freq_slider_change(_):
    box_freq.set_val(fmt(slider_freq.val))
    redraw()


def _on_freq_box_submit(text):
    text = text.strip()
    target_unit_idx = None
    numeric_text = text

    if text and text[-1] in _FREQ_SUFFIX_MAP:
        target_unit_idx = _FREQ_SUFFIX_MAP[text[-1]]
        numeric_text = text[:-1]

    try:
        value = float(numeric_text)
    except ValueError:
        box_freq.set_val(fmt(slider_freq.val))
        return

    if target_unit_idx is not None and target_unit_idx != freq_unit_idx[0]:
        # Convert the entered value (in the new unit) to the current unit so the
        # slider stays consistent, then switch units.
        new_factor = FREQ_UNITS[target_unit_idx][1]
        old_factor = freq_factor()
        freq_hz = value * new_factor          # absolute Hz
        freq_unit_idx[0] = target_unit_idx
        button_freq.label.set_text(FREQ_UNITS[freq_unit_idx[0]][0])
        value = freq_hz / new_factor          # value expressed in new unit

    value = max(slider_freq.valmin, min(slider_freq.valmax, value))
    slider_freq.set_val(value)
    box_freq.set_val(fmt(value))
    redraw()


slider_freq.on_changed(_on_freq_slider_change)
box_freq.on_submit(_on_freq_box_submit)


def cycle_freq_unit(_):
    freq_unit_idx[0] = (freq_unit_idx[0] + 1) % len(FREQ_UNITS)
    button_freq.label.set_text(FREQ_UNITS[freq_unit_idx[0]][0])
    redraw()


def cycle_phase_unit(_):
    current_rad = phase_to_rad(slider_phase.val)
    phase_unit_idx[0] = (phase_unit_idx[0] + 1) % len(PHASE_UNITS)
    new_max = PHASE_UNITS[phase_unit_idx[0]][1]
    half_turn_new = PHASE_UNITS[phase_unit_idx[0]][2]
    new_label = PHASE_UNITS[phase_unit_idx[0]][3]
    new_val = max(0.0, min(new_max, current_rad * half_turn_new / np.pi))
    slider_phase.valmin = 0.0
    slider_phase.valmax = new_max
    slider_phase.ax.set_xlim(0.0, new_max)
    slider_phase.label.set_text(new_label)
    slider_phase.set_val(new_val)
    button_phase.label.set_text(PHASE_UNITS[phase_unit_idx[0]][0])


button_freq.on_clicked(cycle_freq_unit)
button_phase.on_clicked(cycle_phase_unit)

plt.show()
