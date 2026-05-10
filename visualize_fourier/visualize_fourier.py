"""visualize_fourier.py – up to 8 stacked sine wave panels, add/remove with +/- buttons."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from sin_wave import SinWave

plt.style.use("dark_background")

MAX_WAVES = 8
BTN_H     = 0.04
BTN_W     = 0.06


waves: list = []


def _rebuild(target_n):
    """Destroy all axes and rebuild with target_n panels."""
    global waves

    for w in waves:
        for ax_obj in [w.ax,
                       w.sl_amp.ax,  w.sl_freq.ax,  w.sl_phase.ax,
                       w.box_amp.ax, w.box_freq.ax, w.box_phase.ax,
                       w.btn_freq.ax, w.btn_phase.ax]:
            ax_obj.remove()
    waves.clear()

    if target_n == 0:
        fig.canvas.draw_idle()
        return

    n       = target_n
    btn_top = 0.97
    btn_bot = btn_top - BTN_H
    avail   = btn_bot - 0.02
    panel_h = avail / n

    for i in range(n):
        bot      = avail - (i + 1) * panel_h + 0.02
        plot_h   = panel_h * 0.56
        wid_h    = panel_h * 0.38
        plot_bot = bot + panel_h - plot_h
        wid_bot  = bot

        plot_rect   = [0.08, plot_bot, 0.86, plot_h]
        widget_rect = [0.08, wid_bot,  0.86, wid_h]
        waves.append(SinWave(fig, plot_rect, widget_rect))

    fig.canvas.draw_idle()


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 7))
fig.patch.set_facecolor("black")

btn_top = 0.97
btn_bot = btn_top - BTN_H

ax_plus  = fig.add_axes([0.88, btn_bot, BTN_W, BTN_H])
ax_minus = fig.add_axes([0.80, btn_bot, BTN_W, BTN_H])
btn_plus  = Button(ax_plus,  "+", color="#22c55e", hovercolor="#16a34a")
btn_minus = Button(ax_minus, "−", color="#ef4444", hovercolor="#dc2626")

for b in (btn_plus, btn_minus):
    b.label.set_fontsize(16)
    b.label.set_color("white")


def on_plus(_):
    if len(waves) < MAX_WAVES:
        _rebuild(len(waves) + 1)


def on_minus(_):
    if len(waves) > 0:
        _rebuild(len(waves) - 1)


btn_plus.on_clicked(on_plus)
btn_minus.on_clicked(on_minus)

_rebuild(1)

plt.show()
