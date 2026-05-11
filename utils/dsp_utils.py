"""dsp_utils.py – pure DSP helpers, no Qt/pyqtgraph dependencies."""

import ast
import operator
import numpy as np

# ── safe math eval ────────────────────────────────────────────────────────────
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_ALLOWED_UNARY  = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED_NAMES  = {"pi": np.pi}
FREQ_SUFFIX     = {"k": 1, "M": 2, "G": 3}


def safe_eval(expr: str) -> float:
    """Safely evaluate a numeric math expression string (supports pi, +−×÷**)."""
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
        raise ValueError(f"disallowed expression node: {type(node)}")
    return visit(ast.parse(expr.strip(), mode="eval"))


# ── number formatting ─────────────────────────────────────────────────────────
def fmt(v: float) -> str:
    return f"{v:.3g}"


def fmt_phase(val: float, phase_unit_idx: int, phase_units: list) -> str:
    if phase_units[phase_unit_idx][0] != "rad":
        return fmt(val)
    if val == 0:
        return "0"
    pm = val / np.pi
    if abs(pm - round(pm)) < 0.01:
        n = int(round(pm))
        return "pi" if n == 1 else f"{n}pi"
    return f"{pm:.3g}pi"


def fmt_hz(hz: float) -> str:
    """Format a frequency in Hz as a human-readable string."""
    ax = abs(hz)
    if ax >= 1e9:   return f"{hz/1e9:.4g} GHz"
    if ax >= 1e6:   return f"{hz/1e6:.4g} MHz"
    if ax >= 1e3:   return f"{hz/1e3:.4g} kHz"
    return f"{hz:.4g} Hz"


# ── DSP ───────────────────────────────────────────────────────────────────────
def to_dbm(mag: np.ndarray, ref_ohm: float = 50.0) -> np.ndarray:
    """Convert linear voltage magnitude array to dBm (50-ohm reference)."""
    power_mw = (mag ** 2) / (2.0 * ref_ohm) * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(power_mw > 0, power_mw, 1e-30))


def compute_continuous_fft(signal: np.ndarray, t: np.ndarray):
    """Approximate continuous Fourier Transform. Returns (freqs_hz, mag_dbm)."""
    N  = len(signal)
    dt = (t[-1] - t[0]) / (N - 1)
    X  = np.fft.rfft(signal) * dt
    return np.fft.rfftfreq(N, d=dt), to_dbm(np.abs(X))


def compute_dft(signal: np.ndarray, sample_rate_hz: float):
    """DFT of discrete samples. Returns (freqs_hz, mag_dbm)."""
    N = len(signal)
    X = np.fft.rfft(signal) / N
    return np.fft.rfftfreq(N, d=1.0 / sample_rate_hz), to_dbm(np.abs(X))


def fft_view_range(freqs: np.ndarray, mag_dbm: np.ndarray,
                   nyquist: float | None = None, threshold_db: float = 30.0):
    """
    Return (f_start, f_end, x_start_hz, x_end_hz, peak_dbm) for an FFT plot.
    Spikes are bins within threshold_db of the peak.
    x_start/end are padded half-a-decade in log space.
    """
    mask = (freqs > 0) & np.isfinite(mag_dbm)
    if nyquist is not None:
        mask &= freqs <= nyquist
    fa, ma = freqs[mask], mag_dbm[mask]
    if not len(fa):
        return None
    peak   = float(np.max(ma))
    spikes = ma >= peak - threshold_db
    f_lo   = float(fa[spikes][0])  if np.any(spikes) else float(fa[0])
    f_hi   = float(fa[spikes][-1]) if np.any(spikes) else float(fa[-1])
    x_start = 10 ** (np.log10(max(f_lo, 1e-3)) - 0.5)
    x_end   = 10 ** (np.log10(f_hi) + 0.5)
    return fa, ma, x_start, x_end, peak


def zero_order_hold(t: np.ndarray, y: np.ndarray):
    """Return (t_hold, y_hold) for a zero-order hold reconstruction."""
    if len(t) < 2:
        return np.array([]), np.array([])
    return np.repeat(t, 2)[1:], np.repeat(y, 2)[:-1]
