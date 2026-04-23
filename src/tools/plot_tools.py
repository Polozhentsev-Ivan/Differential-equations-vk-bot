"""Инструменты визуализации ОДУ: поле направлений, изоклины, интегральные кривые."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sympy import Symbol, Function, lambdify, solve

from tools.sympy_tools import parse_ode

x_sym = Symbol("x")
y_func = Function("y")

_ISOCLINE_CMAP = plt.cm.coolwarm


def _extract_rhs(equation_str: str):
    """Извлекает f(x, y) из ОДУ y' = f(x, y) как SymPy-выражение."""
    eq = parse_ode(equation_str)
    dy = y_func(x_sym).diff(x_sym)
    solutions = solve(eq, dy)
    if not solutions:
        raise ValueError(f"Не удалось выделить y' из: {equation_str}")
    return solutions[0]


def _rhs_to_numpy(rhs_expr):
    """Конвертирует SymPy-выражение f(x, y) в numpy-функцию f(x_arr, y_arr)."""
    y_sym = Symbol("y")
    expr = rhs_expr.subs(y_func(x_sym), y_sym)
    return lambdify((x_sym, y_sym), expr, modules=["numpy"])


def _safe_eval(f_np, X, Y):
    """Вычисляет f(X, Y) с обработкой скалярных/векторных случаев."""
    try:
        Z = np.asarray(f_np(X, Y), dtype=float)
        if Z.shape != X.shape:
            raise ValueError
        return Z
    except Exception:
        return np.vectorize(lambda xi, yi: float(f_np(xi, yi)))(X, Y)


def _draw_lineal_elements(ax, X, Y, slopes, half_len, **kwargs):
    """Рисует штрихи (линейные элементы) с заданными наклонами в точках (X, Y).

    Каждый штрих — отрезок длиной 2*half_len, центрированный в точке,
    с наклоном = slopes[i, j].
    """
    segs = []
    angles = np.arctan(slopes)
    dx = half_len * np.cos(angles)
    dy = half_len * np.sin(angles)

    it = np.nditer([X, Y, dx, dy], flags=["multi_index"])
    while not it.finished:
        xi, yi, dxi, dyi = float(it[0]), float(it[1]), float(it[2]), float(it[3])
        if np.isfinite(dxi) and np.isfinite(dyi):
            segs.append([(xi - dxi, yi - dyi), (xi + dxi, yi + dyi)])
        it.iternext()

    if segs:
        lc = LineCollection(segs, **kwargs)
        ax.add_collection(lc)
    return lc if segs else None


def _sample_contour_points(cs_result, spacing: float = 0.6):
    """Извлекает точки из matplotlib contour, прореживая по расстоянию."""
    points = []
    for level_segs in cs_result.allsegs:
        for seg in level_segs:
            verts = np.asarray(seg)
            if len(verts) < 2:
                continue
            points.append(verts[0])
            cumulative = 0.0
            for i in range(1, len(verts)):
                d = np.linalg.norm(verts[i] - verts[i - 1])
                cumulative += d
                if cumulative >= spacing:
                    points.append(verts[i])
                    cumulative = 0.0
    return np.array(points) if points else np.empty((0, 2))


def plot_isoclines_full(
    equation_str: str,
    c_values: Optional[list[float]] = None,
    x_range: tuple[float, float] = (-5, 5),
    y_range: tuple[float, float] = (-5, 5),
    n_grid: int = 20,
    n_solution_curves: int = 8,
    output_path: Optional[str | Path] = None,
) -> str:
    """Строит комплексный график: поле направлений + изоклины + интегральные кривые.

    Returns:
        Путь к сохранённому PNG файлу.
    """
    if c_values is None:
        c_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    rhs_expr = _extract_rhs(equation_str)
    f_np = _rhs_to_numpy(rhs_expr)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    half_len = min(span_x, span_y) / n_grid * 0.45

    # ── 1. Фоновое поле направлений (тонкие серые штрихи) ──
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    slopes_bg = _safe_eval(f_np, X, Y)

    _draw_lineal_elements(
        ax, X, Y, slopes_bg, half_len,
        colors="#9e9e9e", linewidths=0.6, alpha=0.45, zorder=1,
    )

    # ── 2. Изоклины (контурные кривые) + штрихи наклона C на них ──
    x_fine = np.linspace(x_range[0], x_range[1], 500)
    y_fine = np.linspace(y_range[0], y_range[1], 500)
    Xf, Yf = np.meshgrid(x_fine, y_fine)
    Zf = _safe_eval(f_np, Xf, Yf)

    norm = Normalize(vmin=min(c_values), vmax=max(c_values))
    sm = ScalarMappable(cmap=_ISOCLINE_CMAP, norm=norm)

    iso_half = half_len * 1.3
    spacing = min(span_x, span_y) / n_grid * 1.1

    for c_val in c_values:
        color = _ISOCLINE_CMAP(norm(c_val))

        cs = ax.contour(
            Xf, Yf, Zf,
            levels=[c_val],
            colors=[color],
            linewidths=1.8,
            linestyles="solid",
            zorder=2,
        )
        ax.clabel(cs, fmt=f"C={c_val:.4g}", fontsize=8)

        pts = _sample_contour_points(cs, spacing=spacing)
        if len(pts) == 0:
            continue

        Xp = pts[:, 0].reshape(1, -1)
        Yp = pts[:, 1].reshape(1, -1)
        slope_arr = np.full_like(Xp, c_val)

        _draw_lineal_elements(
            ax, Xp, Yp, slope_arr, iso_half,
            colors=color, linewidths=2.2, alpha=0.95, zorder=3,
        )

    # ── 3. Интегральные кривые ──
    from scipy.integrate import solve_ivp

    y0_values = np.linspace(y_range[0], y_range[1], n_solution_curves)
    x_span_fwd = (0, x_range[1])
    x_span_bwd = (0, x_range[0])
    t_eval_fwd = np.linspace(0, x_range[1], 300)
    t_eval_bwd = np.linspace(0, x_range[0], 300)

    def ode_func(t, y_val):
        try:
            return [float(f_np(t, y_val[0]))]
        except (ValueError, OverflowError):
            return [0.0]

    for y0 in y0_values:
        for x_span, t_eval in [(x_span_fwd, t_eval_fwd), (x_span_bwd, t_eval_bwd)]:
            try:
                sol = solve_ivp(
                    ode_func, x_span, [y0],
                    t_eval=t_eval, max_step=0.1, rtol=1e-6, atol=1e-8,
                )
                if sol.success:
                    mask = (sol.y[0] >= y_range[0] - 1) & (sol.y[0] <= y_range[1] + 1)
                    ax.plot(
                        sol.t[mask], sol.y[0][mask],
                        color="#1565C0", alpha=0.5, linewidth=0.9, zorder=4,
                    )
            except Exception:
                pass

    # ── 4. Оформление ──
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Наклон C (изоклины)")
    cbar.set_ticks(c_values)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"Поле направлений и изоклины: {equation_str}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")

    if output_path is None:
        output_path = Path("output") / "isoclines.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
