"""Generate assets/demo.png — README showcase figure.

Run from the repo root:
    python make_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from cylfit import fit_cylinder, fit_cylinder_known_radius, generate_noisy_cylinder
from cylfit.elliptical import fit_elliptical_cylinder
from cylfit.cone import fit_cone, _cone_residuals

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#0d1117"   # GitHub dark
PANEL   = "#161b22"
EDGE    = "#30363d"
INLIER  = "#58a6ff"   # bright blue
OUTLIER = "#f85149"   # vivid red
FITTED  = "#3fb950"   # green
AXIS_C  = "#e3b341"   # amber
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#bc8cff"   # purple

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "text.color": TEXT,
    "grid.color": EDGE,
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.titlecolor": TEXT,
})


# ── Geometry helpers ────────────────────────────────────────────────────────

def _basis(axis):
    h = np.array([0., 0., 1.])
    if abs(axis @ h) > 0.95:
        h = np.array([1., 0., 0.])
    u = h - (h @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _cylinder_mesh(p0, axis, radius, h_min, h_max, nt=40, nh=20):
    u, v = _basis(axis)
    theta = np.linspace(0, 2 * np.pi, nt)
    t     = np.linspace(h_min, h_max, nh)
    T, TH = np.meshgrid(t, theta)
    X = (p0[0] + T * axis[0]
         + radius * (np.cos(TH) * u[0] + np.sin(TH) * v[0]))
    Y = (p0[1] + T * axis[1]
         + radius * (np.cos(TH) * u[1] + np.sin(TH) * v[1]))
    Z = (p0[2] + T * axis[2]
         + radius * (np.cos(TH) * u[2] + np.sin(TH) * v[2]))
    return X, Y, Z


def _ellipse_mesh(p0, axis, u_vec, v_vec, a, b, angle, h_min, h_max, nt=48, nh=16):
    theta = np.linspace(0, 2 * np.pi, nt)
    t     = np.linspace(h_min, h_max, nh)
    T, TH = np.meshgrid(t, theta)
    ca, sa = np.cos(angle), np.sin(angle)
    ur =  ca * u_vec + sa * v_vec
    vr = -sa * u_vec + ca * v_vec
    X = p0[0] + T * axis[0] + a * np.cos(TH) * ur[0] + b * np.sin(TH) * vr[0]
    Y = p0[1] + T * axis[1] + a * np.cos(TH) * ur[1] + b * np.sin(TH) * vr[1]
    Z = p0[2] + T * axis[2] + a * np.cos(TH) * ur[2] + b * np.sin(TH) * vr[2]
    return X, Y, Z


def _cone_mesh(apex, axis, half_angle, t_min, t_max, nt=40, nh=20):
    u, v = _basis(axis)
    theta = np.linspace(0, 2 * np.pi, nt)
    t     = np.linspace(max(t_min, 0.02), t_max, nh)
    T, TH = np.meshgrid(t, theta)
    r = T * np.tan(half_angle)
    X = apex[0] + T * axis[0] + r * (np.cos(TH) * u[0] + np.sin(TH) * v[0])
    Y = apex[1] + T * axis[1] + r * (np.cos(TH) * u[1] + np.sin(TH) * v[1])
    Z = apex[2] + T * axis[2] + r * (np.cos(TH) * u[2] + np.sin(TH) * v[2])
    return X, Y, Z


def _annotate_box(ax, lines, loc="upper left"):
    x = 0.04 if "left" in loc else 0.96
    y = 0.97 if "upper" in loc else 0.06
    ha = "left" if "left" in loc else "right"
    va = "top"  if "upper" in loc else "bottom"
    txt = "\n".join(lines)
    ax.text2D(x, y, txt,
              transform=ax.transAxes,
              fontsize=7.5, color=TEXT,
              ha=ha, va=va,
              bbox=dict(facecolor="#21262d", edgecolor=EDGE,
                        alpha=0.88, boxstyle="round,pad=0.35"))


def _style_3d(ax, title):
    ax.set_facecolor(PANEL)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor(EDGE)
    ax.tick_params(colors=SUBTEXT, labelsize=6.5, pad=-2)
    ax.set_xlabel("X", fontsize=7, color=SUBTEXT, labelpad=-4)
    ax.set_ylabel("Y", fontsize=7, color=SUBTEXT, labelpad=-4)
    ax.set_zlabel("Z", fontsize=7, color=SUBTEXT, labelpad=-4)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=6)


# ── Data generation ─────────────────────────────────────────────────────────

print("Generating datasets …")

# 1. Clean fit
syn_clean = generate_noisy_cylinder(
    radius=1.5, noise=0.015, outlier_fraction=0.0,
    n_points=1200, random_state=42, height=7.0,
)
m_clean = fit_cylinder(syn_clean.points, threshold=0.06, ransac_trials=64, random_state=42)

# 2. Heavy outliers
syn_out = generate_noisy_cylinder(
    radius=2.0, noise=0.02, outlier_fraction=0.30,
    n_points=1500, random_state=7, height=8.0,
)
m_out = fit_cylinder(syn_out.points, threshold=0.08, ransac_trials=128, random_state=7)

# 3. Partial arc (180°)
syn_arc = generate_noisy_cylinder(
    radius=1.8, noise=0.012, outlier_fraction=0.0,
    n_points=800, random_state=11, height=6.0, partial_arc=0.5,
)
m_arc = fit_cylinder(syn_arc.points, threshold=0.06, ransac_trials=64, random_state=11)

# 4. Elliptical cylinder
rng4 = np.random.default_rng(33)
a_true, b_true = 2.0, 1.0
theta_e = rng4.uniform(0, 2 * np.pi, 1000)
t_e = rng4.uniform(-4, 4, 1000)
axis_e = np.array([0., 0., 1.])
u_e = np.array([1., 0., 0.])
v_e = np.array([0., 1., 0.])
pts_e = (t_e[:, None] * axis_e
         + a_true * np.cos(theta_e)[:, None] * u_e
         + b_true * np.sin(theta_e)[:, None] * v_e
         + rng4.normal(0, 0.015, (1000, 3)))
m_ell = fit_elliptical_cylinder(pts_e, threshold=0.08, ransac_trials=48, random_state=33)

# 5. Cone fit
rng5 = np.random.default_rng(55)
apex_true = np.array([0., 0., 0.])
axis_cone = np.array([0., 0., 1.])
alpha_true = np.deg2rad(25.0)
t_c = rng5.uniform(0.5, 5.0, 900)
r_c = t_c * np.tan(alpha_true)
phi_c = rng5.uniform(0, 2 * np.pi, 900)
pts_cone = np.column_stack([
    r_c * np.cos(phi_c) + rng5.normal(0, 0.02, 900),
    r_c * np.sin(phi_c) + rng5.normal(0, 0.02, 900),
    t_c + rng5.normal(0, 0.02, 900),
])
m_cone = fit_cone(pts_cone, ransac_trials=64, random_state=55)

# 6. Bias / accuracy plot data
print("Computing accuracy curve …")
noise_levels = np.linspace(0.005, 0.06, 12)
n_seeds = 12
mae_vals = []
std_vals = []
for nl in noise_levels:
    errs = []
    for s in range(n_seeds):
        syn_b = generate_noisy_cylinder(
            radius=1.5, noise=nl, outlier_fraction=0.0,
            n_points=600, random_state=s,
        )
        mb = fit_cylinder(syn_b.points, threshold=nl * 5 + 0.01,
                          ransac_trials=32, random_state=s)
        errs.append(abs(mb.radius - 1.5))
    mae_vals.append(np.mean(errs))
    std_vals.append(np.std(errs))
mae_vals = np.array(mae_vals)
std_vals = np.array(std_vals)

print("Rendering figure …")

# ── Layout ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), dpi=160, facecolor=BG)

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 8, figure=fig,
              left=0.04, right=0.97, top=0.93, bottom=0.08,
              wspace=0.38, hspace=0.32)
ax1 = fig.add_subplot(gs[0, 0:2], projection="3d")
ax2 = fig.add_subplot(gs[0, 2:4], projection="3d")
ax3 = fig.add_subplot(gs[0, 4:6], projection="3d")
ax4 = fig.add_subplot(gs[0, 6:8], projection="3d")
ax5 = fig.add_subplot(gs[1, 0:2], projection="3d")
ax6 = fig.add_subplot(gs[1, 2:4])
ax7 = fig.add_subplot(gs[1, 4:6])
ax8 = fig.add_subplot(gs[1, 6:8])


# ═══════════════════════════════════════════════════════════════════════════
# Panel 1 — Clean cylinder
# ═══════════════════════════════════════════════════════════════════════════
_style_3d(ax1, "① Clean Fit")

pts = syn_clean.points
res = np.abs(m_clean.residuals)
norm1 = Normalize(vmin=0, vmax=res.max())
cols = plt.cm.cool(norm1(res))
ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
            c=cols, s=2.5, alpha=0.7, linewidths=0, depthshade=True)

X, Y, Z = _cylinder_mesh(m_clean.axis_point, m_clean.axis_direction,
                         m_clean.radius, m_clean.height_min, m_clean.height_max)
ax1.plot_surface(X, Y, Z, color=FITTED, alpha=0.15, linewidth=0)
ax1.plot_wireframe(X, Y, Z, color=FITTED, alpha=0.35, linewidth=0.35, rstride=4, cstride=4)

# Axis arrow
p0, d = m_clean.axis_point, m_clean.axis_direction
hl = (m_clean.height_max - m_clean.height_min) * 0.55
ax1.quiver(p0[0], p0[1], p0[2], d[0]*hl, d[1]*hl, d[2]*hl,
           color=AXIS_C, linewidth=1.6, arrow_length_ratio=0.12)

r_err = abs(m_clean.radius - syn_clean.radius)
_annotate_box(ax1, [
    f"r_true = {syn_clean.radius:.2f}",
    f"r_fit  = {m_clean.radius:.4f}",
    f"error  = {r_err*1000:.1f} m×10⁻³",
    f"RMSE   = {m_clean.rmse*1000:.1f} m×10⁻³",
    f"inliers = {m_clean.inlier_mask.mean():.0%}",
    f"iters   = {m_clean.iterations}",
], "upper left")
ax1.view_init(elev=22, azim=35)

# Mini colorbar for residual scale
sm1 = ScalarMappable(norm=norm1, cmap="cool")
sm1.set_array([])
cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.55, pad=0.01, aspect=20,
                   orientation="vertical")
cb1.ax.tick_params(colors=SUBTEXT, labelsize=5.5)
cb1.set_label("|residual| (m)", color=SUBTEXT, fontsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 2 — 30 % outliers
# ═══════════════════════════════════════════════════════════════════════════
_style_3d(ax2, "② 30% Outliers  (MAGSAC+PROSAC)")

pts = syn_out.points
inl = m_out.inlier_mask
ax2.scatter(pts[inl, 0], pts[inl, 1], pts[inl, 2],
            c=INLIER, s=2.5, alpha=0.6, linewidths=0, depthshade=True, label="inlier")
ax2.scatter(pts[~inl, 0], pts[~inl, 1], pts[~inl, 2],
            c=OUTLIER, s=2.5, alpha=0.5, linewidths=0, depthshade=True, label="outlier")

X, Y, Z = _cylinder_mesh(m_out.axis_point, m_out.axis_direction,
                         m_out.radius, m_out.height_min, m_out.height_max)
ax2.plot_surface(X, Y, Z, color=FITTED, alpha=0.12, linewidth=0)
ax2.plot_wireframe(X, Y, Z, color=FITTED, alpha=0.32, linewidth=0.35, rstride=4, cstride=4)

p0, d = m_out.axis_point, m_out.axis_direction
hl = (m_out.height_max - m_out.height_min) * 0.55
ax2.quiver(p0[0], p0[1], p0[2], d[0]*hl, d[1]*hl, d[2]*hl,
           color=AXIS_C, linewidth=1.6, arrow_length_ratio=0.12)

r_err2 = abs(m_out.radius - syn_out.radius)
_annotate_box(ax2, [
    f"r_true  = {syn_out.radius:.2f}",
    f"r_fit   = {m_out.radius:.4f}",
    f"error   = {r_err2*100:.2f}%",
    f"RMSE    = {m_out.rmse*1000:.1f} m×10⁻³",
    f"inliers = {m_out.inlier_mask.mean():.0%}",
    f"outlier = 30%",
], "upper left")
ax2.view_init(elev=20, azim=-40)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 3 — Partial arc
# ═══════════════════════════════════════════════════════════════════════════
_style_3d(ax3, "③ Partial Arc (180°)")

pts = syn_arc.points
res3 = np.abs(m_arc.residuals)
norm3 = Normalize(vmin=0, vmax=max(res3.max(), 1e-9))
cols3 = plt.cm.plasma(norm3(res3))
ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
            c=cols3, s=3, alpha=0.75, linewidths=0, depthshade=True)

# Show the full fitted cylinder (ghost) and the arc region
X, Y, Z = _cylinder_mesh(m_arc.axis_point, m_arc.axis_direction,
                         m_arc.radius, m_arc.height_min, m_arc.height_max)
ax3.plot_surface(X, Y, Z, color=FITTED, alpha=0.07, linewidth=0)
ax3.plot_wireframe(X, Y, Z, color=FITTED, alpha=0.28, linewidth=0.3, rstride=4, cstride=4)

p0, d = m_arc.axis_point, m_arc.axis_direction
hl = (m_arc.height_max - m_arc.height_min) * 0.55
ax3.quiver(p0[0], p0[1], p0[2], d[0]*hl, d[1]*hl, d[2]*hl,
           color=AXIS_C, linewidth=1.6, arrow_length_ratio=0.12)

r_err3 = abs(m_arc.radius - syn_arc.radius)
_annotate_box(ax3, [
    f"r_true  = {syn_arc.radius:.2f}",
    f"r_fit   = {m_arc.radius:.4f}",
    f"error   = {r_err3*100:.2f}%",
    f"coverage = 50%",
    f"RMSE    = {m_arc.rmse*1000:.1f} m×10⁻³",
    f"converged = {'✓' if m_arc.converged else '✗'}",
], "upper left")
ax3.view_init(elev=18, azim=60)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 4 — Elliptical cylinder (3-D)
# ═══════════════════════════════════════════════════════════════════════════
_style_3d(ax4, "④ Elliptical Cylinder")

res4 = np.abs(m_ell.residuals)
norm4 = Normalize(vmin=0, vmax=max(res4.max(), 1e-9))
cols4 = plt.cm.winter(norm4(res4))
ax4.scatter(pts_e[:, 0], pts_e[:, 1], pts_e[:, 2],
            c=cols4, s=2.5, alpha=0.65, linewidths=0, depthshade=True)

from cylfit.core import _orthonormal_basis as _core_basis
u_fit, v_fit = _core_basis(m_ell.axis_direction)
X4, Y4, Z4 = _ellipse_mesh(
    m_ell.axis_point, m_ell.axis_direction,
    u_fit, v_fit,
    m_ell.semi_major, m_ell.semi_minor,
    m_ell.cross_section_angle,
    m_ell.height_min, m_ell.height_max,
)
ax4.plot_surface(X4, Y4, Z4, color=ACCENT, alpha=0.12, linewidth=0)
ax4.plot_wireframe(X4, Y4, Z4, color=ACCENT, alpha=0.30, linewidth=0.35, rstride=4, cstride=4)

p0e = m_ell.axis_point
d4 = m_ell.axis_direction
hl4 = (m_ell.height_max - m_ell.height_min) * 0.55
ax4.quiver(p0e[0], p0e[1], p0e[2], d4[0]*hl4, d4[1]*hl4, d4[2]*hl4,
           color=AXIS_C, linewidth=1.6, arrow_length_ratio=0.12)

_annotate_box(ax4, [
    f"a_true = {a_true:.2f}  b_true = {b_true:.2f}",
    f"a_fit  = {m_ell.semi_major:.4f}",
    f"b_fit  = {m_ell.semi_minor:.4f}",
    f"a_err  = {abs(m_ell.semi_major-a_true)*100:.2f}%",
    f"ratio  = {m_ell.aspect_ratio:.3f}",
    f"RMSE   = {m_ell.rmse*1000:.1f} m×10⁻³",
], "upper left")
ax4.view_init(elev=24, azim=25)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 5 — Cone fit (3-D)
# ═══════════════════════════════════════════════════════════════════════════
_style_3d(ax5, "⑤ Cone  (Half-angle 25°)")

res5 = np.abs(m_cone.residuals)
norm5 = Normalize(vmin=0, vmax=max(res5.max(), 1e-9))
cols5 = plt.cm.autumn(1.0 - norm5(res5))
ax5.scatter(pts_cone[:, 0], pts_cone[:, 1], pts_cone[:, 2],
            c=cols5, s=2.5, alpha=0.7, linewidths=0, depthshade=True)

t_vals = (pts_cone - m_cone.apex) @ m_cone.axis_direction
t_min_c = max(t_vals.min(), 0.1)
t_max_c = t_vals.max()
X5, Y5, Z5 = _cone_mesh(m_cone.apex, m_cone.axis_direction,
                         np.deg2rad(m_cone.half_angle_deg),
                         t_min_c, t_max_c)
ax5.plot_surface(X5, Y5, Z5, color="#f0883e", alpha=0.22, linewidth=0)
ax5.plot_wireframe(X5, Y5, Z5, color="#f0883e", alpha=0.50, linewidth=0.5,
                   rstride=3, cstride=3)

# Apex marker
ap = m_cone.apex
ax5.scatter([ap[0]], [ap[1]], [ap[2]], color=AXIS_C, s=50, zorder=10, marker="*")
d5 = m_cone.axis_direction
ax5.quiver(ap[0], ap[1], ap[2], d5[0]*t_max_c*0.5, d5[1]*t_max_c*0.5, d5[2]*t_max_c*0.5,
           color=AXIS_C, linewidth=1.4, arrow_length_ratio=0.10)

alpha_err = abs(m_cone.half_angle_deg - 25.0)
_annotate_box(ax5, [
    f"α_true = 25.00°",
    f"α_fit  = {m_cone.half_angle_deg:.3f}°",
    f"error  = {alpha_err:.3f}°",
    f"RMSE   = {m_cone.rmse*1000:.1f} m×10⁻³",
    f"inliers = {m_cone.inlier_mask.mean():.0%}",
    f"converged = {'✓' if m_cone.converged else '✗'}",
], "upper left")
ax5.view_init(elev=20, azim=30)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 6 — Ellipse cross-section (2-D)
# ═══════════════════════════════════════════════════════════════════════════
ax6.set_facecolor(PANEL)
for sp in ax6.spines.values():
    sp.set_edgecolor(EDGE)
ax6.tick_params(colors=SUBTEXT, labelsize=7)
ax6.set_title("⑥ Ellipse Cross-section (2D)", color=TEXT, fontsize=10, fontweight="bold", pad=6)
ax6.set_xlabel("u  (cross-section plane)", fontsize=7.5, color=SUBTEXT)
ax6.set_ylabel("v  (cross-section plane)", fontsize=7.5, color=SUBTEXT)
ax6.set_aspect("equal")
ax6.grid(True, color=EDGE, alpha=0.5, linewidth=0.5)

# Project points onto cross-section plane
u4, v4 = u_fit, v_fit
centred = pts_e - m_ell.axis_point
pu = centred @ u4
pv = centred @ v4
# Color by elliptic residual magnitude
ax6.scatter(pu, pv, c=res4, cmap="cool", s=3, alpha=0.6, linewidths=0,
            norm=norm4)

# Draw true ellipse
th = np.linspace(0, 2 * np.pi, 400)
ax6.plot(a_true * np.cos(th), b_true * np.sin(th),
         "--", color=SUBTEXT, linewidth=1.2, alpha=0.6, label="True ellipse")

# Draw fitted ellipse
ca, sa = np.cos(m_ell.cross_section_angle), np.sin(m_ell.cross_section_angle)
eu =  ca * m_ell.semi_major * np.cos(th) - sa * m_ell.semi_minor * np.sin(th)
ev =  sa * m_ell.semi_major * np.cos(th) + ca * m_ell.semi_minor * np.sin(th)
ax6.plot(eu, ev, color=ACCENT, linewidth=1.8, label="Fitted ellipse")

# Axes of fitted ellipse
ax6.annotate("",
    xy  =(ca * m_ell.semi_major, sa * m_ell.semi_major),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="-|>", color=AXIS_C, lw=1.5))
ax6.annotate("",
    xy  =(-sa * m_ell.semi_minor, ca * m_ell.semi_minor),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="-|>", color=AXIS_C, lw=1.5))

ax6.annotate(f"a = {m_ell.semi_major:.3f}",
             xy=(ca * m_ell.semi_major, sa * m_ell.semi_major),
             xytext=(8, 6), textcoords="offset points",
             color=AXIS_C, fontsize=7.5)
ax6.annotate(f"b = {m_ell.semi_minor:.3f}",
             xy=(-sa * m_ell.semi_minor, ca * m_ell.semi_minor),
             xytext=(6, 6), textcoords="offset points",
             color=AXIS_C, fontsize=7.5)

ax6.legend(fontsize=7, loc="lower right",
           facecolor="#21262d", edgecolor=EDGE, labelcolor=TEXT)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 7 — Residual histograms (all cases)
# ═══════════════════════════════════════════════════════════════════════════
ax7.set_facecolor(PANEL)
for sp in ax7.spines.values():
    sp.set_edgecolor(EDGE)
ax7.tick_params(colors=SUBTEXT, labelsize=7)
ax7.set_title("⑦ Residual Distributions", color=TEXT, fontsize=10, fontweight="bold", pad=6)
ax7.set_xlabel("|residual|  (m)", fontsize=7.5, color=SUBTEXT)
ax7.set_ylabel("density", fontsize=7.5, color=SUBTEXT)
ax7.grid(True, axis="y", color=EDGE, alpha=0.5, linewidth=0.5)

cases = [
    (np.abs(m_clean.residuals[m_clean.inlier_mask]), "#58a6ff", "Clean  (σ=0.015)", "--"),
    (np.abs(m_out.residuals[m_out.inlier_mask]),     "#3fb950", "30% outliers", "-"),
    (np.abs(m_arc.residuals[m_arc.inlier_mask]),     "#e3b341", "Partial arc", "-."),
    (np.abs(m_cone.residuals[m_cone.inlier_mask]),   "#f0883e", "Cone (α=25°)", ":"),
]
def _smooth_density(data, n_bins=60, smooth_passes=6):
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([0.10, 0.25, 0.30, 0.25, 0.10])
    s = counts.copy()
    for _ in range(smooth_passes):
        s = np.convolve(s, kernel, mode="same")
    return mids, s

for data, col, lbl, ls in cases:
    if data.size == 0:
        continue
    mids, sdens = _smooth_density(data, n_bins=55, smooth_passes=8)
    ax7.fill_between(mids, sdens, alpha=0.18, color=col)
    ax7.plot(mids, sdens, color=col, linewidth=2.0, linestyle=ls,
             label=f"{lbl}  μ={data.mean()*1e3:.1f} mm")

ax7.legend(fontsize=6.5, loc="upper right",
           facecolor="#21262d", edgecolor=EDGE, labelcolor=TEXT)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 8 — Radius MAE vs noise level
# ═══════════════════════════════════════════════════════════════════════════
ax8.set_facecolor(PANEL)
for sp in ax8.spines.values():
    sp.set_edgecolor(EDGE)
ax8.tick_params(colors=SUBTEXT, labelsize=7)
ax8.set_title("⑧ Radius MAE vs Noise  (n=12 seeds)", color=TEXT, fontsize=10,
              fontweight="bold", pad=6)
ax8.set_xlabel("noise  σ  (m)", fontsize=7.5, color=SUBTEXT)
ax8.set_ylabel("radius MAE  (m)", fontsize=7.5, color=SUBTEXT)
ax8.grid(True, color=EDGE, alpha=0.5, linewidth=0.5)

ax8.plot(noise_levels, mae_vals, color=INLIER, linewidth=2, marker="o",
         markersize=4.5, label="MAE (mean)")
ax8.fill_between(noise_levels,
                 mae_vals - std_vals,
                 mae_vals + std_vals,
                 color=INLIER, alpha=0.30, label="±1 std")
# Reference line: MAE = σ (optimal unbiased estimator)
ax8.plot(noise_levels, noise_levels, color=SUBTEXT, linewidth=1,
         linestyle="--", alpha=0.6, label="MAE = σ  (ideal)")

ax8.legend(fontsize=7, loc="upper left",
           facecolor="#21262d", edgecolor=EDGE, labelcolor=TEXT)

# ═══════════════════════════════════════════════════════════════════════════
# Super-title
# ═══════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.975,
         "cylfit  —  Robust 3-D Cylinder Fitting",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color=TEXT, fontfamily="DejaVu Sans")
fig.text(0.5, 0.956,
         "MAGSAC + PROSAC RANSAC  ·  Analytic LM  ·  Ellipse  ·  Cone  ·  Partial arc",
         ha="center", va="top", fontsize=8.5, color=SUBTEXT)

# ═══════════════════════════════════════════════════════════════════════════
# Legend strip at bottom
# ═══════════════════════════════════════════════════════════════════════════
legend_patches = [
    mpatches.Patch(color=INLIER,  label="Inlier point"),
    mpatches.Patch(color=OUTLIER, label="Outlier point"),
    mpatches.Patch(color=FITTED,  label="Fitted cylinder surface"),
    mpatches.Patch(color=ACCENT,  label="Fitted elliptical surface"),
    mpatches.Patch(color="#f0883e", label="Fitted cone surface"),
    mpatches.Patch(color=AXIS_C,  label="Estimated axis / apex"),
]
fig.legend(handles=legend_patches,
           loc="lower center", ncol=6, fontsize=7.5,
           facecolor="#21262d", edgecolor=EDGE, labelcolor=TEXT,
           framealpha=0.9, borderpad=0.5,
           bbox_to_anchor=(0.5, 0.0))

# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs("assets", exist_ok=True)
out = "assets/demo.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close(fig)
