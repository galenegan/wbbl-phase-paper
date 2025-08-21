import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import root
from src.utils.project_utils import get_project_root

params = {
    "axes.labelsize": 28,
    "font.size": 28,
    "legend.fontsize": 18,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
kn = 0.00645
mask = (data["ubr"] > 0.015) & (data["omega"] > 1)
ub = data["ubr"][mask]
omega = data["omega"][mask]
ab_kb = (ub / omega) / kn
Re = (ub**2 / omega) / 1e-6

kb2 = 0.01
ab_kb_2 = (ub / omega) / kb2
kb3 = 0.02844
ab_kb_3 = (ub / omega) / kb3


# %% Kamphuis regimes
def friction_factor(fw, ab_kb):
    return 1 / (4 * np.sqrt(fw)) + np.log10(1 / (4 * np.sqrt(fw))) + 0.35 - (4 / 3) * np.log10(ab_kb)


def regime_curve(Re, ab_kb, fw, limit):
    return Re * np.sqrt(fw / 2) - limit * ab_kb


ab_kb_lin = np.linspace(0.1, 50, 100)

# Friction factor first
fw = np.zeros_like(ab_kb_lin)
a1, a2, a3 = 5.5, -0.2, -6.3
fw_nielsen = np.exp(a1 * ab_kb_lin**a2 + a3)
for ii in range(len(ab_kb_lin)):
    res = root(friction_factor, x0=fw_nielsen[ii], args=(ab_kb_lin[ii]))
    fw[ii] = res.x

Re_rough_turb = np.zeros_like(ab_kb_lin)
Re_transitional = np.zeros_like(ab_kb_lin)
for ii in range(len(ab_kb_lin)):
    res = root(regime_curve, x0=ab_kb_lin[ii] * 500, args=(ab_kb_lin[ii], fw[ii], 200))
    Re_rough_turb[ii] = res.x
    res = root(regime_curve, x0=ab_kb_lin[ii] * 50, args=(ab_kb_lin[ii], fw[ii], 15))
    Re_transitional[ii] = res.x

# %% Plotting
colors = ["#6929c4", "#1192e8", "#005d5d"]
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(Re, ab_kb, "o", color=colors[0], label=r"$k_b = $" + f" {kn * 100:.3f} cm")
ax.loglog(Re, ab_kb_2, "o", color=colors[1], label=r"$k_b = $" + f" {kb2 * 100:.3f} cm")
ax.loglog(Re, ab_kb_3, "o", color=colors[2], label=r"$k_b = $" + f" {kb3 * 100:.3f} cm")
ax.loglog(Re_transitional, ab_kb_lin, color="0.0")
ax.loglog(Re_rough_turb, ab_kb_lin, color="0.0")
leg = plt.legend()
ax.set_xlabel(r"$Re_w$")
ax.set_ylabel(r"$a_b k_b^{-1}$")
ax.set_xlim(1e1, 1e4)
leg.legend_handles[0].set_markersize(6)
leg.legend_handles[1].set_markersize(6)
leg.legend_handles[2].set_markersize(6)
fig.tight_layout(pad=0.5)
plt.savefig("files/friction_regimes.png", dpi=300)
plt.show()
