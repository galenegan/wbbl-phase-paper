import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from src.utils.project_utils import get_project_root

params = {
    "axes.labelsize": 24,
    "font.size": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)


# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino_15deg.npy")
data = np.load(data_path, allow_pickle=True).item()
phase_vec = data["phase"]
phase_plot = np.degrees(np.concatenate((phase_vec, [-phase_vec[0]])))
phase_labels = phase_plot.astype(int)
labels = [str(label) + r"$^\circ$" for label in phase_labels]
colors = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c"]

z_vec = data["z"] - 0.004
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
zidx_tau = (z_vec > -0.0015) & (z_vec < 0.0015)

# Normalization
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)
u_mean = np.nanmean(u[:, idx, :], axis=1)
u_bed_vec = u_mean[z_vec == 0, :].squeeze()
u_bed_vec = np.concatenate((u_bed_vec, [u_bed_vec[0]]))
u_pot_vec = u_mean[0, :].squeeze()
u_pot_vec = np.concatenate((u_pot_vec, [u_pot_vec[0]]))

# %%
phase_hd = np.linspace(phase_plot[0], phase_plot[-1], 1000)
f_up = interp1d(phase_plot, u_pot_vec, kind="cubic")
f_ub = interp1d(phase_plot, u_bed_vec, kind="cubic")
up_vec_hd = f_up(phase_hd)
ub_vec_hd = f_ub(phase_hd)
idx_u_max = np.nanargmax(up_vec_hd)
phase_u_max = phase_hd[idx_u_max]
idx_ub_max = np.nanargmax(ub_vec_hd)
delta_phi = phase_u_max - phase_hd[idx_ub_max]
phase_lag_vec = (delta_phi + 180) % (360) - 180
# %% Jonsson data
phase_j = np.arange(0.0, 350, 15) * np.pi / 180.0 - np.pi
z_j = np.array([23, 20, 17, 14, 11, 9, 7, 5, 4, 3, 2, 1.5, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
dfj = pd.read_excel(os.path.join(get_project_root(), "data/jonsson_velocity_test_1.xlsx"), header=None, names=phase_j)
dfj = dfj.set_index(z_j)

# Normalization and averaging phases
ub = 220
dfj = dfj / ub

u_bed_j = dfj.iloc[-1, :].values.squeeze()
u_bed_j = np.concatenate((u_bed_j, [u_bed_j[0]]))
u_pot_j = dfj.iloc[0, :].values.squeeze()
u_pot_j = np.concatenate((u_pot_j, [u_pot_j[0]]))

# %%
phase_hd = np.linspace(phase_plot[0], phase_plot[-1], 1000)
f_up = interp1d(phase_plot, u_pot_j, kind="cubic")
f_ub = interp1d(phase_plot, u_bed_j, kind="cubic")
up_j_hd = f_up(phase_hd)
ub_j_hd = f_ub(phase_hd)
idx_u_max = np.nanargmax(up_j_hd)
phase_u_max = phase_hd[idx_u_max]
idx_ub_max = np.nanargmax(ub_j_hd)
delta_phi = phase_u_max - phase_hd[idx_ub_max]
phase_lag_j = (delta_phi + 180) % (360) - 180
# %% Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.plot(phase_plot, u_pot_vec, "o", linewidth=2, color="0.0", label=r"$\tilde{u}_p u_0^{-1}$")
ax1.plot(phase_hd, up_vec_hd, '-', linewidth=2, color="0.0")
ax1.plot(phase_plot, u_bed_vec, "o", linewidth=2, color="0.6", label=r"$\tilde{u}_b u_0^{-1}$")
ax1.plot(phase_hd, ub_vec_hd, '-', linewidth=2, color="0.6")
ax1.plot(
    phase_hd[np.argmax(up_vec_hd)],
    np.max(up_vec_hd),
    marker="o",
    markersize=20,
    markeredgecolor="0.0",
    markerfacecolor="none",
    linestyle="--",
    markeredgewidth=1,
)
ax1.plot(
    phase_hd[np.argmax(ub_vec_hd)],
    np.max(ub_vec_hd),
    marker="o",
    markersize=20,
    markeredgecolor="0.6",
    markerfacecolor="none",
    linestyle="--",
    markeredgewidth=1,
)
ax1.set_ylabel(r"$\tilde{u} u_0^{-1}$")
ax1.set_xlabel(r"$\theta$")
ax1.set_xticks(phase_plot)
ax1.set_xticklabels([f"${int(np.round(angle, 0))}^\\circ$" for angle in phase_plot])
ax1.set_title("(a)")
ax1.grid("y")
ax1.set_yticks(np.arange(-1, 1.1, 0.5))
ax1.set_ylim(-1.25, 1.25)
ax1.legend()
ax2.plot(phase_plot, u_pot_j, "o", color="0.0", label=r"$\tilde{u}_p u_0^{-1}$")
ax2.plot(phase_hd, up_j_hd, '-', linewidth=2, color="0.0")
ax2.plot(phase_plot, u_bed_j, "o", linewidth=2, color="0.6", label=r"$\tilde{u}_b u_0^{-1}$")
ax2.plot(phase_hd, ub_j_hd, '-', linewidth=2, color="0.6")
ax2.plot(
    phase_hd[np.argmax(up_j_hd)],
    np.max(up_j_hd),
    marker="o",
    markersize=20,
    markeredgecolor="0.0",
    markerfacecolor="none",
    linestyle="--",
    markeredgewidth=1,
)
ax2.plot(
    phase_hd[np.argmax(ub_j_hd)],
    np.max(ub_j_hd),
    marker="o",
    markersize=20,
    markeredgecolor="0.6",
    markerfacecolor="none",
    linestyle="--",
    markeredgewidth=1,
)
ax2.set_ylabel(r"$\tilde{u} u_0^{-1}$")
ax2.set_xlabel(r"$\theta$")
ax2.set_xticks(phase_plot[::3])
ax2.set_xticklabels([f"${int(np.round(angle, 0))}^\\circ$" for angle in phase_plot[::3]])
ax2.set_title("(b)")
ax2.grid("y")
ax2.set_yticks(np.arange(-1, 1.1, 0.5))
ax2.set_ylim(-1.25, 1.25)
ax2.legend()

fig.tight_layout(pad=1)
plt.savefig("plots/ub_up.png", dpi=300)
plt.show()
