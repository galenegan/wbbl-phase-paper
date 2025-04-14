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
z_vec = data["z"] - 0.004
zavg_u = z_vec > 0.009
zavg_tau = (z_vec > -0.0015) & (z_vec < 0.0015)

# Velocity averaging and normalization
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
u_mean_vec = np.nanmean(u[:, idx, :][zavg_u, :, :], axis=(0, 1))
u_mean_vec_plot = np.concatenate((u_mean_vec, [u_mean_vec[0]]))

# Stress averaging and normalization
tau_wave_max = np.nanmax(np.nanmean(data["tau_wave_maj"][zavg_tau, :, :][:, idx, :], axis=0), axis=1)
tau_mean_vec = np.nanmean(data["tau_wave_maj"][zavg_tau, :, :][:, idx, :] / tau_wave_max.reshape(1, -1, 1), axis=(0, 1))
tau_mean_vec_plot = np.concatenate((tau_mean_vec, [tau_mean_vec[0]]))

# %% Phase lag on interpolated data
phase_hd = np.linspace(phase_plot[0], phase_plot[-1], 1000)
f_u = interp1d(phase_plot, u_mean_vec_plot, kind="cubic")
f_tau = interp1d(phase_plot, tau_mean_vec_plot, kind="cubic")
u_hd = f_u(phase_hd)
tau_hd = f_tau(phase_hd)
idx_u_max = np.nanargmax(u_hd)
phase_u_max = phase_hd[idx_u_max]
idx_tau_max = np.nanargmax(tau_hd)
delta_phi = phase_u_max - phase_hd[idx_tau_max]
phase_lag = (delta_phi + np.pi) % (2 * np.pi) - np.pi

# %% Jonsson data
ub = 220
tau_max = 465
phase_j = np.arange(0.0, 361, 15) - 180
dfv = pd.read_excel(os.path.join(get_project_root(), "data/jonsson_velocity_test_1.xlsx"), header=None, names=phase_j)
dfs = pd.read_excel(os.path.join(get_project_root(), "data/jonsson_stress_test_1.xlsx"), header=None, names=phase_j)
dfv.iloc[:, -1] = dfv.iloc[:, 0]
dfs.iloc[:, -1] = dfs.iloc[:, 0]
u_mean_jon = dfv.loc[dfs.index[0], :] / ub
tau_mean_jon = dfs.loc[dfs.index[-1], :] / tau_max
u_mean_jon_plot = u_mean_jon.values
tau_mean_jon_plot = tau_mean_jon.values

# %% Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(phase_plot, u_mean_vec_plot, "o", color="0.0")
ax1.plot(phase_hd, u_hd, "--", linewidth=2, color="0.0")
ax1.set_ylabel(r"$\tilde{u}_p u_b^{-1}$")
ax1.set_xlabel(r"$\theta$")
ax12 = ax1.twinx()
ax12.plot(phase_plot, tau_mean_vec_plot, "o", color="0.6")
ax12.plot(phase_hd, tau_hd, "--", linewidth=2, color="0.6")
ax12.set_ylabel(r"$\langle \tau_{w,x} \rangle \tau_{wm,x}^{-1}$", color="0.4")
ax1.set_xticks(phase_plot[::3])
ax1.set_xticklabels([f"${int(np.round(angle, 0))}^\\circ$" for angle in phase_plot[::3]])
ax1.set_title("(a)")
ax1.grid("y")
[t.set_color("0.4") for t in ax12.yaxis.get_ticklines()]
[t.set_color("0.4") for t in ax12.yaxis.get_ticklabels()]
ax1.set_yticks(np.arange(-1, 1.1, 0.5))
ax12.set_yticks(np.arange(-1, 1.1, 0.5))
ax1.set_ylim(-1.2, 1.2)
ax2.plot(phase_j, u_mean_jon_plot, "-", linewidth=2, color="0.0")
ax22 = ax2.twinx()
ax22.plot(phase_j, tau_mean_jon_plot, "-", linewidth=2, color="0.6")
ax2.set_ylabel(r"$\tilde{u}_p u_b^{-1}$")
ax2.set_xlabel(r"$\theta$")
ax22.set_ylabel(r"$\tau_b \tau_{wm}^{-1}$", color="0.4")
ax2.set_xticks(phase_j[::3])
ax2.set_xticklabels([f"${int(np.round(angle))}^\\circ$" for angle in phase_j[::3]])
ax2.set_title("(b)")
[t.set_color("0.4") for t in ax22.yaxis.get_ticklines()]
[t.set_color("0.4") for t in ax22.yaxis.get_ticklabels()]
ax2.set_yticks(np.arange(-1, 1.1, 0.5))
ax2.set_ylim(-1.2, 1.2)
ax22.set_yticks(np.arange(-1, 1.1, 0.5))
ax2.grid("y")
fig.tight_layout(pad=1)
plt.savefig("files/mean_phase_lag.png", dpi=300)
plt.show()
