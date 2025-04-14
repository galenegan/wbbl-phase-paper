import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
phase_vec = data["phase"]
phase_labels = np.degrees(data["phase"]).astype(int)
labels = [str(label) + r"$^\circ$" for label in phase_labels]
colors = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c"]

z_vec = data["z"] - 0.004
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
zidx_tau = (z_vec > -0.0015) & (z_vec < 0.0015)

# Normalization
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)
u_mean = np.nanmean(u[:, idx, :], axis=1)
N = len(data["ubr"])
ustar_wave = np.sqrt(np.nanmax(np.nanmean(data["tau_wave_total"][zidx_tau, :, :], axis=0), axis=1) / 1020)
ustar_mean = np.nanmean(ustar_wave[idx])
omega_mean = np.nanmean(data["omega"][idx])
delta_w_mean = ustar_mean / omega_mean
dfv = pd.DataFrame(index=z_vec / delta_w_mean, columns=phase_vec, data=u_mean)

# %% Jonsson data
phase_j = np.arange(0.0, 350, 15) * np.pi / 180.0 - np.pi
z_j = np.array([23, 20, 17, 14, 11, 9, 7, 5, 4, 3, 2, 1.5, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
dfj = pd.read_excel(os.path.join(get_project_root(), "data/jonsson/velocity_test_1.xlsx"), header=None, names=phase_j)
dfj = dfj.set_index(z_j)

# Normalization and averaging phases
ub = 220
ustar = np.sqrt(465)
omega = 2 * np.pi / 8.39
ab = ub / omega
kb = 2.3
delta_w = ustar / omega
dfj = dfj.set_index(z_j / delta_w)
dfj = dfj / ub
dfj = dfj[phase_vec]

# %% Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

for ii in range(u_mean.shape[1]):
    ax1.plot(dfv.values[:, ii], dfv.index, color=colors[ii], linewidth=2)
    ax2.plot(dfj.values[:, ii], dfj.index, color=colors[ii], label=labels[ii], linewidth=2)
ax1.set_ylim(0, 2.25)
ax2.set_ylim(0, 2.25)
ax1.set_xlabel(r"$\tilde{u} u_b^{-1}$")
ax1.set_ylabel(r"$(z - \delta_c) \delta_w^{-1}$")
ax1.set_title("(a)")
ax2.set_title("(b)")
ax2.set_xlabel(r"$\tilde{u} u_b^{-1}$")
ax2.set_ylabel(r"$z \delta_w^{-1}$")
ax2.legend(bbox_to_anchor=(1.01, 1.025))

# %% Averaging over higher ub
z_vec = data["z"] - 0.004
idx = data["ubr"] > 0.1
zidx_tau = (z_vec > -0.0015) & (z_vec < 0.0015)

# Normalization
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)
u_mean = np.nanmean(u[:, idx, :], axis=1)
N = len(data["ubr"])
ustar_wave = np.sqrt(np.nanmax(np.nanmean(data["tau_wave_total"][zidx_tau, :, :], axis=0), axis=1) / 1020)
ustar_mean = np.nanmean(ustar_wave[idx])
omega_mean = np.nanmean(data["omega"][idx])
delta_w_mean = ustar_mean / omega_mean
dfv = pd.DataFrame(index=z_vec / delta_w_mean, columns=phase_vec, data=u_mean)


for ii in range(u_mean.shape[1]):
    ax3.plot(dfv.values[:, ii], dfv.index, color=colors[ii], linewidth=2)
    ax4.plot(dfj.values[:, ii], dfj.index, color=colors[ii], label=labels[ii], linewidth=2)
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 1)
ax3.set_xlabel(r"$\tilde{u} u_b^{-1}$")
ax3.set_ylabel(r"$(z - \delta_c) \delta_w^{-1}$")
ax3.set_title("(c)")
ax4.set_title("(d)")
ax4.set_xlabel(r"$\tilde{u} u_b^{-1}$")
ax4.set_ylabel(r"$z \delta_w^{-1}$")

fig.tight_layout(pad=0.5)
plt.savefig("files/ubar_comparison.png", dpi=300)
plt.show()
