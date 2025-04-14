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
z_tau = (z_vec > -0.0015) & (z_vec < 0.0015)  # Bed stress

# Normalization
tau_wave_max = np.nanmax(np.nanmean(data["tau_wave_maj"][z_tau, :, :], axis=0), axis=1)
tau = data["tau_wave_maj"] / tau_wave_max.reshape(1, -1, 1)
ustar_wave = np.sqrt(tau_wave_max / 1020)

ustar_mean = np.nanmean(ustar_wave[idx])
omega_mean = np.nanmean(data["omega"][idx])
delta_w_mean = ustar_mean / omega_mean
tau_mean = np.nanmean(tau[:, idx, :], axis=1)
dfv = pd.DataFrame(index=z_vec / delta_w_mean, columns=phase_vec, data=tau_mean)

# %% Jonsson data
phase_j = np.arange(0.0, 350, 15) * np.pi / 180.0 - np.pi
z_j = np.array([17, 14, 11, 9, 7, 5, 4, 3, 2, 1.5, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.0])
dfj = pd.read_excel(os.path.join(get_project_root(), "data/jonsson/stress_test_1.xlsx"), header=None, names=phase_j)
dfj = dfj.set_index(z_j)

ub = 220
tau_max = 465
ustar_max = np.sqrt(tau_max)
omega = 2 * np.pi / 8.39
ab = ub / omega
kb = 2.3
relative_roughness_jonsson = ab / kb
delta_w = ustar_max / omega
dfj = dfj.set_index(z_j / delta_w)
dfj = dfj / tau_max
dfj = dfj[phase_vec]

# %% Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

for ii in range(tau_mean.shape[1]):
    ax1.plot(dfv.values[:, ii], dfv.index, color=colors[ii], linewidth=2)
    ax2.plot(dfj.values[:, ii], dfj.index, color=colors[ii], label=labels[ii], linewidth=2)
ax1.set_xlim(-1, 1)
ax2.set_xlim(-1, 1)
ax1.set_ylim(0, 2.25)
ax2.set_ylim(0, 2.25)
ax1.set_xlabel(r"$\tau_{w,x} \tau_{wm,x}^{-1}$")
ax1.set_ylabel(r"$(z - \delta_c) \delta_w^{-1}$")
ax1.set_title("(a)")
ax2.set_title("(b)")
ax2.set_xlabel(r"$\tau_{w,x} \tau_{wm,x}^{-1}$")
ax2.set_ylabel(r"$z \delta_w^{-1}$")
ax2.legend(bbox_to_anchor=(1.01, 1.025))


# %% And now a different averaging scheme
z_vec = data["z"] - 0.004
idx = data["ubr"] > 0.1
z_tau = (z_vec > -0.0015) & (z_vec < 0.0015)  # Bed stress

# Normalization
tau_wave_max = np.nanmax(np.nanmean(data["tau_wave_maj"][z_tau, :, :], axis=0), axis=1)
tau = data["tau_wave_maj"] / tau_wave_max.reshape(1, -1, 1)
ustar_wave = np.sqrt(tau_wave_max / 1020)

ustar_mean = np.nanmean(ustar_wave[idx])
omega_mean = np.nanmean(data["omega"][idx])
delta_w_mean = ustar_mean / omega_mean
tau_mean = np.nanmean(tau[:, idx, :], axis=1)
dfv = pd.DataFrame(index=z_vec / delta_w_mean, columns=phase_vec, data=tau_mean)

for ii in range(tau_mean.shape[1]):
    ax3.plot(dfv.values[:, ii], dfv.index, color=colors[ii], linewidth=2)
    ax4.plot(dfj.values[:, ii], dfj.index, color=colors[ii], label=labels[ii], linewidth=2)
ax3.set_xlim(-1, 1)
ax4.set_xlim(-1, 1)
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 1)
ax3.set_xlabel(r"$\tau_{w,x} \tau_{wm,x}^{-1}$")
ax3.set_ylabel(r"$(z - \delta_c) \delta_w^{-1}$")
ax3.set_title("(c)")
ax4.set_title("(d)")
ax4.set_xlabel(r"$\tau_{w,x} \tau_{wm,x}^{-1}$")
ax4.set_ylabel(r"               $z \delta_w^{-1}$")

fig.tight_layout(pad=0.5)
plt.savefig("files/wave_stress_comparison.png", dpi=300)
plt.show()
