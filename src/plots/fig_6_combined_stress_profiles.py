import numpy as np
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
z_tau = z_vec >= 0

# Normalization
posidx = (np.nanmean(data["u_wc"], axis=(0, 2)) > 0) & (data["ubr"] > 0.015) & (data["omega"] > 1)
negidx = (np.nanmean(data["u_wc"], axis=(0, 2)) < 0) & (data["ubr"] > 0.015) & (data["omega"] > 1)
tau_positive = np.nanmean(data["tau_wc_maj"][:, posidx, :], axis=1)
tau_negative = np.nanmean(data["tau_wc_maj"][:, negidx, :], axis=1)

# %% Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for ii in range(tau_positive.shape[1]):
    ax1.plot(tau_positive[z_tau, ii], z_vec[z_tau] * 100, color=colors[ii], linewidth=2)
    ax2.plot(tau_negative[z_tau, ii], z_vec[z_tau] * 100, color=colors[ii], label=labels[ii], linewidth=2)

ax1.vlines(x=0, ymin=-0.02, ymax=1.05, color="0.5", linestyle="--")
ax2.vlines(x=0, ymin=-0.02, ymax=1.05, color="0.5", linestyle="--")
ax1.set_ylim(-0.02, 1.05)
ax2.set_ylim(-0.02, 1.05)
ax1.set_xlim(-0.25, 0.25)
ax2.set_xlim(-0.25, 0.25)
ax1.set_xlabel(r"$\tau_x$ (Pa)")
ax1.set_ylabel(r"$z - \delta_c$ (cm)")
ax1.set_title(r"$\overline{u} > 0$")
ax2.set_title(r"$\overline{u} < 0$")
ax2.set_xlabel(r"$\tau_x$ (Pa)")
ax2.set_ylabel(r"$z - \delta_c$ (cm)")
ax2.legend(bbox_to_anchor=(1.01, 1.025))
fig.tight_layout(pad=0.5)
plt.savefig("files/combined_stress_profiles.png", dpi=300)
plt.show()
