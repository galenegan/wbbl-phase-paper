import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.project_utils import get_project_root

params = {
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
phase_vec = np.degrees(data["phase"]).astype(int)
z_vec = data["z"] * 100
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
u_mean = np.nanmean(data["u_wave"][:, idx, :], axis=1) * 100
v_mean = np.nanmean(data["v_wave"][:, idx, :], axis=1) * 100
labels = [str(label) + r"$^\circ$" for label in phase_vec]
colors = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c"]
# %% Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for ii in range(u_mean.shape[1]):
    ax1.plot(u_mean[:, ii], z_vec, color=colors[ii], linewidth=2)
    ax2.plot(v_mean[:, ii], z_vec, color=colors[ii], label=labels[ii], linewidth=2)

ax1.axhspan(0.3, 0.5, color="0.5", alpha=0.5, lw=0)
ax2.axhspan(0.3, 0.5, color="0.5", alpha=0.5, lw=0)
ax1.set_xlabel(r"$\tilde{u}$ (cm/s)")
ax1.set_ylabel(r"$z$ (cm)")
ax1.set_title("(a)")
ax2.set_title("(b)")
ax2.set_xlabel(r"$\tilde{v}$ (cm/s)")
ax2.set_ylabel(r"$z$ (cm)")
ax2.legend(bbox_to_anchor=(1.01, 1.025))
fig.tight_layout(pad=0.5)
plt.savefig("files/ubar.png", dpi=300)
plt.show()
