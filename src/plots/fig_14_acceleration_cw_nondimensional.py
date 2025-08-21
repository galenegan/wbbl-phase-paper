import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
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

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "carbon_blue", ["#E5F6FF", "#005D5D"]
)

# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
phase_vec = data["phase"]
phase_plot = np.degrees(np.concatenate((phase_vec, [-phase_vec[0]])))
z_vec = data["z"]
zavg_u = z_vec > 0.013
zavg_tau = (z_vec > 0.0025) & (z_vec < 0.0055)
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
z_canopy = z_vec <= 0.004
tau_wave_max = np.nanmax(data["tau_wc_maj"][zavg_tau, :, :][:, idx, :], axis=(0, 2))
z_above = z_vec > 0.004
rho_s = 1300
rho = 1020
ustar = np.sqrt(tau_wave_max / rho)
g = 9.81
# %% Within the canopy
# Acceleration
dudt = data["dudt_wave"][:, idx, :][0, :, :]
dudt_max_idx = np.nanargmax(dudt, axis=1)
dudt_max = np.nanmax(dudt, axis=1)

S_max = rho * dudt_max / ((rho_s - rho) * g)

# Sediment flux
cpwp = data["cpwp"][:, idx, :][z_canopy, :, :]
cpwp_max = np.zeros_like(dudt_max)
for ii in range(len(dudt_max_idx)):
    cpwp_max[ii] = np.nanmax(cpwp[:, ii, dudt_max_idx[ii]], axis=0)

cpwp_max = cpwp_max / (rho_s * ustar)

# %% Above the canopy
# Calculate dudt_min and cpwp_max_above
dudt_min_idx = np.nanargmin(dudt, axis=1)
dudt_min = np.abs(np.nanmin(dudt, axis=1))

# Sediment flux above canopy
z_above = z_vec > 0.004
cpwp_above = data["cpwp"][:, idx, :][z_above, :, :]
cpwp_max_above = np.zeros_like(dudt_min)
for ii in range(len(dudt_min_idx)):
    cpwp_max_above[ii] = np.nanmax(cpwp_above[:, ii, dudt_min_idx[ii]], axis=0)

S_min = rho * dudt_min / ((rho_s - rho) * g)
cpwp_max_above = cpwp_max_above / (rho_s * ustar)

# %% Create 1x2 subplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

scatter1 = ax1.scatter(S_max, cpwp_max, s=50, color="0.7", alpha=0.3)

# Calculate bin averages for left panel
n_bins = 10
S_max_bins = np.linspace(np.nanmin(S_max), np.nanmax(S_max), n_bins + 1)
bin_centers_max = (S_max_bins[:-1] + S_max_bins[1:]) / 2
cpwp_bin_avg_max = np.zeros(n_bins)
cpwp_bin_std_max = np.zeros(n_bins)

for i in range(n_bins):
    mask = (S_max >= S_max_bins[i]) & (S_max < S_max_bins[i + 1])
    if np.sum(mask) > 0:
        cpwp_bin_avg_max[i] = np.nanmean(cpwp_max[mask])
        cpwp_bin_std_max[i] = np.nanstd(cpwp_max[mask])
    else:
        cpwp_bin_avg_max[i] = np.nan
        cpwp_bin_std_max[i] = np.nan

ax1.errorbar(
    bin_centers_max,
    cpwp_bin_avg_max,
    yerr=cpwp_bin_std_max,
    fmt="o",
    markersize=10,
    color="0.4",
    capsize=3,
    capthick=1,
)

# Format left panel
ax1.set_xlabel(r"$S$")
ax1.set_ylabel(r"$\overline{c^\prime w^\prime} \left(\rho_s u_*\right)^{-1}$")
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style="scientific", axis="both", scilimits=(-2, 2))
ax1.set_title("(a)")
ax1.set_ylim(-0.2e-4, 1.2e-4)

# Right panel
scatter2 = ax2.scatter(S_min, cpwp_max_above, s=50, color="0.7", alpha=0.3)

# Calculate bin averages for right panel
S_min_bins = np.linspace(np.nanmin(S_min), np.nanmax(S_min), n_bins + 1)
bin_centers_min = (S_min_bins[:-1] + S_min_bins[1:]) / 2
cpwp_bin_avg_above = np.zeros(n_bins)
cpwp_bin_std_above = np.zeros(n_bins)

for i in range(n_bins):
    mask = (S_min >= S_min_bins[i]) & (S_min < S_min_bins[i + 1])
    if np.sum(mask) > 0:
        cpwp_bin_avg_above[i] = np.nanmean(cpwp_max_above[mask])
        cpwp_bin_std_above[i] = np.nanstd(cpwp_max_above[mask])
    else:
        cpwp_bin_avg_above[i] = np.nan
        cpwp_bin_std_above[i] = np.nan

ax2.errorbar(
    bin_centers_min,
    cpwp_bin_avg_above,
    yerr=cpwp_bin_std_above,
    fmt="o",
    markersize=10,
    color="0.4",
    capsize=3,
    capthick=1,
)

# Format right panel
ax2.set_xlabel(r"$|S|$")
ax2.set_ylabel(r"$\overline{c^\prime w^\prime} \left(\rho_s u_*\right)^{-1}$")
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style="scientific", axis="both", scilimits=(-2, 2))
ax2.set_title("(b)")
ax2.set_ylim(-0.2e-4, 1.2e-4)
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.savefig("plots/S_cw.png", dpi=300)
plt.show()
