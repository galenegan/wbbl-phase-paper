import matplotlib.pyplot as plt
import numpy as np
from src.utils.crspy import m94
from scipy.io import loadmat
import os
from src.utils.project_utils import get_project_root

params = {
    "axes.labelsize": 28,
    "font.size": 28,
    "legend.fontsize": 18,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
}
plt.rcParams.update(params)

data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
sdata = loadmat(os.path.join(get_project_root(), "data/styles_stress.mat"), simplify_cells=True)["STRESS"]
num_bursts = 384
tau_wc_gm = np.zeros((num_bursts,)) * np.nan
tau_wc_styles = np.zeros((num_bursts,)) * np.nan
tau_wc_vec = np.zeros((num_bursts,)) * np.nan

z_vec = data["z"].squeeze()
z_tau = (z_vec > 0.0025) & (z_vec < 0.0055)

for ii in range(num_bursts):
    kn = 0.00645

    ustrc, ustrr, ustrwm, dwc, fwc, zoa = m94(
        ubr=data["ubr"][ii],
        wr=data["omega"][ii],
        ucr=data["ucr"][ii],
        zr=data["zr"][ii],
        phiwc=data["phiwc"][ii] * np.pi / 180,
        kN=kn,
    )

    tau_wc_gm[ii] = 1020 * ustrr**2
    tau_wc_styles[ii] = 1020 * (sdata[ii]["ustarcw"] / 100) ** 2
    tau_wc_vec[ii] = np.nanmax(np.nanmean(data["tau_wc_total"][z_tau, ii, :], axis=0))


# Bad vals
bad_vals = (tau_wc_gm > 10) | (tau_wc_vec > 1) | (data["ubr"] < 0.015) | (data["omega"] < 1)
tau_wc_gm[bad_vals] = np.nan
tau_wc_styles[bad_vals] = np.nan
tau_wc_vec[bad_vals] = np.nan

gm_error = np.sqrt(np.nanmean((tau_wc_gm - tau_wc_vec) ** 2))
styles_error = np.sqrt(np.nanmean((tau_wc_styles - tau_wc_vec) ** 2))
gm_bias = np.nanmean(tau_wc_gm - tau_wc_vec)
styles_bias = np.nanmean(tau_wc_styles - tau_wc_vec)

# %% Plots
# 1:1 line (reference)
x = tau_wc_gm
y = tau_wc_vec
lims = [np.nanmin([np.nanmin(x), np.nanmin(y)]), np.nanmax([np.nanmax(x), np.nanmax(y)])]

# Compute bias along bins defined by the 1:1 line
bins = np.linspace(lims[0], lims[1], 9)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_indices = np.digitize(x, bins)

# Calculate bias within each bin
bias = [
    np.nanmean(y[bin_indices == i] - x[bin_indices == i]) if np.any(bin_indices == i) else np.nan
    for i in range(1, len(bins))
]
bias = np.array(bias)

# Plot bias-corrected line (1:1 + bias)
one = np.linspace(np.nanmin(tau_wc_vec), np.nanmax(tau_wc_vec), 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ax1.plot(tau_wc_gm, tau_wc_vec, "o", color="0.5", alpha=0.7)
ax1.plot(one, one, "-", color="0.0", linewidth=2, label=f"RMSE = {gm_error:.3f} Pa")
ax1.plot(bin_centers, bin_centers + bias, color="r", alpha=0.7, linewidth=2, label=f"Bias = {gm_bias:.3f} Pa")
ax1.set_xlabel(r"GM $|\boldsymbol{\tau_m}|$ (Pa)")
ax1.set_ylabel(r"Measured $|\boldsymbol{\tau_m}|$ (Pa)")
ax1.set_title("(a)")
ax1.legend()

x = tau_wc_styles
y = tau_wc_vec
lims = [np.nanmin([np.nanmin(x), np.nanmin(y)]), np.nanmax([np.nanmax(x), np.nanmax(y)])]

# Compute bias along bins defined by the 1:1 line
bins = np.linspace(lims[0], lims[1], 9)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_indices = np.digitize(x, bins)

# Calculate bias within each bin
bias = [
    np.nanmean(y[bin_indices == i] - x[bin_indices == i]) if np.any(bin_indices == i) else np.nan
    for i in range(1, len(bins))
]
bias = np.array(bias)

ax2.plot(tau_wc_styles, tau_wc_vec, "o", color="0.5", alpha=0.7)
ax2.plot(one, one, "-", color="0.0", linewidth=2, label=f"RMSE = {styles_error:.3f} Pa")
ax2.plot(bin_centers, bin_centers + bias, color="r", alpha=0.7, linewidth=2, label=f"Bias = {styles_bias:.3f} Pa")
ax2.set_xlabel(r"Styles $|\boldsymbol{\tau_m}|$ (Pa)")
ax2.set_ylabel(r"Measured $|\boldsymbol{\tau_m}|$ (Pa)")
ax2.set_title("(b)")
ax2.legend()

fig.tight_layout(pad=0.5)
plt.savefig("files/combined_stress.png", dpi=300)
plt.show()
