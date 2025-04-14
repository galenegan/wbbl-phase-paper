import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.crspy import m94
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
    tau_wc_vec[ii] = np.nanmax(np.nanmean(data["tau_wc_total"][z_tau, ii, :], axis=0))


# Bad vals
bad_vals = (tau_wc_gm > 10) | (tau_wc_vec > 1) | (data["ubr"] < 0.015) | (data["omega"] < 1)
tau_wc_gm[bad_vals] = np.nan
tau_wc_styles[bad_vals] = np.nan
tau_wc_vec[bad_vals] = np.nan

# %% Binning
df = pd.DataFrame()
df["residual_gm"] = tau_wc_gm - tau_wc_vec
df["ub"] = data["ubr"] * 100

ub_bins = np.arange(0, 12, 1)
df["ub_bins"] = pd.cut(df["ub"], ub_bins)

res_ub_mean = df.groupby("ub_bins")["residual_gm"].mean()
res_ub_std = df.groupby("ub_bins")["residual_gm"].std()
x_mids_ub = [i.mid for i in res_ub_mean.index]

# %% Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    x_mids_ub,
    res_ub_mean,
    yerr=res_ub_std,
    fmt="o-",
    color="#012749",
    ecolor="#012749",
    capsize=3,
    linewidth=2.5,
    alpha=0.7,
)
ax.set_xlabel(r"$u_b$ (cm/s)")
ax.set_ylabel(r"$|\boldsymbol{\tau_m}|$ Residual (Pa)")
fig.tight_layout(pad=0.5)
plt.savefig("files/gm_residuals.png", dpi=300)
plt.show()
