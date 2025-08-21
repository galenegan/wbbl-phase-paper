import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.utils.project_utils import get_project_root
from scipy.interpolate import interp1d
from scipy.optimize import root, curve_fit
from sklearn.metrics import r2_score

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


def displacement_thickness(ubar, z):
    """Calculates a modified displacement thickness for phase resolved boundary layer
    velocity profiles.

    """

    int_range = z > 0  # keep it above the bed
    z_int = np.flipud(z[int_range])  # Flipping for the integral
    uprof_int = np.abs(np.flipud(ubar[int_range]))

    umax = np.nanmax(uprof_int)  # Finding maximum velocity, wherever it may be
    idxmax = np.nanargmax(uprof_int)
    delta = np.trapz(1 - uprof_int[:idxmax] / umax, z_int[:idxmax])

    return delta


# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
ab = data["ubr"] / data["omega"]
kn = 0.00645
ab_kb = ab / kn
z_vec = data["z"] - 0.004
idx = (data["ubr"] > 0.015) & (data["omega"] > 1)
zidx_tau = (z_vec > -0.0015) & (z_vec < 0.0015)
ustar_wave = np.sqrt(np.nanmax(np.nanmean(data["tau_wave_total"][zidx_tau, :, :], axis=0), axis=1) / 1020)
omega = data["omega"]
delta_scaling_vec = ustar_wave / omega
delta_99 = np.zeros((384,)) * np.nan
delta_theta = np.zeros((384,)) * np.nan
for ii in range(len(delta_99)):
    if not idx[ii]:
        continue
    ubar = data["u_wave"][:, ii, :]
    max_phase = np.argmax(np.nanmean(np.abs(ubar), axis=0))

    # Displacement thickness
    delta_theta[ii] = 2 * displacement_thickness(ubar[:, max_phase], z_vec)

ab_kb_vec = ab_kb
bl_scale_factor_vec = delta_theta / delta_scaling_vec
# %% Jonsson data test1
phase_j = np.arange(0.0, 350, 15) * np.pi / 180.0 - np.pi
z_j = np.array([23, 20, 17, 14, 11, 9, 7, 5, 4, 3, 2, 1.5, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
dfj = pd.read_excel(os.path.join(get_project_root(), "data/jonsson_velocity_test_1.xlsx"), header=None, names=phase_j)
dfj = dfj.set_index(z_j)
ubar = dfj.values
max_phase = np.argmax(np.nanmean(np.abs(dfj.values), axis=0))

# Displacement thickness
delta_theta_j1 = 2 * displacement_thickness(ubar[:, max_phase], z_j)
ustar_j1 = np.sqrt(465)
omega_j1 = 2 * np.pi / 8.39
delta_scale_j1 = ustar_j1 / omega_j1
ab_kb_j1 = 285 / 2.3
# %% Jonsson data test2
ubar = np.array([155, 154, 158, 162, 170, 172.00, 155.00, 140.00, 121, 112, 98, 76, 58, 46])
z = np.array([20, 17, 14, 11, 8, 6, 4, 3, 2, 1.5, 1.0, 0.5, 0.2, 0.0])
delta_theta_j2 = 2 * displacement_thickness(ubar, z)
ustar_j2 = np.sqrt(475)
omega_j2 = 2 * np.pi / 7.20
delta_scale_j2 = ustar_j2 / omega_j2
ab_kb_j2 = 179 / 6.3

ab_kb_j = np.array([ab_kb_j1, ab_kb_j2])
bl_scale_factor_j = np.array([delta_theta_j1 / delta_scale_j1, delta_theta_j2 / delta_scale_j2])

# %% Binning vectrino data
df = pd.DataFrame({"ab_kb": ab_kb_vec[idx], "bl_scale": bl_scale_factor_vec[idx]})
df = df.dropna()


bin_edges = np.logspace(np.log10(np.nanmin(df["ab_kb"])), np.log10(np.nanmax(df["ab_kb"])), 11)
bins = pd.cut(df["ab_kb"], bins=bin_edges)

# Calculate mean and std for each bin
binned_stats = df.groupby(bins)["bl_scale"].agg(["mean", "std"]).reset_index()

# Get the bin centers for plotting
binned_stats["bin_center"] = binned_stats["ab_kb"].apply(lambda x: x.mid)


# %% Fitting a function to our data
def fit_func(x, a1, a2, a3):
    return np.exp(a1 * x**a2 + a3)


xdata = np.concatenate((binned_stats["bin_center"].values, ab_kb_j))
ydata = np.concatenate((binned_stats["mean"].values, bl_scale_factor_j))

popt, pcov = curve_fit(fit_func, xdata, ydata, p0=(2, -0.1, -2))
xlin = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 100)
yfit = fit_func(xlin, *popt)
r2 = r2_score(ydata, fit_func(xdata, *popt))


# %% Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    binned_stats["bin_center"],
    binned_stats["mean"],
    yerr=binned_stats["std"],
    fmt="o-",
    color="#012749",
    ecolor="#012749",
    capsize=3,
    linewidth=2.5,
    alpha=0.7,
    label="Vectrino Field",
)
ax.plot(ab_kb_j, bl_scale_factor_j, "o", color="0.0", markersize=10, label="JC Lab")
a1 = np.round(popt[0], 2)
a2 = np.round(popt[1], 2)
a3 = np.round(popt[2], 2)
r2 = np.round(r2, 2)
ax.plot(
    xlin,
    yfit,
    color="r",
    linestyle="--",
    label=r"Fit = $\exp\left({} x^{{{}}} {}\right), \, r^2 = {}$".format(a1, a2, a3, r2),
)
ax.set_xscale("log")
# Get handles and labels
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]])

ax.set_xlabel(r"$a_b k_b^{-1}$")
ax.set_ylabel(r"$2 \delta_\theta \delta_w^{-1}$")
fig.tight_layout(pad=0.5)
plt.savefig("plots/bl_thickness.png", dpi=300)
plt.show()
