import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, curve_fit
import os
from src.utils.project_utils import get_project_root

params = {
    "axes.labelsize": 28,
    "font.size": 28,
    "legend.fontsize": 16,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()

kn = 0.00645
z_vec = data["z"].squeeze()
z_tau = (z_vec > 0.0025) & (z_vec < 0.0055)

tau_max = np.nanmax(np.nanmean(data["tau_wave_total"][z_tau, :, :], axis=0), axis=1)
fw_vec = 2 * tau_max / (1020 * data["ubr"] ** 2)

ab_kb = (data["ubr"] / data["omega"]) / kn
ab_kb_lin = np.linspace(np.nanmin(ab_kb), np.nanmax(ab_kb), 300)
mask = (data["ubr"] > 0.015) & (data["omega"] > 1)

# %% Grant and Madsen
a1 = 7.02
a2 = -0.078
a3 = -8.82
fw_gm = np.exp(a1 * ab_kb_lin**a2 + a3)

gm_error = np.sqrt(np.nanmean((np.exp(a1 * ab_kb[mask] ** a2 + a3) - fw_vec[mask]) ** 2))

# %% Nielsen parameterization
a1 = 5.5
a2 = -0.2
a3 = -6.3
fw_nielsen = np.exp(a1 * ab_kb_lin**a2 + a3)
nielsen_error = np.sqrt(np.nanmean((np.exp(a1 * ab_kb[mask] ** a2 + a3) - fw_vec[mask]) ** 2))
# %% Rogers
a1 = 5.213
a2 = -0.194
a3 = -5.977
fw_rogers = np.exp(a1 * ab_kb_lin**a2 + a3)
fw_rogers[ab_kb_lin < 0.0369] = 50
rogers_error = np.sqrt(np.nanmean((np.exp(a1 * ab_kb[mask] ** a2 + a3) - fw_vec[mask]) ** 2))


# %% Jonsson
def jfunc(fw, ab_kb):
    return 1 / (4 * np.sqrt(fw)) + np.log10(1 / (4 * np.sqrt(fw))) + 0.08 - np.log10(ab_kb)


fw_jonsson = np.zeros((len(ab_kb_lin),))
for ii in range(len(ab_kb_lin)):
    res = root(jfunc, x0=fw_nielsen[ii], args=(ab_kb_lin[ii]))
    fw_jonsson[ii] = min(res.x, 0.3)

fw_jonsson_error = np.zeros((len(ab_kb[mask])))
for ii in range(len(ab_kb[mask])):
    res = root(jfunc, x0=fw_vec[mask][ii], args=ab_kb[mask][ii])
    fw_jonsson_error[ii] = min(res.x, 0.3)

jonsson_error = np.sqrt(np.nanmean((fw_jonsson_error - fw_vec[mask]) ** 2))


# %% kamphuis
def kfunc(fw, ab_kb):
    return 1 / (4 * np.sqrt(fw)) + np.log10(1 / (4 * np.sqrt(fw))) + 0.35 - (4 / 3) * np.log10(ab_kb)


fw_kamphuis = np.zeros((len(ab_kb_lin),))

for ii in range(len(ab_kb_lin)):
    res = root(kfunc, x0=fw_nielsen[ii], args=(ab_kb_lin[ii]))
    fw_kamphuis[ii] = res.x

fw_kamphuis_error = np.zeros((len(ab_kb[mask])))
for ii in range(len(ab_kb[mask])):
    res = root(kfunc, x0=fw_vec[mask][ii], args=ab_kb[mask][ii])
    fw_kamphuis_error[ii] = res.x

kamphuis_error = np.sqrt(np.nanmean((fw_kamphuis_error - fw_vec[mask]) ** 2))


# %% Gon et al
def gon_func(ab_kb, alpha):
    return 1.94 * (ab_kb / alpha) ** -0.97


popt, pcov = curve_fit(gon_func, xdata=ab_kb[mask], ydata=fw_vec[mask], p0=(1,), maxfev=10000)

fw_gon_for_error = gon_func(ab_kb[mask], *popt)
gon_error = np.sqrt(np.nanmean((fw_gon_for_error - fw_vec[mask]) ** 2))

fw_gon = gon_func(ab_kb_lin, *popt)

# %% Laminar
ab = data["ubr"] / data["omega"]
Re_w = ab**2 * data["omega"] / 1e-6
fw_laminar = 2 / np.sqrt(Re_w)


def lam_fit(x, a1, a2, a3):
    return np.exp(a1 * x**a2 + a3)


popt_lam, pcov_lam = curve_fit(lam_fit, xdata=ab_kb[mask], ydata=fw_laminar[mask], p0=(1, -1, -1), maxfev=10000)
laminar_error = np.sqrt(np.nanmean((fw_laminar[mask] - fw_vec[mask]) ** 2))


# %% Bin averaging function
def bin_average_data(x_data, y_data, n_bins=11):
    """Bin average y_data in bins of x_data and return bin centers, means, and std devs"""
    log_x_min, log_x_max = np.log10(np.nanmin(x_data)), np.log10(np.nanmax(x_data))
    bin_edges = np.logspace(log_x_min, log_x_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask_bin = (x_data >= bin_edges[i]) & (x_data < bin_edges[i + 1])
        if np.sum(mask_bin) > 0:
            bin_means[i] = np.nanmean(y_data[mask_bin])
            bin_stds[i] = np.nanstd(y_data[mask_bin])

    return bin_centers, bin_means, bin_stds


# Calculate bin-averaged data
bin_centers, bin_means, bin_stds = bin_average_data(ab_kb[mask], fw_vec[mask])

# %% Plot
colors = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c"]

fig, ax = plt.subplots()
ax.loglog(ab_kb[mask], fw_vec[mask], "o", color="0.7", alpha=0.3)
ax.errorbar(
    bin_centers, bin_means, yerr=bin_stds, fmt="o", markersize=10, color="#012749",
    ecolor="#012749", alpha=0.8, capsize=3, capthick=1
)
ax.loglog(ab_kb_lin, fw_gon, "-", linewidth=2, color="#8a3800", label=f"Gon 2020, RMSE = {gon_error:.3f}")
ax.loglog(ab_kb_lin, fw_rogers, ":", linewidth=4, color=colors[1], label=f"Rogers 2016, RMSE = {rogers_error:.3f}")
ax.loglog(ab_kb_lin, fw_nielsen, "--", linewidth=2, color=colors[0], label=f"Nielsen 1992, RMSE = {nielsen_error:.3f}")
ax.loglog(ab_kb_lin, fw_jonsson, "-", linewidth=3, color=colors[4], label=f"Jonsson 1966, RMSE = {jonsson_error:.3f}")
ax.loglog(
    ab_kb_lin, fw_kamphuis, "--", linewidth=3, color=colors[2], label=f"Kamphuis 1975, RMSE = {kamphuis_error:.2f}"
)
ax.loglog(
    ab_kb_lin,
    lam_fit(ab_kb_lin, *popt_lam),
    ":",
    linewidth=4,
    color=colors[6],
    label=f"Laminar, RMSE = {laminar_error:.3f}",
)

ax.loglog(ab_kb_lin, fw_gm, "-.", linewidth=4, color=colors[3], label=f"GM 1979, RMSE = {gm_error:.3f}")
ax.set_xlabel(r"$a_b k_b^{-1}$")
ax.set_xlim(0.5, 11)
ax.set_ylim(0.02, 10)
ax.set_ylabel(r"$f_w$")
ax.legend(handlelength=4)
plt.rcParams.update(params)
fig.set_size_inches(12, 8)
fig.tight_layout(pad=0.5)
plt.savefig("plots/wave_friction.png", dpi=300)
plt.show()
