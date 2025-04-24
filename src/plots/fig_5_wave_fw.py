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

# %% Laminar
ab = data["ubr"] / data["omega"]
Re_w = ab**2 * data["omega"] / 1e-6
fw_laminar = 2 / np.sqrt(Re_w)


def lam_fit(x, a1, a2, a3):
    return np.exp(a1 * x**a2 + a3)


popt_lam, pcov_lam = curve_fit(lam_fit, xdata=ab_kb[mask], ydata=fw_laminar[mask], p0=(1, -1, -1), maxfev=10000)
laminar_error = np.sqrt(np.nanmean((fw_laminar[mask] - fw_vec[mask]) ** 2))
# %% Plot
colors = ["#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c"]

fig, ax = plt.subplots()
ax.loglog(ab_kb[mask], fw_vec[mask], "o", color="0.5", alpha=0.5)
ax.loglog(ab_kb_lin, fw_nielsen, "-", linewidth=2, color=colors[0], label=f"Nielsen 1992, RMSE = {nielsen_error:.2f}")
ax.loglog(ab_kb_lin, fw_rogers, ":", linewidth=4, color=colors[1], label=f"Rogers 2016, RMSE = {rogers_error:.2f}")
ax.loglog(ab_kb_lin, fw_jonsson, "--", linewidth=3, color=colors[4], label=f"Jonsson 1966, RMSE = {jonsson_error:.2f}")
ax.loglog(ab_kb_lin, fw_gm, "-.", linewidth=4, color=colors[3], label=f"GM 1979, RMSE = {gm_error:.2f}")
ax.loglog(
    ab_kb_lin,
    lam_fit(ab_kb_lin, *popt_lam),
    ":",
    linewidth=4,
    color=colors[6],
    label=f"Laminar, RMSE = {laminar_error:.2f}",
)
ax.loglog(
    ab_kb_lin, fw_kamphuis, "--", linewidth=3, color=colors[2], label=f"Kamphuis 1975, RMSE = {kamphuis_error:.2f}"
)
ax.set_xlabel(r"$a_b k_b^{-1}$")
ax.set_xlim(0.3, 11)
ax.set_ylim(0.02, 10)
ax.set_ylabel(r"$f_w$")
ax.legend()
plt.rcParams.update(params)
fig.set_size_inches(12, 6)
fig.tight_layout(pad=0.5)
# plt.savefig("files/wave_friction.png", dpi=300)
plt.show()
