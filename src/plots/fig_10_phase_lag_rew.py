import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
data_path = os.path.join(get_project_root(), "data/vectrino_15deg.npy")
data = np.load(data_path, allow_pickle=True).item()
phase_vec = data["phase"]
z_vec = data["z"] - 0.004
z_tau = (z_vec > -0.0015) & (z_vec < 0.0015)
z_Ri = (z_vec > -0.0015) & (z_vec < 0.0015)
z_u = z_vec > 0.0095

# Velocity  normalization and vertical average
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)
u_mean_vec = np.nanmean(u[z_u, :, :], axis=0)

# Stress normalization and vertical average
tau_wave_max = np.nanmax(np.nanmean(data["tau_wave_maj"][z_tau, :, :], axis=0), axis=1)
tau_mean_vec = np.nanmean(data["tau_wave_maj"][z_tau, :, :] / tau_wave_max.reshape(1, -1, 1), axis=0)

# %% Calculating phase lag between velocity and stress
phase_hd = np.linspace(phase_vec[0], phase_vec[-1], 1000)
phase_lag = np.zeros((384,)) * np.nan
for ii in range(u_mean_vec.shape[0]):
    try:
        f_u = interp1d(phase_vec, u_mean_vec[ii, :], kind="cubic")
        f_tau = interp1d(phase_vec, tau_mean_vec[ii, :], kind="cubic")
        u_hd = f_u(phase_hd)
        tau_hd = f_tau(phase_hd)
        idx_u_max = np.nanargmax(u_hd)
        phase_u_max = phase_hd[idx_u_max]
        idx_tau_max = np.nanargmax(tau_hd)
        delta_phi = phase_u_max - phase_hd[idx_tau_max]
        phase_lag[ii] = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    except ValueError:
        continue

# %% Stratification parameters
rho_0 = 1020
rho = rho_0 + data["ssc"]
drho_dz = np.gradient(rho, data["z"], axis=0)
N2 = np.abs((9.81 / rho_0) * drho_dz)
dudz = np.gradient(data["u_wave"], data["z"], axis=0)
dvdz = np.gradient(data["v_wave"], data["z"], axis=0)
S2 = dudz**2 + dvdz**2
Ri = N2 / S2
Ri[np.isinf(Ri)] = np.nan
Ri_mean = np.nanmean(Ri[z_Ri, :, 0], axis=0)

# %% Plotting as a function of wave reynolds number
time_mask = (data["ubr"] > 0.015) & (data["omega"] > 1)
Re = (data["ubr"] ** 2 / data["omega"]) / 1e-6
fig, ax = plt.subplots()
sc = ax.scatter(
    Re[time_mask],
    np.degrees(phase_lag[time_mask]),
    c=np.log10(Ri_mean[time_mask]),
    marker="o",
    cmap="copper",
    vmin=-2,
    vmax=2,
)

cb = fig.colorbar(sc, ax=ax)
cb.set_label(r"$\log_{10}(Ri)$")
ax.hlines(y=45, xmin=5e1, xmax=1e4, linestyle="--", color="0.5")
ax.hlines(y=-45, xmin=5e1, xmax=1e4, linestyle="--", color="0.5")
ax.hlines(y=0, xmin=5e1, xmax=1e4, linestyle="--", color="0.5")
ax.set_xlim(5e1, 1e4)
ax.set_xscale("log")
ax.set_xlabel(r"$Re_w$")
ax.set_ylabel(r"$\Delta \theta$")
ax.set_yticks(np.arange(-180, 181, 90))
ax.set_yticklabels([f"${int(np.round(angle))}^\\circ$" for angle in np.arange(-180, 181, 90)])
fig.set_size_inches(8, 6)
fig.tight_layout(pad=0.5)
plt.savefig("files/phase_lag_rew.png", dpi=300)
plt.show()
