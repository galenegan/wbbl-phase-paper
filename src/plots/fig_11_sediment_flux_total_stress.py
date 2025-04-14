import numpy as np
import matplotlib.pyplot as plt
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

# %% Vectrino data
data_path = os.path.join(get_project_root(), "data/vectrino.npy")
data = np.load(data_path, allow_pickle=True).item()
phase_vec = data["phase"]
phase_plot = np.degrees(np.concatenate((phase_vec, [-phase_vec[0]])))
z_vec = data["z"]
zavg_u = z_vec > 0.013
zavg_tau = (z_vec > 0.0025) & (z_vec < 0.0055)
posidx = (np.nanmean(data["u_wc"], axis=(0, 2)) > 0) & (data["ubr"] > 0.015) & (data["omega"] > 1)
negidx = (np.nanmean(data["u_wc"], axis=(0, 2)) < 0) & (data["ubr"] > 0.015) & (data["omega"] > 1)

# %% Positive index
# Sediment flux
cpwp = np.nanmean(data["cpwp"][:, posidx, :], axis=1)
cpwp_plot = np.concatenate((cpwp, cpwp[:, 0].reshape(-1, 1)), axis=1)

# Velocity
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)

u_mean = np.nanmean(u[:, posidx, :][zavg_u, :, :], axis=(0, 1))
u_mean_plot = np.concatenate((u_mean, [u_mean[0]]))

# Acceleration
dudt = data["dudt_wave"] / data["ubr"].reshape(1, -1, 1)
dudt_mean = np.nanmean(dudt[:, posidx, :][zavg_u, :, :], axis=(0, 1))
dudt_mean_plot = np.concatenate((dudt_mean, [dudt_mean[0]]))

tau_wave_max = np.nanmax(data["tau_wc_maj"][zavg_tau, :, :][:, posidx, :], axis=(0, 2))
tau_mean_vec = np.nanmean(
    data["tau_wc_maj"][zavg_tau, :, :][:, posidx, :] / tau_wave_max.reshape(1, -1, 1), axis=(0, 1)
)
tau_mean_vec_plot = np.concatenate((tau_mean_vec, [tau_mean_vec[0]]))
# %% Plotting
P, Z = np.meshgrid(phase_plot, 100 * z_vec)
fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True, figsize=(20, 16), layout="constrained")
ax1.plot(phase_plot, tau_mean_vec_plot, "-", linewidth=2.5, color="0.0")
ax1.set_ylabel(r"$\langle \tau_{w,x} \rangle \tau_{wm,x}^{-1}$")
ax1.hlines(0, xmin=-180, xmax=180, color="0.0", linestyle="--")
ax1.set_title("(a)")
ax2.plot(phase_plot, u_mean_plot, "-", linewidth=2.5, color="0.0")
ax2.hlines(0, xmin=-180, xmax=180, color="0.0", linestyle="--")
ax22 = ax2.twinx()
ax22.plot(phase_plot, dudt_mean_plot, "-", linewidth=2.5, color="0.6")
ax2.set_ylabel(r"$\tilde{u}_p u_b^{-1}$")
ax22.set_ylabel(r"$\partial_t \tilde{u}_p u_b^{-1}$ (s$^{-1}$)", color="0.4")
[t.set_color("0.4") for t in ax22.yaxis.get_ticklines()]
[t.set_color("0.4") for t in ax22.yaxis.get_ticklabels()]
ax2.set_title("(b)")

levels = np.linspace(-2.5e-4, 2.5e-4, 11)
pc = ax3.contourf(P, Z, cpwp_plot, vmin=-2.5e-4, vmax=2.5e-4, levels=levels, cmap="RdBu")
cb = fig.colorbar(pc, ax=ax3, pad=-0.2)
pc.set_clim(-2.5e-4, 2.5e-4)
cb.set_label(r"$\overline{c^\prime w^\prime}$ (g m$^{-2}$ s$^{-1}$)")

# Set scientific format on colorbar ticks
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # Force sci notation outside this range
cb.ax.yaxis.set_major_formatter(formatter)

ax3.set_xticks(phase_plot)
ax3.set_xticklabels([f"${int(angle)}^\\circ$" for angle in phase_plot])
ax3.set_xlabel(r"$\theta$")
ax3.set_ylabel(r"$z$ (cm)")
ax3.set_title("(c)")


# %% Negative index
# Sediment flux
cpwp = np.nanmean(data["cpwp"][:, negidx, :], axis=1)
cpwp_plot = np.concatenate((cpwp, cpwp[:, 0].reshape(-1, 1)), axis=1)

# Velocity
u = data["u_wave"] / data["ubr"].reshape(1, -1, 1)

u_mean = np.nanmean(u[:, negidx, :][zavg_u, :, :], axis=(0, 1))
u_mean_plot = np.concatenate((u_mean, [u_mean[0]]))

# Acceleration
dudt = data["dudt_wave"] / data["ubr"].reshape(1, -1, 1)
dudt_mean = np.nanmean(dudt[:, negidx, :][zavg_u, :, :], axis=(0, 1))
dudt_mean_plot = np.concatenate((dudt_mean, [dudt_mean[0]]))

tau_wave_max = np.nanmax(data["tau_wc_maj"][zavg_tau, :, :][:, negidx, :], axis=(0, 2))
tau_mean_vec = np.nanmean(
    data["tau_wc_maj"][zavg_tau, :, :][:, negidx, :] / tau_wave_max.reshape(1, -1, 1), axis=(0, 1)
)
tau_mean_vec_plot = np.concatenate((tau_mean_vec, [tau_mean_vec[0]]))
# %% Plotting
ax4.plot(phase_plot, tau_mean_vec_plot, "-", linewidth=2.5, color="0.0")
ax4.set_ylabel(r"$\langle \tau_{w,x} \rangle \tau_{wm,x}^{-1}$")
ax4.hlines(0, xmin=-180, xmax=180, color="0.0", linestyle="--")
ax4.set_title("(d)")
ax5.plot(phase_plot, u_mean_plot, "-", linewidth=2.5, color="0.0")
ax5.hlines(0, xmin=-180, xmax=180, color="0.0", linestyle="--")
ax52 = ax5.twinx()
ax52.plot(phase_plot, dudt_mean_plot, "-", linewidth=2.5, color="0.6")
ax5.set_ylabel(r"$\tilde{u}_p u_b^{-1}$")
ax52.set_ylabel(r"$\partial_t \tilde{u}_p u_b^{-1}$ (s$^{-1}$)", color="0.4")
[t.set_color("0.4") for t in ax52.yaxis.get_ticklines()]
[t.set_color("0.4") for t in ax52.yaxis.get_ticklabels()]
ax5.set_title("(e)")

levels = np.linspace(-2.5e-4, 2.5e-4, 11)
pc = ax6.contourf(P, Z, cpwp_plot, vmin=-2.5e-4, vmax=2.5e-4, levels=levels, cmap="RdBu")
cb = fig.colorbar(pc, ax=ax6, pad=-0.25)
pc.set_clim(-2.5e-4, 2.5e-4)
cb.set_label(r"$\overline{c^\prime w^\prime}$ (g m$^{-2}$ s$^{-1}$)")

# Set scientific format on colorbar ticks
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # Force sci notation outside this range
cb.ax.yaxis.set_major_formatter(formatter)

ax6.set_xticks(phase_plot)
ax6.set_xticklabels([f"${int(angle)}^\\circ$" for angle in phase_plot])
ax6.set_xlabel(r"$\theta$")
ax6.set_ylabel(r"$z$ (cm)")
ax6.set_title("(f)")

# plt.savefig("files/sediment_flux_sep.png", dpi=300)
plt.show()
