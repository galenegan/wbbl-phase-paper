import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.interpolate import interp1d
import os
from src.utils.project_utils import get_project_root
from src.utils.vectrino_utils import velocity_decomp, pa_rotation

# location of vectrino .mat files
mat_filepath = "/path/to/mats"

inputs = np.load(os.path.join(get_project_root(), "inputs.npy"), allow_pickle=True).item()
burstnums = list(range(384))
fs = 64
thetamaj_summer = -28.4547
dphi = np.pi / 4  # Discretizing the phase
phasebins = np.arange(-np.pi, np.pi, dphi)
num_z = 14

# Main outputs
z_out = np.linspace(0.014, 0.001, num_z)
vel_wave_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
vel_wave_min = np.zeros((num_z, len(burstnums), len(phasebins)))
accel_wave_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
accel_wave_min = np.zeros((num_z, len(burstnums), len(phasebins)))
vel_wc_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
vel_wc_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wave_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wave_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wave_total = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wc_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wc_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_wc_total = np.zeros((num_z, len(burstnums), len(phasebins)))
cpwp = np.zeros((num_z, len(burstnums), len(phasebins)))
ssc = np.zeros((num_z, len(burstnums), len(phasebins)))

# Reynolds shear stresses
tau_rs_wave_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_wave_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_wave_total = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_wc_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_wc_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_wc_total = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_vt_maj = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_vt_min = np.zeros((num_z, len(burstnums), len(phasebins)))
tau_rs_vt_total = np.zeros((num_z, len(burstnums), len(phasebins)))


# Intermediate things
upwp_pot_all = np.zeros((len(burstnums),))
vpwp_pot_all = np.zeros((len(burstnums),))
upwp_cov_all = np.zeros((len(burstnums),))

# Stratification parameters
vke = np.zeros((num_z, len(burstnums), len(phasebins)))
rho_rms = np.zeros((num_z, len(burstnums), len(phasebins)))

# GM Inputs
ubr = np.zeros((len(burstnums),))
omega = np.zeros((len(burstnums),))
ucr = np.zeros((len(burstnums),))
phiwc = np.zeros((len(burstnums),))
zr = np.zeros((len(burstnums),))

for n in burstnums:
    try:
        vec = sio.loadmat(f"{mat_filepath}/vectrino_{n}.mat")

        # Velocity in different directions
        vel_maj, vel_min = vec["velmaj"], vec["velmin"]
        vel_z = (vec["velz1"] + vec["velz2"]) / 2
        veleast, velnorth = pa_rotation(vel_maj, vel_min, -thetamaj_summer)
        vel_maj, vel_min = veleast, velnorth

        p, m = np.shape(vel_maj)

        # Spectral wave-turbulence decomposition
        u_wave, v_wave, w_wave = velocity_decomp(vel_maj, vel_min, vel_z, fs=fs)
        u_bar = np.nanmean(vel_maj, axis=1, keepdims=True)
        v_bar = np.nanmean(vel_min, axis=1, keepdims=True)
        w_bar = np.nanmean(vel_z, axis=1, keepdims=True)
        u_prime = vel_maj - u_bar
        v_prime = vel_min - v_bar
        w_prime = vel_z - w_bar
        u_turb = u_prime - u_wave
        v_turb = v_prime - v_wave
        w_turb = w_prime - w_wave

        # Getting the sediment flux and ssc
        ssc_full = vec["SSC"]
        c_bar = np.nanmean(ssc_full, axis=1, keepdims=True)
        cpwp_full = (vec["SSC"] - c_bar) * w_prime

        # Getting things necessary to calculate the Pietrzak parameter
        rho_f = 1020 + ssc_full
        w_prime_squared = w_prime**2

        # Vertical coordinate
        z = vec["z"].squeeze()
        p_cutoff = 0.014
        pidx = z > p_cutoff  # Potential flow region

        # calculate analytic signal based on de-meaned and low pass filtered velocity
        hu = sig.hilbert(np.nanmean(u_wave[pidx, :], axis=0))

        # Phase based on analytic signal
        p = np.arctan2(hu.imag, hu.real)

        # Calculating potential velocity, stress in potential region
        upwp = np.nanmean(u_turb * w_turb, axis=1, keepdims=True)
        vpwp = np.nanmean(v_turb * w_turb, axis=1, keepdims=True)
        H = inputs["depth"][n]
        u_pot_wave = np.nanmean(u_wave[pidx, :], axis=0, keepdims=True)
        v_pot_wave = np.nanmean(v_wave[pidx, :], axis=0, keepdims=True)
        upwp_pot = np.nanmean(upwp[pidx, :], axis=0).squeeze()
        vpwp_pot = np.nanmean(vpwp[pidx, :], axis=0).squeeze()

        # Correlations for Reynolds stresses
        uw_wave = u_wave * w_wave
        vw_wave = v_wave * w_wave
        uw_wc = u_prime * w_prime
        vw_wc = v_prime * w_prime
        uw_turb = u_turb * w_turb
        vw_turb = v_turb * w_turb

        # GM inputs
        v_east_mean = np.nanmean(veleast[pidx, :])
        v_north_mean = np.nanmean(velnorth[pidx, :])
        ucr[n] = np.sqrt(v_east_mean**2 + v_north_mean**2)
        ubr[n] = np.sqrt(np.var(u_pot_wave) + np.var(v_pot_wave))
        current_direction = 180 * np.arctan2(v_north_mean, v_east_mean) / np.pi
        wave_cur_direction = current_direction - inputs["wavedir"][n]
        phiwc[n] = (wave_cur_direction + 180) % 360 - 180
        zr[n] = np.nanmean(z[pidx])
        omega[n] = inputs["omega"][n]

        # and misc outputs
        upwp_pot_all[n] = upwp_pot
        vpwp_pot_all[n] = vpwp_pot
        upwp_cov_all[n] = np.cov(np.nanmean(vel_maj[pidx, :], axis=0), np.nanmean(vel_z[pidx, :], axis=0))[0, 1]

        # For combined shear stress
        ustar_squared = -upwp_pot / (1 - zr[n] / H)
        vstar_squared = -vpwp_pot / (1 - zr[n] / H)

        # Velocity integral
        zout_idx = (z > p_cutoff - num_z * 0.001) & (z < p_cutoff)
        z_burst = z[zout_idx]
        velint_wave_maj = np.zeros((len(z_burst), m))
        velint_wave_min = np.zeros((len(z_burst), m))
        velint_wc_maj = np.zeros((len(z_burst), m))
        velint_wc_min = np.zeros((len(z_burst), m))
        tvec = np.arange(0, m / fs, (1 / fs))
        for j in range(len(z_burst)):
            intidx = (z > p_cutoff - (j + 1) * 0.001) & (z < p_cutoff)  # Integration indices in BL
            uint_wc = u_prime[intidx, :]
            uint_wave = u_wave[intidx, :]
            vint_wc = v_prime[intidx, :]
            vint_wave = v_wave[intidx, :]
            zint = z[intidx]

            velint_wave_maj[j, :] = np.trapz(
                np.gradient(uint_wave - u_pot_wave, tvec, edge_order=2, axis=1), zint, axis=0
            )
            velint_wave_min[j, :] = np.trapz(
                np.gradient(vint_wave - v_pot_wave, tvec, edge_order=2, axis=1), zint, axis=0
            )

            velint_wc_maj[j, :] = np.trapz(
                np.gradient(uint_wave - u_pot_wave, tvec, edge_order=2, axis=1) - ustar_squared / H, zint, axis=0
            )

            velint_wc_min[j, :] = np.trapz(
                np.gradient(vint_wave - v_pot_wave, tvec, edge_order=2, axis=1) - vstar_squared / H, zint, axis=0
            )

        # Calculating gradient and shear stress
        tau_wave_maj_temp = 1020 * velint_wave_maj
        tau_wave_min_temp = 1020 * velint_wave_min
        tau_wc_maj_temp = 1020 * (ustar_squared * (1 - zr[n] / H) + velint_wc_maj)
        tau_wc_min_temp = 1020 * (vstar_squared * (1 - zr[n] / H) + velint_wc_min)

        # Wave velocity acceleration
        dudt = np.gradient(u_wave, tvec, edge_order=2, axis=1)
        dvdt = np.gradient(v_wave, tvec, edge_order=2, axis=1)

        for j in range(len(phasebins)):

            if j == 0:
                # For -pi
                idx1 = (p >= phasebins[-1] + (dphi / 2)) | (p <= phasebins[0] + (dphi / 2))  # Measured
            else:
                # For phases in the middle
                idx1 = (p >= phasebins[j] - (dphi / 2)) & (p <= phasebins[j] + (dphi / 2))  # measured

            # Wave stress
            f = interp1d(z_burst, np.nanmean(tau_wave_maj_temp[:, idx1], axis=1), fill_value="extrapolate")
            tau_wave_maj[:, n, j] = f(z_out)
            f = interp1d(z_burst, np.nanmean(tau_wave_min_temp[:, idx1], axis=1), fill_value="extrapolate")
            tau_wave_min[:, n, j] = f(z_out)
            tau_wave_total[:, n, j] = np.sqrt(tau_wave_maj[:, n, j] ** 2 + tau_wave_min[:, n, j] ** 2)

            # Total (WC) stress
            f = interp1d(z_burst, np.nanmean(tau_wc_maj_temp[:, idx1], axis=1), fill_value="extrapolate")
            tau_wc_maj[:, n, j] = f(z_out)
            f = interp1d(z_burst, np.nanmean(tau_wc_min_temp[:, idx1], axis=1), fill_value="extrapolate")
            tau_wc_min[:, n, j] = f(z_out)
            tau_wc_total[:, n, j] = np.sqrt(tau_wc_maj[:, n, j] ** 2 + tau_wc_min[:, n, j] ** 2)

            # RS-based stresses
            # Wave
            viscous_term = 1e-6 * np.gradient(np.nanmean(u_wave[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(uw_wave[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_wave_maj[:, n, j] = f(z_out)

            viscous_term = 1e-6 * np.gradient(np.nanmean(v_wave[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(vw_wave[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_wave_min[:, n, j] = f(z_out)
            tau_rs_wave_total[:, n, j] = np.sqrt(tau_rs_wave_maj[:, n, j] ** 2 + tau_rs_wave_min[:, n, j] ** 2)

            # WC
            viscous_term = 1e-6 * np.gradient(np.nanmean(u_prime[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(uw_wc[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_wc_maj[:, n, j] = f(z_out)

            viscous_term = 1e-6 * np.gradient(np.nanmean(v_prime[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(vw_wc[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_wc_min[:, n, j] = f(z_out)
            tau_rs_wc_total[:, n, j] = np.sqrt(tau_rs_wc_maj[:, n, j] ** 2 + tau_rs_wc_min[:, n, j] ** 2)

            # Viscous and turbulent
            viscous_term = 1e-6 * np.gradient(np.nanmean(u_prime[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(uw_turb[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_vt_maj[:, n, j] = f(z_out)

            viscous_term = 1e-6 * np.gradient(np.nanmean(v_prime[:, idx1], axis=1), z)[zout_idx]
            turb_term = np.nanmean(vw_turb[:, idx1], axis=1)[zout_idx]
            total_stress = 1020 * (viscous_term - turb_term)
            f = interp1d(z_burst, total_stress, fill_value="extrapolate")
            tau_rs_vt_min[:, n, j] = f(z_out)
            tau_rs_vt_total[:, n, j] = np.sqrt(tau_rs_vt_maj[:, n, j] ** 2 + tau_rs_vt_min[:, n, j] ** 2)

            # Wave velocity
            f = interp1d(z_burst, np.nanmean(u_wave[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            vel_wave_maj[:, n, j] = f(z_out)
            f = interp1d(z_burst, np.nanmean(v_wave[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            vel_wave_min[:, n, j] = f(z_out)

            # Wave acceleration
            f = interp1d(z_burst, np.nanmean(dudt[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            accel_wave_maj[:, n, j] = f(z_out)
            f = interp1d(z_burst, np.nanmean(dvdt[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            accel_wave_min[:, n, j] = f(z_out)

            # Total (WC) velocity
            f = interp1d(z_burst, np.nanmean(vel_maj[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            vel_wc_maj[:, n, j] = f(z_out)
            f = interp1d(z_burst, np.nanmean(vel_min[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            vel_wc_min[:, n, j] = f(z_out)

            # SSC and sediment flux
            f = interp1d(z_burst, np.nanmean(ssc_full[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            ssc[:, n, j] = f(z_out)

            f = interp1d(z_burst, np.nanmean(cpwp_full[:, idx1], axis=1)[zout_idx], fill_value="extrapolate")
            cpwp[:, n, j] = f(z_out)

            # Stratification parameter
            rho_rms_phase = np.sqrt(np.nanmean(rho_f[:, idx1] ** 2, axis=1))
            f = interp1d(z_burst, rho_rms_phase[zout_idx], fill_value="extrapolate")
            rho_rms[:, n, j] = f(z_out)

            w_prime_squared_phase = np.nanmean(w_prime_squared[:, idx1], axis=1)
            f = interp1d(z_burst, w_prime_squared_phase[zout_idx], fill_value="extrapolate")
            vke[:, n, j] = f(z_out)

    except Exception as e:
        print(n, e)
    print(n)

out = {
    "tau_wave_maj": tau_wave_maj,
    "tau_wave_min": tau_wave_min,
    "tau_wave_total": tau_wave_total,
    "tau_wc_maj": tau_wc_maj,
    "tau_wc_min": tau_wc_min,
    "tau_wc_total": tau_wc_total,
    "tau_rs_wave_maj": tau_rs_wave_maj,
    "tau_rs_wave_min": tau_rs_wave_min,
    "tau_rs_wave_total": tau_rs_wave_total,
    "tau_rs_wc_maj": tau_rs_wc_maj,
    "tau_rs_wc_min": tau_rs_wc_min,
    "tau_rs_wc_total": tau_rs_wc_total,
    "tau_rs_vt_maj": tau_rs_vt_maj,
    "tau_rs_vt_min": tau_rs_vt_min,
    "tau_rs_vt_total": tau_rs_vt_total,
    "u_wave": vel_wave_maj,
    "v_wave": vel_wave_min,
    "dudt_wave": accel_wave_maj,
    "dvdt_wave": accel_wave_min,
    "u_wc": vel_wc_maj,
    "v_wc": vel_wc_min,
    "upwp_pot": upwp_pot_all,
    "vpwp_pot": vpwp_pot_all,
    "cpwp": cpwp,
    "ssc": ssc,
    "vke": vke,
    "rho_rms": rho_rms,
    "ubr": ubr,
    "ucr": ucr,
    "phiwc": phiwc,
    "zr": zr,
    "omega": omega,
    "z": z_out,
    "phase": phasebins,
}
np.save(os.path.join(get_project_root(), "data/vectrino.npy"), out)
