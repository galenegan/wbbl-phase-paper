# Packages
import numpy as np
import pandas as pd
import scipy.signal as sig

np.seterr(divide="ignore", invalid="ignore", over="ignore")


def naninterp(x):
    return pd.Series(x).interpolate(method="linear").ffill().bfill().values


def pa_rotation(u, v, theta):
    # Storing as complex variable w = u + iv
    w = u + 1j * v

    wr = w * np.exp(-1j * theta * np.pi / 180)
    vel_maj = np.real(wr)
    vel_min = np.imag(wr)

    return vel_maj, vel_min


def velocity_decomp(u, v, w, fs):
    """A method to decompose the wave velocity from the full velocity
    signal. Based on the Bricker & Monismith phase method, but returns the
    full timeseries via IFFT, rather than the wave stresses via spectral sum.

    Parameters
    ----------
    u, v, w: 1d or 2d numpy array
      A velocity time series vector

    fs: float
      Sampling frequency (Hz)

    Returns
    ---------
    u_wave, v_wave, w_wave: Tuple of 1d numpy arrays
      The wave velocity time series vectors


    """
    assert u.shape == v.shape
    assert u.shape == w.shape

    if u.ndim == 2:
        rows, cols = u.shape
    elif u.ndim == 1:
        rows = 1
        cols = len(u)
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        w = np.atleast_2d(w)

    # Putting everything in a dictionary
    U = {"u": u, "v": v, "w": w}

    u_wave = np.zeros((rows, cols)) * np.nan
    v_wave = np.zeros((rows, cols)) * np.nan
    w_wave = np.zeros((rows, cols)) * np.nan

    for component_name, vel in U.items():
        for row in range(rows):
            u_r = naninterp(vel[row, :])
            if np.sum(np.isnan(u_r)) >= 0.5 * len(u_r):
                continue

            n = len(u_r)
            nfft = n

            u_r = sig.detrend(u_r)

            # Amplitude of the wave component
            Amu = np.fft.fft(u_r) / np.sqrt(n)

            # frequency resolution
            df = fs / (nfft - 1)

            # nyquist frequency
            nnyq = int(np.floor(nfft / 2 + 1))
            freq = np.arange(0, nnyq) * df

            # Computing the full spectra
            Suu = np.real(Amu * np.conj(Amu)) / (nnyq * df)
            Suu = Suu.squeeze()[:nnyq]

            # Locations of peak frequency
            offset = np.sum(freq <= 0.1)

            uumax = np.argmax(Suu[(freq > 0.1) & (freq < 0.7)]) + offset

            if component_name == "u" or component_name == "v":
                widthratiolow = 2.5
                widthratiohigh = 1.5
            elif component_name == "w":
                widthratiolow = 4
                widthratiohigh = 3.5

            freq_max = freq[uumax]
            waverange_u = np.arange(
                max(uumax - (freq_max / widthratiolow) // df, 0),
                min(uumax + (freq_max / widthratiohigh) // df, len(freq) - 1),
            ).astype(int)
            interprange_u = np.arange(1, np.nanargmin(np.abs(freq - 2))).astype(int)
            waverange_u = waverange_u[(waverange_u >= 0) & (waverange_u < nnyq)]
            interprange_u = interprange_u[(interprange_u >= 0) & (interprange_u < nnyq)]

            Suu_turb = Suu[interprange_u]
            freq_uu = freq[interprange_u]
            Suu_turb = np.delete(Suu_turb, waverange_u - interprange_u[0])
            freq_uu = np.delete(freq_uu, waverange_u - interprange_u[0])
            Suu_turb = Suu_turb[freq_uu > 0]
            freq_uu = freq_uu[freq_uu > 0]

            # Linear interpolation over turbulent spectra
            F = np.log(freq_uu)
            S = np.log(Suu_turb)
            Puu = np.polyfit(F, S, deg=1)
            Puuhat = np.exp(np.polyval(Puu, np.log(freq)))

            # Wave spectra
            Suu_wave = Suu[waverange_u] - Puuhat[waverange_u]

            # Amplitude and phase
            Amu_wave = np.sqrt((Suu_wave + 0j) * df * nnyq)
            phase_u = np.arctan2(np.imag(Amu), np.real(Amu)).squeeze()

            Amp = np.zeros_like(freq)
            Amp[waverange_u] = Amu_wave
            Amp = np.concatenate((Amp[:-1], np.flipud(Amp[:-1])))
            if len(Amp) == len(phase_u) - 1:
                Amp = np.zeros_like(freq)
                Amp[waverange_u] = Amu_wave
                Amp = np.concatenate((Amp[:], np.flipud(Amp[:-1])))

            z = Amp * (np.cos(phase_u) + 1j * np.sin(phase_u))

            uw = np.fft.ifft(z) * np.sqrt(n)

            if component_name == "u":
                u_wave[row, :] = uw.real
            elif component_name == "v":
                v_wave[row, :] = uw.real
            elif component_name == "w":
                w_wave[row, :] = uw.real

    return u_wave, v_wave, w_wave
