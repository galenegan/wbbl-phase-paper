import numpy as np


def m94(ubr, wr, ucr, zr, phiwc, kN):
    """
    M94 - Grant-Madsen model from Madsen(1994)
    ustrc, ustrr, ustrm, dwc, fwc, zoa =
        m94( ubr, wr, ucr, zr, phiwc, kN, iverbose )

    Input:
        ubr = rep. wave-orbital velocity amplitude outside wbl [m/s]
        wr = rep. angular wave frequency = 2pi/T [rad/s]
        ucr = current velocity at height zr [m/s]
        zr = reference height for current velocity [m]
        phiwc = angle between currents and waves at zr (radians)
        kN = bottom roughness height (e.q. Nikuradse k) [m]
        iverbose = True/False; when True, extra output
    Returned as tuple
        ustrc  = current friction velocity         u*c [m/s]
        ustrr  = w-c combined friction velocity    u*r [m/s]
        ustrwm = wave max. friction velocity      u*wm [m/s]
        dwc = wave boundary layer thickness [m]
        fwc = wave friction factor [ ]
        zoa = apparent bottom roughness [m]

    Chris Sherwood, USGS
    November 2005: Removed when waves == 0
    July 2005: Removed bug found by JCW and RPS
    March 2014: Ported from Matlab to Python
    """
    MAXIT = 20
    vk = 0.41
    rmu = np.zeros((MAXIT, 1))
    Cmu = np.zeros((MAXIT, 1))
    fwci = np.zeros((MAXIT, 1))
    dwci = np.zeros((MAXIT, 1))
    ustrwm2 = np.zeros((MAXIT, 1))
    ustrr2 = np.zeros((MAXIT, 1))
    ustrci = np.zeros((MAXIT, 1))

    # ...junk return values
    ustrc = 99.99
    ustrwm = 99.99
    ustrr = 99.99
    fwc = 0.4
    zoa = kN / 30.0
    zoa = zoa
    dwc = kN

    # ...some data checks
    if wr <= 0.0:
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa

    if ubr < 0.0:
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa

    if kN < 0.0:
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa

    zo = kN / 30
    cosphiwc = abs(np.cos(phiwc))
    rmu[0] = 0.0
    Cmu[0] = 1.0
    cukw = Cmu[0] * ubr / (kN * wr)
    fwci[0] = fwc94(Cmu[0], cukw)  # Eqn. 32 or 33

    ustrwm2[0] = 0.5 * fwci[0] * ubr * ubr  # Eqn. 29
    ustrr2[0] = Cmu[0] * ustrwm2[0]  # Eqn. 26
    ustrr = np.sqrt(ustrr2[0])
    dwci[0] = kN
    if cukw >= 8.0:
        dwci[0] = 2 * vk * ustrr / wr
    lnzr = np.log(zr / dwci[0])
    lndw = np.log(dwci[0] / zo)
    lnln = lnzr / lndw
    bigsqr = -1.0 + np.sqrt(1 + ((4.0 * vk * lndw) / (lnzr * lnzr)) * ucr / ustrr)
    ustrci[0] = 0.5 * ustrr * lnln * bigsqr
    nit = 1

    for i in range(1, MAXIT):
        rmu[i] = ustrci[i - 1] * ustrci[i - 1] / ustrwm2[i - 1]
        Cmu[i] = np.sqrt(1.0 + 2.0 * rmu[i] * cosphiwc + rmu[i] * rmu[i])  # Eqn 27
        cukw = Cmu[i] * ubr / (kN * wr)
        fwci[i] = fwc94(Cmu[i], cukw)  # Eqn. 32 or 33
        ustrwm2[i] = 0.5 * fwci[i] * ubr * ubr  # Eqn. 29
        ustrr2[i] = Cmu[i] * ustrwm2[i]  # Eqn. 26
        ustrr = np.sqrt(ustrr2[i])
        dwci[i] = kN
        if (Cmu[i] * ubr / (kN * wr)) >= 8.0:
            dwci[i] = 2 * vk * ustrr / wr  # Eqn.36
        lnzr = np.log(zr / dwci[i])
        lndw = np.log(dwci[i] / zo)
        lnln = lnzr / lndw
        bigsqr = -1.0 + np.sqrt(1 + ((4.0 * vk * lndw) / (lnzr * lnzr)) * ucr / ustrr)
        ustrci[i] = 0.5 * ustrr * lnln * bigsqr  # Eqn. 38
        diffw = abs((fwci[i] - fwci[i - 1]) / fwci[i])
        # print i,diffw
        if diffw < 0.0005:
            break
        ustrwm = np.sqrt(ustrwm2[nit])
        ustrc = ustrci[nit]
        ustrr = np.sqrt(ustrr2[nit])
        zoa = np.exp(np.log(dwci[nit]) - (ustrc / ustrr) * np.log(dwci[nit] / zo))  # Eqn. 11
        fwc = fwci[nit]
        dwc = dwci[nit]
        nit = nit + 1

    return ustrc, ustrr, ustrwm, dwc, fwc, zoa


def fwc94(cmu, cukw):
    """
    fwc94 - Wave-current friction factor
    fwc = fwc94( cmu, cukw )
    Equations 32 and 33 in Madsen, 1994

    csherwood@usgs.gov 4 March 2014
    """
    fwc = 0.00999  # meaningless (small) return value
    if cukw <= 0.0:
        print
        "ERROR: cukw too small in fwc94: {0}\n".format(cukw)
        return fwc

    if cukw < 0.2:
        fwc = np.exp(7.02 * 0.2 ** (-0.078) - 8.82)
    if (cukw >= 0.2) and (cukw <= 100.0):
        fwc = cmu * np.exp(7.02 * cukw ** (-0.078) - 8.82)
    elif (cukw > 100.0) and (cukw <= 10000.0):
        fwc = cmu * np.exp(5.61 * cukw ** (-0.109) - 7.30)
    elif cukw > 10000.0:
        fwc = cmu * np.exp(5.61 * 10000.0 ** (-0.109) - 7.30)

    return fwc
