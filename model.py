"""Energy-balance, purity and optical models for the TGMS photothermal review.

All constants are documented with the source used in the manuscript.
Units: SI unless stated.  Molar formulation of the leaf/organ energy budget
follows Campbell & Norman (1998) and Jones (2014).
"""
import numpy as np

CP = 29.3          # J mol-1 K-1, molar heat capacity of air
SIGMA = 5.670e-8   # W m-2 K-4
EPS = 0.97         # thermal emissivity of plant surface
RHO_CP = 1200.0    # J m-3 K-1 (volumetric, used for cross-checks)


def g_radiative(T_air_C=22.0, eps=EPS):
    """Radiative conductance, mol m-2 s-1 (Campbell & Norman 1998, eq. 12.7)."""
    T = T_air_C + 273.15
    return 4.0 * eps * SIGMA * T ** 3 / CP


def g_boundary(u, d, geometry="cylinder"):
    """Boundary-layer conductance to heat, mol m-2 s-1.

    u  wind speed (m s-1), d characteristic dimension (m).
    Laminar forced convection; the 1.4 factor is the outdoor turbulence
    enhancement of Campbell & Norman (1998, section 7.6).
    """
    k = 0.205 if geometry == "cylinder" else 0.135
    u = np.maximum(u, 0.05)
    return 1.4 * k * np.sqrt(u / d)


def dT_steady(dR, u=1.0, d=0.010, T_air_C=22.0, dLE=0.0):
    """Organ-to-air temperature rise (K) for an extra absorbed flux dR (W m-2).

    dLE is any extra latent loss (W m-2) induced by the warming; the panicle
    at booting is enclosed by the flag-leaf sheath and transpires little, so
    the default of zero is the most favourable case for the intervention.
    """
    g = g_boundary(u, d) + g_radiative(T_air_C)
    return (dR - dLE) / (CP * g)


def absorbed_increment(da, S_global, nir_fraction=0.52, f_intercept=1.0):
    """Extra absorbed flux (W m-2) from an increase da in NIR absorptance.

    nir_fraction: share of global shortwave irradiance above 700 nm
    (ASTM G173 AM1.5G direct+circumsolar ~ 0.52 of 1000 W m-2).
    f_intercept: fraction of that flux actually intercepted by the coated
    organ surface (projected-to-total area ratio and shading by canopy).
    """
    return da * S_global * nir_fraction * f_intercept


def diurnal_irradiance(hours, S_peak=850.0, sunrise=6.0, sunset=18.0):
    """Clear-sky sinusoid; zero at night."""
    x = np.pi * (hours - sunrise) / (sunset - sunrise)
    return np.where((hours >= sunrise) & (hours <= sunset), S_peak * np.sin(x), 0.0)


def diurnal_air_temperature(hours, tmin=19.0, tmax=26.0, t_min_h=5.0, t_max_h=14.5):
    """Smooth asymmetric diurnal air temperature (sine-ramp approximation)."""
    h = np.asarray(hours, dtype=float) % 24
    amp = (tmax - tmin) / 2.0
    mean = (tmax + tmin) / 2.0
    phase = 2 * np.pi * (h - t_max_h) / 24.0
    return mean + amp * np.cos(phase) * (1 - 0.12 * np.sin(2 * np.pi * (h - t_min_h) / 24.0))


def sterility_logistic(T, csit=23.5, k=1.6):
    """Pollen-sterility fraction as a logistic function of the mean temperature
    over the sensitive window.  csit is the inflexion (the operationally
    defined critical sterility-inducing temperature); k sets the steepness.
    Calibrated to give ~0.5 at the CSIT and >0.99 about 3 K above it, which is
    the behaviour reported for tms5-based lines screened at 23.5 degC.
    """
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(T, dtype=float) - csit)))


def purity(f_restored, selfed_set=0.25, outcross_set=0.35):
    """Hybrid-seed purity (fraction true hybrid) on a seed-number basis.

    f_restored: fraction of female florets whose pollen regains function
    selfed_set: seed set on those florets from self pollen
    outcross_set: seed set from the pollinator on the remaining florets
    Selfed florets are assumed to be pollinated by their own pollen first.
    """
    f = np.asarray(f_restored, dtype=float)
    selfed = f * selfed_set
    hybrid = (1 - f) * outcross_set + f * max(outcross_set - selfed_set, 0.0) * 0.0
    return hybrid / (hybrid + selfed)


def water_offset(depth_cm, dT_max=3.0, d50=8.0):
    """Empirical saturating offset of young-panicle temperature by irrigation
    depth (K), fitted to the qualitative Japanese deep-water results
    (Kobayashi et al. 1979; Satake et al. 1988): little effect until the water
    reaches the growing point, saturating near 15-20 cm.
    """
    d = np.asarray(depth_cm, dtype=float)
    return dT_max * d ** 2 / (d50 ** 2 + d ** 2)


if __name__ == "__main__":
    print("g_r (22 C)           %.3f mol m-2 s-1" % g_radiative())
    for u in (0.3, 1.0, 3.0):
        g = g_boundary(u, 0.010) + g_radiative()
        print("u=%.1f  g_total=%.2f  resistance %.1f W m-2 K-1  dT per 100 W m-2 = %.2f K"
              % (u, g, CP * g, dT_steady(100.0, u)))
    print("clear noon, da=0.25:", round(float(dT_steady(absorbed_increment(0.25, 850.0), 1.0)), 2), "K")
    print("overcast,   da=0.25:", round(float(dT_steady(absorbed_increment(0.25, 180.0), 1.0)), 2), "K")
