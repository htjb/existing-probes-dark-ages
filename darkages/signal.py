import jax.numpy as jnp
import jax
from darkages.utils import const

def n_H_tot(z, cosmo):
    # mean hydrogen number density in m^-3
    H0 = cosmo.H0 * 1e3 / 3.086e22
    rho_bar = (3 * H0**2 / (8 * jnp.pi * const.G)) * \
        cosmo.Omega_b * (1 + z)**3
    return ((1-cosmo.Y_He) * rho_bar) / const.m_p 

@jax.jit
def xc(z, xe, Tk, cosmo):

    nH = n_H_tot(z, cosmo)

    xc = (const.Tstar * kappa(z, Tk, xe) * nH) / (const.A10 * const.Tcmb0*(1+z))

    return xc

@jax.jit
def kappa(z, Tk, xe):
    # H-H collisions (Zygelman 2005, valid up to ~300K)
    kappa_HH = 3.1e-11 * Tk**0.357 * jnp.exp(-32.0/Tk) * 1e-6  # m^3/s

    # e-H collisions from https://arxiv.org/pdf/2108.00115 seems to fit furlanetto 2006 well
    kappa_eH = 10**(-9.607 + 0.5*jnp.log10(Tk) * jnp.exp(- (jnp.log10(Tk))**4 / 1800.0)) * 1e-6

    # p-H collisions (assume ~same as e-H to leading order)
    kappa_pH = kappa_eH 

    return kappa_HH * (1 - xe) + (kappa_eH + kappa_pH) * xe


@jax.jit
def Tcmb(z):
    T = const.Tcmb0 * (1 + z)  # CMB temperature in Kelvin, scaled by redshift
    return T

def Ts(T_gas, T_cmb, xc):
    Tspin = (T_cmb**(-1) + xc*T_gas**(-1))/(1+xc)
    return Tspin**(-1)

def T21(z, T_gas, T_cmb, T_s, xe, cosmo, model='barkana'):
    """Calculate 21cm brightness temperature."""
    if model == 'furlanetto': # i think from furlanetto 2006
        return 27*(1-xe)* ((1-cosmo.Y_He)/0.76)* \
            (cosmo.Omega_bh2)/0.023 * \
                jnp.sqrt(0.15/(cosmo.Omega_m*(cosmo.H0/100)**2)*((1+z)/10)) * \
                    (1 - T_cmb/T_s)  # 21cm brightness temperature in mKelvin
    elif model == 'barkana': #from https://arxiv.org/abs/2310.15530
        return 54*(1-xe)*((1-cosmo.Y_He)/0.76)* \
            (cosmo.Omega_bh2)/0.02242 * \
                jnp.sqrt(0.1424/(cosmo.Omega_m*(cosmo.H0/100)**2)*((1+z)/40)) * \
                    (1 - T_cmb/T_s)  # 21cm brightness temperature in mKelvin