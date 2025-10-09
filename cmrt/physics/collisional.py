import jax.numpy as jnp
import jax
from cmrt.utils.utils import const

def n_H_tot(z, cosmo, yhe):
    # mean hydrogen number density in m^-3
    H0 = cosmo.H0 * 1e3 / 3.086e22
    rho_bar = 3/8*H0**2/jnp.pi/const.G*cosmo.Omega_b*(1+z)**3
    return ((1-yhe) * rho_bar) / const.m_p

@jax.jit
def xc(z, xe, Tk, cosmo, yhe):

    nH = n_H_tot(z, cosmo, yhe)

    xc = const.Tstar / (const.A10 * const.Tcmb0*(1+z)) * kappa(z, Tk, xe) * nH *(1- xe)

    return xc

@jax.jit
def kappa(z, Tk, xe):
    # H-H collisions (Zygelman 2005, valid up to ~300K)
    kappa_HH = 3.1e-11 * Tk**0.357 * jnp.exp(-32.0/Tk) * 1e-6  # m^3/s

    # e-H collisions (Furlanetto et al. 2006, fit)
    kappa_eH = 1.0e-8 * Tk**-0.5 * jnp.exp(-35.0/Tk) * 1e-6  # m^3/s

    # p-H collisions (assume ~same as e-H to leading order)
    kappa_pH = kappa_eH  

    return kappa_HH * (1 - xe) + (kappa_eH + kappa_pH) * xe


@jax.jit
def Tcmb(z):
    T = const.Tcmb0 * (1 + z)  # CMB temperature in Kelvin, scaled by redshift
    return T

def Ts(z, T_gas, T_cmb, xc):
    """ Calculate spin temperature Tspin."""
    Tspin = (T_cmb**(-1) + xc*T_gas**(-1))/(1+xc)
    return Tspin**(-1)

def T21(z, T_gas, T_cmb, T_s, xe, cosmo, yhe):
    """Calculate 21cm brightness temperature."""
    return 27*(1-xe)*((1-yhe)/0.76)* \
        (cosmo.Omega_b*(cosmo.H0/100)**2)/0.023 * \
            jnp.sqrt(0.15/(cosmo.Omega_m*(cosmo.H0/100)**2)*(1+z/10)) * \
                (1 - T_cmb/T_s)  # 21cm brightness temperature in mKelvin