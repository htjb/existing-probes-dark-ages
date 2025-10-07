import jax.numpy as jnp
import jax
from cmrt.utils.utils import const

def n_H_tot(z, cosmo):
    # mean hydrogen number density in m^-3
    H0 = cosmo.H0 * 1e3 / 3.086e22
    rho_bar = 3/8*H0**2/jnp.pi/const.G*cosmo.Omega_b*(1+z)**3
    return (const.X_H * rho_bar) / const.m_p

@jax.jit
def xc(z, xe, Tk, cosmo):

    nH = n_H_tot(z, cosmo)

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
