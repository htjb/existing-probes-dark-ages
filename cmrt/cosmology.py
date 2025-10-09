import jax.numpy as jnp
from jax import jit
from collections import namedtuple

cosmology = namedtuple('cosmo', ['H0', 'Omega_m', 'Omega_l', 'Omega_r',
                                 'Omega_b', 'Omega_c', 'z_init',
                                 'Y_He'])

@jit
def HubbleParameter(z, cosmo):
    """ Calculate H(z)."""

    H0 = cosmo.H0
    omega_m = cosmo.Omega_m
    omega_l = cosmo.Omega_l
    omega_r = cosmo.Omega_r

    return H0 * jnp.sqrt(omega_m * (1 + z)**3 + omega_r*(1+z)**4 +  omega_l)
