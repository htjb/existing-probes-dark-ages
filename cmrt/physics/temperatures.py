import jax.numpy as jnp
from jax import jit
from cmrt.utils.utils import const

@jit
def Tcmb(z):
    T = const.Tcmb0 * (1 + z)  # CMB temperature in Kelvin, scaled by redshift
    return T

def Ts(z, T_gas, T_cmb, xc):
    """ Calculate spin temperature Tspin."""
    Tspin = (T_cmb**(-1) + xc*T_gas**(-1))/(1+xc)
    return Tspin**(-1)

def T21(z, T_gas, T_cmb, T_s, xe, cosmo):
    """Calculate 21cm brightness temperature."""
    return 27*(1-xe)*(const.X_H/0.76)* \
        (cosmo.Omega_b*(cosmo.H0/100)**2)/0.023 * \
            jnp.sqrt(0.15/(cosmo.Omega_m*(cosmo.H0/100)**2)*(1+z/10)) * \
                (1 - T_cmb/T_s)  # 21cm brightness temperature in mKelvin