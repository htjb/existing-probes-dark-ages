from collections import namedtuple
from jax import jit
from jax import numpy as jnp

# speed of light, gravitational constant, hydrogen mass fraction, and proton mass
constants = namedtuple('constants', ['c', 'G', 'm_p', 'A10', 'Tstar', 'Tcmb0',
                                     'm_e', 'k_b', 'h_P'])
# X_H is the hydrogen mass fraction i.e. 1 - Y_He
const = constants(c=2.99792458e8, G=6.67430e-11,  # c in m/s, G in m^3/(kg*s^2)
                  m_p=1.67e-27,
                  A10=2.85e-15, Tstar=0.068, Tcmb0=2.725,
                  m_e=9.109e-31, k_b=1.380649e-23, h_P=6.62607015e-34/(2*jnp.pi)
                  )  # m_p in kg, A10 in m^3/s^2, T* in mK