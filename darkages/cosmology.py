"""Define the comsological parameters used in DarkAges."""

from collections import namedtuple

cosmology = namedtuple('cosmo', ['H0', 'Omega_m',
                                 'Omega_b', 'Omega_c', 
                                 'Omega_bh2', 'Omega_ch2',
                                 'z_init',
                                 'Y_He'])
