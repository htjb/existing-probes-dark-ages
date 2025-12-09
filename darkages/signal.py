"""Functions to compute the 21cm signal from the Dark Ages."""

import jax
import jax.numpy as jnp

from darkages.cosmology import cosmology
from darkages.hyrec import call_hyrec, set_up_hyrec
from darkages.recfast import call_recfast, update_recfast_ini
from darkages.utils import const


@jax.jit
def n_H_tot(z: int, cosmo: cosmology) -> jnp.ndarray:
    """Mean hydrogen number density in m^-3.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        n_H: Mean hydrogen number density in m^-3.
    """
    H0 = cosmo.H0 * 1e3 / 3.086e22
    rho_bar = (
        (3 * H0**2 / (8 * jnp.pi * const.G)) * cosmo.Omega_b * (1 + z) ** 3
    )
    return ((1 - cosmo.Y_He) * rho_bar) / const.m_p


@jax.jit
def xc(
    z: int, xe: jnp.ndarray, Tk: jnp.ndarray, cosmo: cosmology
) -> jnp.ndarray:
    """Calculate the coupling coefficient xc.

    Args:
        z: Redshift.
        xe: Free electron fraction.
        Tk: Kinetic temperature in Kelvin.
        cosmo: Cosmology object.

    Returns:
        xc: Coupling coefficient.
    """
    nH = n_H_tot(z, cosmo)

    xc = (const.Tstar * kappa(z, Tk, xe) * nH) / (
        const.A10 * const.Tcmb0 * (1 + z)
    )

    return xc


@jax.jit
def kappa(z: int, Tk: jnp.ndarray, xe: jnp.ndarray) -> jnp.ndarray:
    """Calculate the collisional coupling coefficient kappa.

    Args:
        z: Redshift.
        Tk: Kinetic temperature in Kelvin.
        xe: Free electron fraction.

    Returns:
        kappa: Collisional coupling coefficient in m^3/s.
    """
    # H-H collisions (Zygelman 2005, valid up to ~300K)
    kappa_HH = 3.1e-11 * Tk**0.357 * jnp.exp(-32.0 / Tk) * 1e-6  # m^3/s

    # e-H collisions from https://arxiv.org/pdf/2108.00115
    kappa_eH = (
        10
        ** (
            -9.607
            + 0.5 * jnp.log10(Tk) * jnp.exp(-((jnp.log10(Tk)) ** 4) / 1800.0)
        )
        * 1e-6
    )

    # p-H collisions (assume ~same as e-H to leading order)
    # there is a factor related to the ratio of masses but
    # they are subdominant anyway
    kappa_pH = kappa_eH

    return kappa_HH * (1 - xe) + (kappa_eH + kappa_pH) * xe


@jax.jit
def Tcmb(z: int) -> jnp.ndarray:
    """Calculate CMB temperature at redshift z.

    Args:
        z: Redshift.

    Returns:
        T: CMB temperature in Kelvin.
    """
    T = const.Tcmb0 * (1 + z)  # CMB temperature in Kelvin, scaled by redshift
    return T


def Ts(T_gas: jnp.ndarray, T_cmb: jnp.ndarray, xc: jnp.ndarray) -> jnp.ndarray:
    """Calculate spin temperature.

    Args:
        T_gas: Kinetic temperature in Kelvin.
        T_cmb: CMB temperature in Kelvin.
        xc: Coupling coefficient.

    Returns:
        Tspin: Spin temperature in Kelvin.
    """
    Tspin = (T_cmb ** (-1) + xc * T_gas ** (-1)) / (1 + xc)
    return Tspin ** (-1)


@jax.jit
def T21(
    z: int,
    T_gas: jnp.ndarray,
    T_cmb: jnp.ndarray,
    T_s: jnp.ndarray,
    xe: jnp.ndarray,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate 21cm brightness temperature.

    Args:
        z: Redshift.
        T_gas: Kinetic temperature in Kelvin.
        T_cmb: CMB temperature in Kelvin.
        T_s: Spin temperature in Kelvin.
        xe: Free electron fraction.
        cosmo: Cosmology object.

    Returns:
        T21: 21cm brightness temperature in mKelvin.
    """
    return (
        54
        * (1 - xe)
        * ((1 - cosmo.Y_He) / 0.76)
        * (cosmo.Omega_bh2)
        / 0.02242
        * jnp.sqrt(
            0.1424 / (cosmo.Omega_m * (cosmo.H0 / 100) ** 2) * ((1 + z) / 40)
        )
        * (1 - T_cmb / T_s)
    )  # 21cm brightness temperature in mKelvin


xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


def generate_signal(
    f_grid: jnp.ndarray,
    sample: jnp.ndarray,
    z_init: int,
    rec_model: str = "recfast",
) -> jnp.ndarray:
    """Generate 21cm signal for a given cosmological sample.

    Args:
        f_grid: Frequency grid in MHz.
        sample: Cosmological parameters [H0, Omega_m, Omega_b, Omega_c,
                Y_He].
        z_init: Initial redshift.
        rec_model: Recombination model to use ('recfast' or 'hyrec').

    Returns:
        T21_values: 21cm brightness temperature values over the frequency grid.
    """
    cosmo = cosmology(
        H0=sample[0],
        Omega_m=sample[1],
        Omega_b=sample[2] / (sample[0] / 100) ** 2,
        Omega_c=sample[3] / (sample[0] / 100) ** 2,
        Omega_bh2=sample[2],
        Omega_ch2=sample[3],
        z_init=z_init,
        Y_He=sample[4],
    )  # Example cosmology parameters
    try:
        if rec_model == "recfast":
            update_recfast_ini(cosmo)
            z_grid = 1420.4 / (f_grid) - 1
            xe, T_gas = call_recfast(base_dir="./", redshift=z_grid)
            xe, T_gas = jnp.array(xe), jnp.array(T_gas)
        elif rec_model == "hyrec":
            set_up_hyrec(
                H0=cosmo.H0,
                omb=cosmo.Omega_b,
                omc=cosmo.Omega_c,
                omk=0,
                yhe=cosmo.Y_He,
                base_dir="./",
            )
            z_grid = 1420.4 / (f_grid) - 1
            xe, T_gas = call_hyrec(base_dir="./", redshift=z_grid)
            xe, T_gas = jnp.array(xe), jnp.array(T_gas)

        xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

        T_cmb = Tcmb(z_grid)

        T_s = Ts(T_gas, T_cmb, xc_values)

        T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
        return T21_values
    except Exception as e:
        print(f"Error generating signal for sample {sample}: {e}")
        return jnp.full_like(f_grid, jnp.nan)
