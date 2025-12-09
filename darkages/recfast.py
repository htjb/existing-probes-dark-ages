"""A wrapper for Recfast++."""

import os
import re
from typing import NamedTuple

import numpy as np


def update_recfast_ini(cosmo: NamedTuple, base_dir: str = "./") -> None:
    """Build the Recfast++ .ini file.

    Update H0, Omega_b, Omega_c, and Y_He in a Recfast++ .ini file.

    Args:
        cosmo: Cosmology namedtuple with attributes H0, Omega_b, Omega_c, Y_He.
        base_dir: Directory to save the modified ini file.
    """
    with open("recfast-.vx/runfiles/parameters.ini", "r") as f:
        lines = f.readlines()

    # Compute derived parameter
    omega_m = cosmo.Omega_b + cosmo.Omega_c

    # Regex replacements
    replacements = {
        r"^(Yp\s*=\s*)([0-9Ee\.\+\-]+)": f"\\g<1>{cosmo.Y_He}",
        r"^(Omega_b\s*=\s*)([0-9Ee\.\+\-]+)": f"\\g<1>{cosmo.Omega_b}",
        r"^(Omega_m\s*=\s*)([0-9Ee\.\+\-]+)": f"\\g<1>{omega_m}",
        r"^(h100\s*=\s*)([0-9Ee\.\+\-]+)": f"\\g<1>{cosmo.H0 / 100.0}",
    }

    new_lines = []
    for line in lines:
        new_line = line
        for pattern, repl in replacements.items():
            new_line = re.sub(pattern, repl, new_line)
        new_lines.append(new_line)

    with open(base_dir + "recfast_input.ini", "w") as f:
        f.writelines(new_lines)


def call_recfast(
    base_dir: str = "./", redshift: int = 1100
) -> tuple[np.ndarray, np.ndarray]:
    """Code to call recfast.

    Code runs the recfast executable.

    Args:
        base_dir: Directory where the recfast_input.ini file is located.
        redshift: Redshift at which to interpolate the output.

    Returns:
        init_xe: Interpolated free electron fraction at the given redshift.
        init_Tk: Interpolated gas temperature at the given redshift.
    """
    os.system(
        "./recfast-.vx/Recfast++ "
        + base_dir
        + "/recfast_input.ini"
        + "> /dev/null 2>&1"
    )

    data = np.loadtxt(
        "recfast-output/Xe_Recfast++.Rec_corrs_CT2010.dat", skiprows=4
    )
    recz = data[:, 0][::-1]
    xe = data[:, 1][::-1]
    Tk = data[:, 4][::-1]

    init_xe = np.interp(redshift, recz, xe)
    init_Tk = np.interp(redshift, recz, Tk)  # in Kelvin

    return init_xe, init_Tk
