"""Examine the chains and compare outputs from HyRec and Recfast."""

import emcee
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples, read_chains
from anesthetic.plot import kde_contour_plot_2d
from jax import numpy as jnp

z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1000)

probes = [  # planck cmb baseline high l TT + low l  and low EE
    "COM_CosmoParams_fullGrid_R3.01/base/"
    + "plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE",
    # wmap cmb full temperature and polarisation 9 year
    "COM_CosmoParams_fullGrid_R3.01/base/WMAP/base_WMAP",
    # bao DR12, MGS, 6dF
    "COM_CosmoParams_fullGrid_R3.01/base/BAO_Cooke17/base_BAO_Cooke17",
    # lensprior is Standard base parameters with
    # ns = 0.96 ± 0.02, Ωbh2 = 0.0222 ± 0.0005, 100>H0>40, τ=0.055
    # des cosmic shear + galaxy auto + cross
    "COM_CosmoParams_fullGrid_R3.01/base/DES_lenspriors/base_DES_lenspriors",
]
probes = probes[::-1]  # reverse order for plotting
colors = ["C0", "C1", "C2", "C3"][::-1]
params = ["H0", "omegam", "omegabh2"]  # , 'omegach2', 'yheused']
# fig, ax = make_1d_axes(params, figsize=(6.3, 3), facecolor='w', ncol=3)

legend_labels = ["DES", "BAO", "WMAP", "Planck"]

fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5), sharex=True, sharey=True)
mins, maxs = [], []
for i in range(len(probes)):
    file_path = probes[i]
    chains = read_chains(file_path)
    chains = chains[params]

    chains = chains.compress()
    samples = chains.values

    # autocorrelation time is like a measure of independence
    # and tau is how many steps to skip to get independent samples
    tau = emcee.autocorr.integrated_time(samples, tol=0)
    samples = samples[:: int(tau)]  # [:300]
    chains = MCMCSamples(samples, columns=params)
    omegam = chains["omegam"]
    H0 = chains["H0"]
    mins.append([H0.min(), omegam.min()])
    maxs.append([H0.max(), omegam.max()])
    kde_contour_plot_2d(
        axes,
        omegam,
        H0,
        levels=[0.95, 0.68],
        color=colors[i],
        label=legend_labels[i],
    )
axes.legend(loc="upper right", fontsize=8)
axes.set_xlabel(r"$\Omega_m$")
axes.set_ylabel(r"$H_0$ [km/s/Mpc]")
plt.tight_layout()
plt.savefig("chain_comparison.pdf")
plt.show()
