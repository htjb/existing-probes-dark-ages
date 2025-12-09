"""Run the analysis in the paper."""

import emcee
import jax
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import read_chains
from anesthetic.samples import MCMCSamples
from fgivenx import plot_contours, plot_dkl
from jax import numpy as jnp
from tqdm import tqdm

from darkages.signal import generate_signal


def prior_sample(n_samples: int = 1000) -> np.ndarray:
    """Generate prior samples for H0, Omega_m, Omega_b, Omega_c, yhe.

    Args:
        n_samples: Number of prior samples to generate.

    Returns:
        samples: Array of shape (n_samples, 5) with prior samples.
    """
    H0 = np.random.uniform(40, 100, n_samples)
    Omega_m = np.random.uniform(0.1, 0.5, n_samples)
    Omega_b = np.random.uniform(0.005, 0.1, n_samples)
    yhe = np.random.uniform(0.2, 0.3, n_samples)
    samples = np.vstack([H0, Omega_m, Omega_b, (Omega_m - Omega_b), yhe]).T
    return samples


def get_minima(signal: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the minima of the 21cm brightness temperature signal.

    Args:
        signal: 1D array of 21cm brightness temperature signal.

    Returns:
        min_values: Minimum values of the signal.
        min_indices: Indices of the minimum values in the signal array.
    """
    min_indices = jnp.nanargmin(signal)
    min_values = jnp.nanmin(signal)
    return min_values, min_indices


def signal_call(f_grid: jnp.ndarray, sample: jnp.ndarray) -> jnp.ndarray:
    """Wrapper to call generate_signal for use in plotting functions.

    Args:
        f_grid: Frequency grid in MHz.
        sample: Cosmological parameters [H0, Omega_m, Omega_b, Omega_c,
                Y_He].

    Returns:
        T21_values: 21cm brightness temperature values over the frequency grid.
    """
    return generate_signal(f_grid, sample, z_init=1100, rec_model="hyrec")


z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1500)

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

titles = [
    "Planck 2018 high l TT +\nlowl + lowE ",
    "WMAP Full 9 Year",
    "BAO DR12 +\nMGS + 6dF",
    "DES Y1 Cosmic Shear +\nGalaxy Clustering + Cross",
]

plot_signal = True
plot_kl = False
plot_minima = False
plot_delta = False
recombination_model = "hyrec"  # 'recfast' or 'hyrec'

c = ["C0", "C1", "C2", "C3"]

if plot_signal:
    z_grid = jnp.linspace(z_init, 30, 500)
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 5), sharex=True)
    ax = ax.flatten()
    for i in range(len(probes)):
        file_path = probes[i]
        chains = read_chains(file_path)
        params = ["H0", "omegam", "omegabh2", "omegach2", "yheused"]
        chains = chains[params]
        if i == 0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0] / 100) ** 2
            prior[:, 3] *= (prior[:, 0] / 100) ** 2
        chains = chains.compress()
        samples = chains.values

        # autocorrelation time is like a measure of independence
        # and tau is how many steps to skip to get independent samples
        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[:: int(tau)]
        # samples = samples[:100]

        print("Number of samples in posterior:", len(samples))
        print("Number of samples in prior:", len(prior))

        plot_contours(
            signal_call,
            1420.4 / (1 + z_grid),
            prior,
            ax[i],
            colors=plt.cm.Blues_r,
            alpha=0.5,
            lines=False,
            fineness=1,
        )
        plot_contours(
            signal_call,
            1420.4 / (1 + z_grid),
            samples,
            ax[i],
            colors=plt.cm.Reds_r,
            alpha=0.5,
            lines=False,
            fineness=1,
        )
        ax[i].set_title(titles[i], fontsize=10)
        ax[i].set_ylim(-350, 10)
        ax[i].grid()
    ax[2].set_xlabel(r"$\nu$ [MHz]")
    ax[3].set_xlabel(r"$\nu$ [MHz]")
    ax[0].set_ylabel(r"$T_{21}$ [mK]")
    ax[2].set_ylabel(r"$T_{21}$ [mK]")
    ax[-1].plot([], [], color=plt.cm.Reds_r(0.5), label="Posterior", alpha=0.5)
    ax[-1].plot([], [], color=plt.cm.Blues_r(0.5), label="Prior", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        "21cm_brightness_temperature_contours_" + recombination_model + ".pdf"
    )
    # plt.show()
    plt.close()

if plot_kl:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for i in range(len(probes)):
        file_path = probes[i]
        chains = read_chains(file_path)
        params = ["H0", "omegam", "omegabh2", "omegach2", "yheused"]
        chains = chains[params]
        if i == 0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0] / 100) ** 2
            prior[:, 3] *= (prior[:, 0] / 100) ** 2
        chains = chains.compress()
        samples = chains.values

        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[:: int(tau)]
        # samples = samples[:100]

        plot_dkl(
            signal_call,
            1420.4 / (1 + z_grid),
            samples,
            prior,
            ax,
            color=c[i],
            label=titles[i],
        )
    plt.legend(fontsize=8)
    ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_ylabel(r"$D_{KL}$ [bits]")
    plt.tight_layout()
    plt.savefig("21cm_brightness_temperature_dkl.pdf")
    plt.show()

probes = probes[::-1]
titles = titles[::-1]
c = c[::-1]

if plot_minima or plot_delta:
    z_grid = jnp.linspace(z_init, 30, 5000)
    vmapped_get_minima = jax.vmap(get_minima, in_axes=0)
    for i in range(len(probes)):
        file_path = probes[i]
        chains = read_chains(file_path)
        params = ["H0", "omegam", "omegabh2", "omegach2", "yheused"]
        chains = chains[params]
        if i == 0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0] / 100) ** 2
            prior[:, 3] *= (prior[:, 0] / 100) ** 2
        chains = chains.compress()
        samples = chains.values

        # autocorrelation time is like a measure of independence
        # and tau is how many steps to skip to get independent samples
        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[:: int(tau)]  # [:300]

        signals = jnp.array(
            [signal_call(1420.4 / (1 + z_grid), s) for s in tqdm(samples)]
        )

        if i == 0:
            prior_signals = jnp.array(
                [signal_call(1420.4 / (1 + z_grid), s) for s in tqdm(prior)]
            )

        if plot_minima:
            min_values, min_indices = vmapped_get_minima(signals)

            mask = ~jnp.isnan(min_values)
            min_values = min_values[mask]
            min_indices = min_indices[mask]

            minimum_freqs = 1420.4 / (1 + z_grid[min_indices])
            minima_samples = MCMCSamples(
                data=jnp.array([minimum_freqs, min_values]).T,
                columns=["Frequency [MHz]", "Min T21 [mK]"],
            )
            print(
                f"{titles[i]}: Average Min T21 = "
                + f"{jnp.nanmean(min_values):.2f} mK"
                + f" $\\pm$ {jnp.nanstd(min_values):.2f} mK \n"
                f"Average nu_c {jnp.nanmean(minimum_freqs):.2f} MHz"
                + f" $\\pm$ {jnp.nanstd(minimum_freqs):.2f} MHz"
            )
            if i == 0:
                prior_min_values, prior_min_indices = vmapped_get_minima(
                    prior_signals
                )
                prior_minimum_freqs = 1420.4 / (1 + z_grid[prior_min_indices])
                print(
                    "Prior: Average Min T21 = "
                    + f"{jnp.nanmean(prior_min_values):.2f} mK"
                    + f" $\\pm$ {jnp.nanstd(prior_min_values):.2f} mK \n"
                    f"Average nu_c {jnp.nanmean(prior_minimum_freqs):.2f} MHz"
                    + f" $\\pm$ {jnp.nanstd(prior_minimum_freqs):.2f} MHz"
                )
            if i == 0:
                try:
                    ax = minima_samples.plot_2d(
                        ["Frequency [MHz]", "Min T21 [mK]"],
                        color=c[i],
                        alpha=0.5,
                        label=titles[i].split(" ")[0],
                        figsize=(3.5, 3.5),
                        kinds={"lower": "kde_2d"},
                    )
                except Exception as e:
                    print(
                        f"Error plotting minima for with kde {titles[i]}: {e}"
                    )
                    ax = minima_samples.plot_2d(
                        ["Frequency [MHz]", "Min T21 [mK]"],
                        color=c[i],
                        alpha=0.5,
                        label=titles[i].split(" ")[0],
                        figsize=(3.5, 3.5),
                        kinds={"lower": "scatter_2d"},
                    )
            else:
                minima_samples.plot_2d(
                    ax,
                    color=c[i],
                    alpha=0.5,
                    label=titles[i].split(" ")[0],
                    kinds={"lower": "kde_2d"},
                )
if plot_minima:
    ax.iloc[0, 0].set_xlabel(r"$\nu_c$ [MHz]")
    ax.iloc[0, 0].set_ylabel(r"$T_{21}(\nu_c)$ [mK]")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(
        "21cm_brightness_temperature_minima_" + recombination_model + ".pdf"
    )
plt.close()
