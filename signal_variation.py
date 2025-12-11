"""Signal variation due to cosmological parameters."""

import matplotlib.pyplot as plt
import numpy as np
from anesthetic import read_chains
from jax import numpy as jnp

from darkages.signal import generate_signal


def prior_sample(n_samples: int = 1000) -> np.ndarray:
    """Generate prior samples for H0, Omega_m.

    Args:
        n_samples: Number of prior samples to generate.

    Returns:
        samples: Array of shape (n_samples, 2) with prior samples.
    """
    H0 = np.linspace(40, 100, n_samples)
    Omega_m = np.linspace(0.1, 0.5, n_samples)
    samples = np.vstack([H0, Omega_m]).T
    return samples


z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1000)


probes = [
    "COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE"
]

recombination_model = "hyrec"  # 'recfast' or 'hyrec'

prior = prior_sample(100)
print(prior.min(axis=0), prior.max(axis=0))

chains = read_chains(probes[0])
params = ["H0", "omegam", "omegabh2", "omegach2", "yheused"]
chains = chains[params]
chains = chains.compress()
samples = np.mean(chains.values, axis=0)
# samples[2] *= (samples[0] / 100) ** 2  # convert to Omega_b
# samples[3] *= (samples[0] / 100) ** 2  # convert to Omega_c
samples = np.round(samples, 3)
print("Fiducial parameters:", samples)

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True, sharey=True)
axes = axes.flatten()
parameters = ["H0", "Omega_m"]
titles = [r"$H_0$", r"$\Omega_m$"]
for i in range(len(parameters)):
    axes[i].set_title(titles[i])
    norm = plt.Normalize(prior[:, i].min(), prior[:, i].max())
    cmap = plt.cm.inferno_r
    for j in range(len(prior)):
        params = samples.copy()
        params[i] = prior[j, i]
        print(params)
        signal = generate_signal(
            1420.4 / (1 + z_grid),
            jnp.array(params),
            z_init=z_init,
            rec_model=recombination_model,
        )
        color = cmap(norm(prior[j, i]))
        axes[i].plot(1420.4 / (z_grid + 1), signal, color=color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[i])  # , label=titles[i])
    axes[i].set_xlabel(r"$\nu$ [MHz]")
    if i == 0:
        axes[i].set_ylabel(r"$T_{21}$ [mK]")

plt.tight_layout()
plt.savefig(
    "21cm_brightness_temperature_variation_" + recombination_model + ".pdf"
)
plt.show()
