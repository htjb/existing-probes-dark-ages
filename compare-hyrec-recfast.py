"""Code to compare HyRec and Recfast outputs given the same cosmology."""

import matplotlib.pyplot as plt
from anesthetic import read_chains
from jax import numpy as jnp

from darkages.signal import generate_signal

z_init = 2000
z_grid = jnp.linspace(z_init, 30, 1000)
f_grid = 1420.4 / (1 + z_grid)

probes = (
    "COM_CosmoParams_fullGrid_R3.01/base/"
    + "plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE"
)

chains = read_chains(probes)
params = ["H0", "omegam", "omegabh2", "omegach2", "yheused"]
chains = chains[params]
chains.compress()
samples = jnp.mean(chains.values, axis=0)

recT21, recxe, recTk, recxc = generate_signal(
    f_grid,
    samples,
    z_init,
    rec_model="recfast",
    detailed_output=True,
    verbose=True,
)
hyrec_T21, hyrec_xe, hyrec_Tk, hyrec_xc = generate_signal(
    f_grid,
    samples,
    z_init,
    rec_model="hyrec",
    detailed_output=True,
    verbose=True,
)

fig, axes = plt.subplots(2, 2, figsize=(6, 5))
axes = axes.flatten()
axes[0].plot(z_grid, recxe, label="Recfast")
axes[0].plot(z_grid, hyrec_xe, label="HyRec")
axes[0].set_xlabel("Redshift $z$")
axes[0].set_ylabel("Free electron fraction $x_e$")
axes[0].legend()

axes[1].plot(z_grid, recTk, label="Recfast")
axes[1].plot(z_grid, hyrec_Tk, label="HyRec")
axes[1].set_xlabel("Redshift $z$")
axes[1].set_ylabel("Gas temperature $T_k$ [K]")
axes[1].legend()

axes[2].plot(z_grid, recxc, label="Recfast")
axes[2].plot(z_grid, hyrec_xc, label="HyRec")
axes[2].set_xlabel("Redshift $z$")
axes[2].set_ylabel("Collisional coupling $x_c$")
axes[2].legend()

axes[3].plot(z_grid, recT21, label="Recfast")
axes[3].plot(z_grid, hyrec_T21, label="HyRec")
axes[3].set_xlabel("Redshift $z$")
axes[3].set_ylabel("21cm brightness temperature $T_{21}$ [mK]")
axes[3].legend()
plt.savefig("compare-hyrec-recfast-outputs.pdf", bbox_inches="tight")
plt.show()

print(jnp.min(recxe), jnp.min(hyrec_xe))
print(jnp.max(recxe), jnp.max(hyrec_xe))
print(jnp.min(recTk), jnp.min(hyrec_Tk))
print(jnp.max(recTk), jnp.max(hyrec_Tk))

frac_dif = (hyrec_xe - recxe) / recxe

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
# axes = [axes]
axes[0].plot(z_grid, frac_dif)

axes[0].fill_between(
    [30, 1100],
    frac_dif.min() * 1.1,
    frac_dif.max() * 1.1,
    color="gray",
    alpha=0.5,
)
# axes[0].set_ylim(frac_dif.min() * 1.1, frac_dif.max() * 1.1)
axes[0].axhline(0, color="k", ls="--")
axes[0].set_xlabel("Redshift $z$")
axes[0].set_ylabel(
    r"$(x_e^\mathrm{hyrec} - x_e^\mathrm{recfast})/x_e^\mathrm{recfast}$"
)

frac_dif_Tk = (hyrec_Tk - recTk) / recTk
axes[1].plot(z_grid, frac_dif_Tk)
axes[1].fill_between(
    [30, 1100],
    frac_dif_Tk.min() * 1.1,
    frac_dif_Tk.max() * 1.1,
    color="gray",
    alpha=0.5,
)
axes[1].set_ylim(frac_dif_Tk.min() * 1.1, frac_dif_Tk.max() * 1.1)
axes[1].axhline(0, color="k", ls="--")
axes[1].set_xlabel("Redshift $z$")
axes[1].set_ylabel(r"$(T_k^{HyRec} - T_k^{Recfast})/T_k^{Recfast}$")

plt.tight_layout()
plt.savefig("compare-hyrec-recfast.pdf", bbox_inches="tight")
plt.show()
