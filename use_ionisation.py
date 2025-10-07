import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
from cmrt.utils.cosmology import cosmology
from cmrt.physics.collisional import xc
from cmrt.physics.temperatures import Tcmb, Ts, T21
from cmrt.physics.hyrec import set_up_hyrec, call_hyrec
from anesthetic import read_chains
from tqdm import tqdm
from fgivenx import plot_contours

def generate_signal(f_grid, sample):
    cosmo = cosmology(H0=sample[0], Omega_m=sample[1],
                      Omega_b=sample[2]/(sample[0]/100)**2,
                      Omega_c=sample[4]/(sample[0]/100)**2,
                      Omega_l=sample[3], Omega_r=9e-5,
                      z_init=z_init, Y_He=sample[5])  # Example cosmology parameters
    set_up_hyrec(H0=cosmo.H0, ombh2=cosmo.Omega_b,
                 omch2=cosmo.Omega_c, 
                 omk=0, yhe=cosmo.Y_He,
                 base_dir='./')
    z_grid = 1420.4/(f_grid)-1
    try:
        xe, T_gas = call_hyrec(base_dir='./', redshift=z_grid)
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)

        xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

        T_s = Ts(z_grid, T_gas, T_cmb, xc_values)
        T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
        return T21_values
    except:
        return jnp.full_like(f_grid, jnp.nan)

#for i in range(len(samples)):

z_init = 1100
z_grid = jnp.linspace(z_init, 30, 50)


# H0, Omega_m, Omega_l, Omega_b, ns, As
T_cmb = Tcmb(z_grid)
xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))

#cache = 'cache'
#prior_cache = 'cache_prior'

fig, ax = plt.subplots(2, 2, figsize=(6.3, 6), 
                       sharex=True)
ax = ax.flatten()

probes = ['data/runs_default/chains/planck', 
          'data/runs_default/chains/DES',
          'data/runs_default/chains/BAO',
          'data/runs_default/chains/SH0ES',  
          #'ormondroyd/chains/desidr2/desidr2_lcdm',
          #'ormondroyd/chains/des5y/des5y_lcdm',
          #'ormondroyd/chains/pantheonplus/pantheonplus_lcdm',
          ]

titles = ['Planck 2015 TT + lowl + lowTEB ', 'DES Y1 $3 \times 2$', 'BOSS BAO + RSD', 'Psuedo SH0ES']

for i in range(len(probes)):
    file_path = probes[i]
    chains = read_chains(file_path)
    print(chains.columns)
    if i == 0:
        prior = chains.prior()
    params = ['H0', 'omegam', 'omegabh2', 'omegal', 'omegach2', 'yheused']
    chains = chains[params]
    if i ==0:
        prior = prior[params]
        prior = prior.compress()
        prior = prior.values
    chains = chains.compress()
    samples = chains.values

    print("Number of samples in posterior:", len(samples))
    print("Number of samples in prior:", len(prior))

    plot_contours(generate_signal, 1420.4/(1+z_grid), prior, ax[i], colors=plt.cm.Blues_r,
                alpha=0.5,
                )
    plot_contours(generate_signal, 1420.4/(1+z_grid), samples, ax[i], colors=plt.cm.Reds_r,
                    alpha=0.5,
                )
    ax[i].set_title(titles[i])
    ax[i].set_ylim(-1000, 50)
ax[-1].plot([], [], color=plt.cm.Reds_r(0.5), label='Posterior', alpha=0.5)
ax[-1].plot([], [], color=plt.cm.Blues_r(0.5), label='Prior', alpha=0.5)
plt.xlabel(r'$\nu$ [MHz]')
plt.ylabel(r'$T_{21}$ [mK]')
plt.legend()
plt.tight_layout()
plt.savefig('21cm_brightness_temperature_contours.pdf', 
            dpi=300)
plt.show()