import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import jax
from cmrt.cosmology import cosmology
from cmrt.signal import xc, Tcmb, Ts, T21
from cmrt.hyrec import set_up_hyrec, call_hyrec
from cmrt.recfast import update_recfast_ini, call_recfast
from anesthetic import read_chains

def prior_sample(n_samples=1000):
    H0 = np.random.uniform(40, 100, n_samples)
    Omega_m = np.random.uniform(0.1, 0.5, n_samples)
    Omega_b = np.random.uniform(0.005, 0.1, n_samples)
    yhe = np.random.uniform(0.2, 0.3, n_samples)
    samples = np.vstack([H0, Omega_m, Omega_b,
                         (Omega_m - Omega_b), yhe]).T
    return samples

def generate_signal(f_grid, sample):
    cosmo = cosmology(H0=sample[0], Omega_m=sample[1],
                      Omega_b=sample[2]/(sample[0]/100)**2,
                      Omega_c=sample[3]/(sample[0]/100)**2,
                      Omega_bh2=sample[2], Omega_ch2=sample[3],
                      z_init=z_init, Y_He=sample[4])  # Example cosmology parameters
    try:
        if recombination_model == 'recfast':
            update_recfast_ini(cosmo)
            z_grid = 1420.4/(f_grid)-1
            xe, T_gas = call_recfast(base_dir='./', redshift=z_grid)
            xe, T_gas = jnp.array(xe), jnp.array(T_gas)
        elif recombination_model == 'hyrec':
            set_up_hyrec(H0=cosmo.H0, omb=cosmo.Omega_b,
                        omc=cosmo.Omega_c, 
                        omk=0, yhe=cosmo.Y_He,
                        base_dir='./')
            z_grid = 1420.4/(f_grid)-1
            xe, T_gas = call_hyrec(base_dir='./', redshift=z_grid)
            xe, T_gas = jnp.array(xe), jnp.array(T_gas)

        xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

        T_cmb = Tcmb(z_grid)

        T_s = Ts(T_gas, T_cmb, xc_values)
        T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
        return T21_values
    except Exception as e:
        #print(f"Error generating signal for sample {sample}: {e}")
        return jnp.full_like(f_grid, jnp.nan)

z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1000)


# H0, Omega_m, Omega_l, Omega_b, ns, As
T_cmb = Tcmb(z_grid)
xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


probes = [
        'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE'
          ]

recombination_model = 'hyrec' # 'recfast' or 'hyrec'

prior = prior_sample(100)
prior[:, 2] *= (prior[:, 0]/100)**2
prior[:, 3] *= (prior[:, 0]/100)**2
chains = read_chains(probes[0])
params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
chains = chains[params]
chains = chains.compress()
samples = np.mean(chains.values, axis=0)
samples = np.round(samples, 3)

fig, axes = plt.subplots(1, 3, figsize=(6, 2.5),
                         sharex=True, sharey=True)
axes = axes.flatten()
parameters = ['H0', 'Omega_m', 'Omega_b', 'Omega_c', 'Y_He']
titles = [r'$H_0$', r'$\Omega_m$', r'$\Omega_b$', r'$\Omega_c$', r'$Y_{He}$']
for i in range(len(parameters)):
    if i in [3, 4]:
        continue
    else:
        ax = axes[i if i < 3 else i-4]
    ax.set_title(titles[i])
    norm = plt.Normalize(prior[:, i].min(), prior[:, i].max())
    cmap = plt.cm.inferno_r
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue", "blue"])
    for j in range(len(prior)):
        params = samples.copy()
        params[i] = prior[j, i]
        signal = generate_signal(1420.4/(1+z_grid), jnp.array(params))
        color = cmap(norm(prior[j, i]))
        ax.plot(1420.4/(z_grid+1), signal, color=color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)#, label=titles[i])
    ax.set_xlabel(r'$\nu$ [MHz]')
    if i == 0:
        ax.set_ylabel(r'$T_{21}$ [mK]')

plt.tight_layout()
plt.savefig('21cm_brightness_temperature_variation_' 
            + recombination_model + '.pdf')
plt.show()
