import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import jax
from darkages.cosmology import cosmology
from darkages.signal import xc, Tcmb, Ts, T21
from darkages.hyrec import set_up_hyrec, call_hyrec
from anesthetic import read_chains
from anesthetic.samples import MCMCSamples
from tqdm import tqdm
from fgivenx import plot_contours, plot_dkl
from darkages.recfast import update_recfast_ini, call_recfast
import emcee

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
        print(f"Error generating signal for sample {sample}: {e}")
        return jnp.full_like(f_grid, jnp.nan)


def get_minima(signal):
    min_indices = jnp.nanargmin(signal)
    min_values =  jnp.nanmin(signal)
    return min_values, min_indices
#for i in range(len(samples)):

z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1500)


# H0, Omega_m, Omega_l, Omega_b, ns, As
xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


probes = [# planck cmb baseline high l TT + low l  and low EE
        'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE',
            # wmap cmb full temperature and polarisation 9 year
          'COM_CosmoParams_fullGrid_R3.01/base/WMAP/base_WMAP',
            # bao DR12, MGS, 6dF
          'COM_CosmoParams_fullGrid_R3.01/base/BAO_Cooke17/base_BAO_Cooke17',
          #lensprior is Standard base parameters with ns = 0.96 ± 0.02, Ωbh2 = 0.0222 ± 0.0005, 100>H0>40, τ=0.055
          # des cosmic shear + galaxy auto + cross
          'COM_CosmoParams_fullGrid_R3.01/base/DES_lenspriors/base_DES_lenspriors',
        #'data/runs_default/chains/planck', 
          #'data/runs_default/chains/DES',
          #'data/runs_default/chains/BAO',
          #'data/runs_default/chains/SH0ES',  
          #'ormondroyd/chains/desidr2/desidr2_lcdm',
          #'ormondroyd/chains/des5y/des5y_lcdm',
          #'ormondroyd/chains/pantheonplus/pantheonplus_lcdm',
          ]

titles = ['Planck 2018 high l TT +\nlowl + lowE ', 
          'WMAP Full 9 Year', 'BAO DR12 +\nMGS + 6dF', 
          'DES Y1 Cosmic Shear +\nGalaxy Clustering + Cross'
          ]

plot_signal = True
plot_kl = False
plot_minima = False
plot_delta = False
recombination_model = 'hyrec' # 'recfast' or 'hyrec'

c = ['C0', 'C1', 'C2', 'C3']

if plot_signal:
    z_grid = jnp.linspace(z_init, 30, 500)
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 5), 
                        sharex=True)
    ax = ax.flatten()
    for i in range(len(probes)):
        file_path = probes[i]
        chains = read_chains(file_path)
        params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
        chains = chains[params]
        if i ==0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0]/100)**2
            prior[:, 3] *= (prior[:, 0]/100)**2
        chains = chains.compress()
        samples = chains.values

        # autocorrelation time is like a measure of independence
        # and tau is how many steps to skip to get independent samples
        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[::int(tau)]
        #samples = samples[:100]

        print("Number of samples in posterior:", len(samples))
        print("Number of samples in prior:", len(prior))

        plot_contours(generate_signal, 1420.4/(1+z_grid), prior, ax[i], colors=plt.cm.Blues_r,
                    alpha=0.5, lines=False,
                    fineness=1,
                    )
        plot_contours(generate_signal, 1420.4/(1+z_grid), samples, ax[i], colors=plt.cm.Reds_r,
                        alpha=0.5, lines=False,
                        fineness=1
                    )
        ax[i].set_title(titles[i], fontsize=10)
        ax[i].set_ylim(-350, 10)
        ax[i].grid()
    ax[2].set_xlabel(r'$\nu$ [MHz]')
    ax[3].set_xlabel(r'$\nu$ [MHz]')
    ax[0].set_ylabel(r'$T_{21}$ [mK]')
    ax[2].set_ylabel(r'$T_{21}$ [mK]')
    ax[-1].plot([], [], color=plt.cm.Reds_r(0.5), label='Posterior', alpha=0.5)
    ax[-1].plot([], [], color=plt.cm.Blues_r(0.5), label='Prior', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('21cm_brightness_temperature_contours_' + recombination_model + '.pdf')
    #plt.show()
    plt.close()

if plot_kl:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for i in range(len(probes)):
        file_path = probes[i]
        chains = read_chains(file_path)
        params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
        chains = chains[params]
        if i ==0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0]/100)**2
            prior[:, 3] *= (prior[:, 0]/100)**2
        chains = chains.compress()
        samples = chains.values

        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[::int(tau)]
        #samples = samples[:100]

        plot_dkl(generate_signal, 1420.4/(1+z_grid), samples, prior, ax, color=c[i],
                label=titles[i])
    plt.legend(fontsize=8)
    ax.set_xlabel(r'$\nu$ [MHz]')
    ax.set_ylabel(r'$D_{KL}$ [bits]')
    plt.tight_layout()
    plt.savefig('21cm_brightness_temperature_dkl.pdf')
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
        params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
        chains = chains[params]
        if i ==0:
            prior = prior_sample(n_samples=1000)
            prior[:, 2] *= (prior[:, 0]/100)**2
            prior[:, 3] *= (prior[:, 0]/100)**2
        chains = chains.compress()
        samples = chains.values

        # autocorrelation time is like a measure of independence
        # and tau is how many steps to skip to get independent samples
        tau = emcee.autocorr.integrated_time(samples, tol=0)
        samples = samples[::int(tau)]#[:300]

        signals = jnp.array([generate_signal(1420.4/(1+z_grid), s) 
                                             for s in tqdm(samples)])
        
        if i == 0:
            prior_signals = jnp.array([generate_signal(1420.4/(1+z_grid), s) for s in tqdm(prior)])

        if plot_minima:
            min_values, min_indices = vmapped_get_minima(signals)

            mask = ~jnp.isnan(min_values)
            min_values = min_values[mask]
            min_indices = min_indices[mask]

            minimum_freqs = 1420.4/(1+z_grid[min_indices])
            minima_samples = MCMCSamples(data=jnp.array([minimum_freqs, min_values]).T,
                                         columns=['Frequency [MHz]', 'Min T21 [mK]'])
            print(f"{titles[i]}: Average Min T21 = {jnp.nanmean(min_values):.2f} mK"
                  + f" $\\pm$ {jnp.nanstd(min_values):.2f} mK \n" 
                  f"Average nu_c {jnp.nanmean(minimum_freqs):.2f} MHz" +
                  f" $\\pm$ {jnp.nanstd(minimum_freqs):.2f} MHz")
            if i == 0:
                prior_min_values, prior_min_indices = vmapped_get_minima(prior_signals)
                prior_minimum_freqs = 1420.4/(1+z_grid[prior_min_indices])
                print(f"Prior: Average Min T21 = {jnp.nanmean(prior_min_values):.2f} mK"
                      + f" $\\pm$ {jnp.nanstd(prior_min_values):.2f} mK \n" 
                      f"Average nu_c {jnp.nanmean(prior_minimum_freqs):.2f} MHz" +
                      f" $\\pm$ {jnp.nanstd(prior_minimum_freqs):.2f} MHz")
            if i == 0:
                try:
                    ax = minima_samples.plot_2d(['Frequency [MHz]', 'Min T21 [mK]'], 
                                            color=c[i],
                                            alpha=0.5,
                                            label=titles[i].split(' ')[0], 
                                            figsize=(3.5, 3.5),
                                            kinds={'lower': 'kde_2d'})
                except:
                    ax = minima_samples.plot_2d(['Frequency [MHz]', 'Min T21 [mK]'], 
                                            color=c[i],
                                            alpha=0.5,
                                            label=titles[i].split(' ')[0], 
                                            figsize=(3.5, 3.5),
                                            kinds={'lower': 'scatter_2d'})
            else:
                minima_samples.plot_2d(ax, color=c[i],
                                        alpha=0.5,
                                        label=titles[i].split(' ')[0],
                                        kinds={'lower': 'kde_2d'})
if plot_minima:
    ax.iloc[0, 0].set_xlabel(r'$\nu_c$ [MHz]')
    ax.iloc[0, 0].set_ylabel(r'$T_{21}(\nu_c)$ [mK]')
    plt.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig('21cm_brightness_temperature_minima_' + recombination_model + '.pdf')
plt.close()
