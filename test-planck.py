from anesthetic import read_chains
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
from cmrt.cosmology import cosmology
from cmrt.signal import xc, Tcmb, Ts, T21
from cmrt.hyrec import set_up_hyrec, call_hyrec
from cmrt.recfast import update_recfast_ini, call_recfast


def generate_signal(f_grid, sample, model='recfast'):
    cosmo = cosmology(H0=sample[0], Omega_m=sample[1],
                      Omega_b=sample[2]/(sample[0]/100)**2,
                      Omega_c=sample[3]/(sample[0]/100)**2,
                      Omega_bh2=sample[2], Omega_ch2=sample[3],
                      z_init=z_init, Y_He=sample[4],)  # Example cosmology parameters
    
    if model == 'recfast':
        update_recfast_ini(cosmo)
        z_grid = 1420.4/(f_grid)-1
        xe, T_gas = call_recfast(base_dir='./', redshift=z_grid)
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)
    elif model == 'hyrec':
        set_up_hyrec(H0=cosmo.H0, omb=cosmo.Omega_b,
                    omc=cosmo.Omega_c, 
                    omk=0, yhe=cosmo.Y_He,
                    base_dir='./')
        z_grid = 1420.4/(f_grid)-1
        xe, T_gas = call_hyrec(base_dir='./', redshift=z_grid)
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)

    xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

    T_s = Ts(T_gas, T_cmb, xc_values)
    T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
    return T21_values, xe, T_gas, xc_values


z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1000)

# H0, Omega_m, Omega_l, Omega_b, ns, As
T_cmb = Tcmb(z_grid)
xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


probes = '/Users/harry/Documents/research-adjacent-projects/dark-ages-simulation/' + \
        'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE'
          

chains = read_chains(probes)
params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused', 'logL']
chains = chains[params]
chains.compress()
maxlike_samples = chains.values[np.argmax(chains['logL'])][:-1]
mean_samples = np.mean(chains.values, axis=0)[:-1]

mondal_barkana_params = jnp.array([67.66, 0.14240/0.6766**2,
                                   0.02242,  0.11933, mean_samples[4]])
samples = mondal_barkana_params

xes = []
for m in ['recfast', 'hyrec']:
    try:
        signal, xe, Tgas, xcv = generate_signal(1420.4/(z_grid + 1), samples,
                                            model=m)
        plt.plot(1420.4/(z_grid + 1), signal, label=m)
        print(jnp.min(signal))
        xes.append(xe)
    except Exception as e:
        print(f"Error generating signal for model {m}: {e}")
        xes.append(jnp.array([]))
plt.legend()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Brightness Temperature [mK]')
plt.show()

plt.plot(z_grid, (xes[1]-xes[0])/xes[0])
plt.xlabel('Redshift')
plt.ylabel('Delta xe/xe')
plt.show()