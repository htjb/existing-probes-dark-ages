from jax import numpy as jnp
import jax
from anesthetic import read_chains
import matplotlib.pyplot as plt
from cmrt.cosmology import cosmology
from cmrt.signal import xc, Tcmb, Ts, T21
from cmrt.hyrec import set_up_hyrec, call_hyrec
from cmrt.recfast import update_recfast_ini, call_recfast


def generate_signal(z_grid, sample, model='recfast'):
    cosmo = cosmology(H0=sample[0], Omega_m=sample[1],
                      Omega_b=sample[2]/(sample[0]/100)**2,
                      Omega_c=sample[3]/(sample[0]/100)**2,
                      Omega_bh2=sample[2], Omega_ch2=sample[3],
                      z_init=z_init, Y_He=sample[4])  # Example cosmology parameters
    
    if model == 'recfast':
        update_recfast_ini(cosmo)
        xe, T_gas = call_recfast(base_dir='./', redshift=z_grid)
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)
    elif model == 'hyrec':
        set_up_hyrec(H0=cosmo.H0, omb=cosmo.Omega_b,
                    omc=cosmo.Omega_c, 
                    omk=0, yhe=cosmo.Y_He,
                    base_dir='./')
        xe, T_gas = call_hyrec(base_dir='./', redshift=z_grid)
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)

    xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

    T_s = Ts(T_gas, T_cmb, xc_values)
    T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
    return T21_values, xe, T_gas, xc_values

z_init = 2000
z_grid = jnp.linspace(z_init, 30, 1000)

# H0, Omega_m, Omega_l, Omega_b, ns, As
T_cmb = Tcmb(z_grid)
xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


probes = '/Users/harry/Documents/research-adjacent-projects/dark-ages-simulation/' + \
        'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE'
          

chains = read_chains(probes)
params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
chains = chains[params]
chains.compress()
samples = jnp.mean(chains.values, axis=0)

T21_rec, recxe, recTk, rec_xc = generate_signal(z_grid, samples, model='recfast')
T21_hyrec, hyrec_xe, hyrec_Tk, hyrec_xc = generate_signal(z_grid, samples, model='hyrec')

print(jnp.min(recxe), jnp.min(hyrec_xe))
print(jnp.max(recxe), jnp.max(hyrec_xe))
print(jnp.min(recTk), jnp.min(hyrec_Tk))
print(jnp.max(recTk), jnp.max(hyrec_Tk))

frac_dif = (hyrec_xe - recxe)/recxe

fig, axes = plt.subplots(1, 1, figsize=(3, 2.5))
axes = [axes]
axes[0].plot(z_grid, frac_dif)

axes[0].fill_between([30, 1100], frac_dif.min()*1.1, frac_dif.max()*1.1, color='gray', alpha=0.5)
axes[0].set_ylim(frac_dif.min()*1.1, frac_dif.max()*1.1)
axes[0].axhline(0, color='k', ls='--')
axes[0].set_xlabel('Redshift $z$')
axes[0].set_ylabel(r'$(x_e^\mathrm{hyrec} - x_e^\mathrm{recfast})/x_e^\mathrm{recfast}$')

"""frac_dif_Tk = (hyrec_Tk - recTk)/recTk
axes[1].plot(z_grid, frac_dif_Tk)
axes[1].fill_between([30, 1100], frac_dif_Tk.min()*1.1, frac_dif_Tk.max()*1.1, color='gray', alpha=0.5)
axes[1].set_ylim(frac_dif_Tk.min()*1.1, frac_dif_Tk.max()*1.1)
axes[1].axhline(0, color='k', ls='--')
axes[1].set_xlabel('Redshift $z$')
axes[1].set_ylabel(r'$(T_k^{HyRec} - T_k^{Recfast})/T_k^{Recfast}$')"""

plt.tight_layout()
plt.savefig('compare-hyrec-recfast.pdf', bbox_inches='tight')
plt.show()