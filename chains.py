import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
from darkages.cosmology import cosmology
from darkages.signal import xc, Tcmb, Ts, T21
from darkages.hyrec import set_up_hyrec, call_hyrec
from darkages.recfast import update_recfast_ini, call_recfast
from anesthetic import read_chains, MCMCSamples
import emcee
from anesthetic.plot import kde_contour_plot_2d

def generate_signal(f_grid, sample, rec_model='recfast'):
    cosmo = cosmology(H0=sample[0], Omega_m=sample[1],
                      Omega_b=sample[2]/(sample[0]/100)**2,
                      Omega_c=sample[3]/(sample[0]/100)**2,
                      Omega_bh2=sample[2], Omega_ch2=sample[3],
                      z_init=z_init, Y_He=sample[4])  # Example cosmology parameters
    try:
        if rec_model == 'recfast':
            update_recfast_ini(cosmo)
            z_grid = 1420.4/(f_grid)-1
            xe, T_gas = call_recfast(base_dir='./', redshift=z_grid)
            xe, T_gas = jnp.array(xe), jnp.array(T_gas)
        elif rec_model == 'hyrec':
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

z_init = 1100
z_grid = jnp.linspace(z_init, 30, 1000)

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
          ]
probes = probes[::-1]  # reverse order for plotting
colors = ['C0', 'C1', 'C2', 'C3'][::-1]
params = ['H0', 'omegam', 'omegabh2']#, 'omegach2', 'yheused']
#fig, ax = make_1d_axes(params, figsize=(6.3, 3), facecolor='w', ncol=3)

fig, all_axes = plt.subplots(1, 3, figsize=(6, 2.5),
                         sharex=True, sharey=True)
axes = all_axes[1:]
contour_axes = all_axes[0:1]
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
    samples = samples[::int(tau)]#[:300]
    chains = MCMCSamples(samples, columns=params)
    omegabh2 = chains['omegabh2']
    H0 = chains['H0']
    mins.append([H0.min(), omegabh2.min()])
    maxs.append([H0.max(), omegabh2.max()])
    kde_contour_plot_2d(contour_axes[0],
                        H0, omegabh2,
                        levels=[0.95, 0.68],
                        color=colors[i],
                        label=file_path.split('/')[1],
                        )
#ax.iloc[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
mins = jnp.array(mins)
maxs = jnp.array(maxs)
overall_min = mins.min(axis=0)
overall_max = maxs.max(axis=0)

recombination_model = ['hyrec', 'recfast'] # 'recfast' or 'hyrec'


parameters = ['H0', 'Omega_m', 'Omega_b', 'Omega_c', 'Y_He']
titles = [r'$H_0$', r'$\Omega_m$', r'$\Omega_b$', r'$\Omega_c$', r'$Y_{He}$']

X, Y = jnp.meshgrid(jnp.linspace(50, 80, 20), jnp.linspace(0.01, 0.05, 20))

#X, Y = jnp.meshgrid(jnp.linspace(overall_min[0], overall_max[0], 20),
#                    jnp.linspace(overall_min[1], overall_max[1], 20))

print("Generating signals over parameter grid...")
fiducial_sample = jnp.array([66.89, 0.321, 0.022, 0.121, 0.24])
maxima = []
for rec_model in recombination_model:
    print(f"Using recombination model: {rec_model}")
    Z = jnp.empty(X.shape)
    if rec_model == 'hyrec':
        all_samples = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sample = jnp.array([X[i, j], 0.321,
                                Y[i, j],
                                (0.321 - Y[i, j]/(X[i, j]/100)**2)*(X[i, j]/100)**2,
                                0.24])
            if rec_model == 'hyrec':
                all_samples.append(sample)
            signal = generate_signal(1420.4/(1+z_grid), sample,
                                     rec_model)
            Z= Z.at[i, j].set(jnp.nanmax(jnp.abs(signal)))
    maxima.append(Z)
maxima = jnp.array(maxima)
all_samples = jnp.array(all_samples)

for i in range(len(recombination_model)):
    ax = axes[i]
    ax.set_title(recombination_model[i])
    ax.set_title(titles[i])
    c = ax.contourf(X, Y, maxima[i], levels=20, cmap='viridis',
                    vmin=0, vmax=jnp.nanmax(maxima))
    fig.colorbar(c, ax=ax, label=r'Max $|T_{21}|$ [mK]')
                 #vmin=0, vmax=jnp.nanmax(maxima))
    # axis are H0 and Omega_m
    ax.set_title(recombination_model[i])
    ax.set_xlabel(r'$H_{0}$ [km/s/Mpc]')
    if i == 0:
        ax.set_ylabel(r'$ \Omega_{b}$') 
    ax.plot(fiducial_sample[0], fiducial_sample[2],
            marker='x', color='red', markersize=10,
            label='Fiducial parameters')
plt.tight_layout()
plt.show()
