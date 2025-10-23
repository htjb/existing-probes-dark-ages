import matplotlib.pyplot as plt
from anesthetic import read_chains
import emcee
from anesthetic import MCMCSamples


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

for i in range(len(probes)):
    file_path = probes[i]
    chains = read_chains(file_path)
    params = ['H0', 'omegam', 'omegabh2', 'omegach2', 'yheused']
    chains = chains[params]

    chains = chains.compress()
    samples = chains.values

    # autocorrelation time is like a measure of independence
    # and tau is how many steps to skip to get independent samples
    tau = emcee.autocorr.integrated_time(samples, tol=0)
    samples = samples[::int(tau)]#[:300]
    chains = MCMCSamples(samples, columns=params)
    
    if i == 0:
        ax = chains.plot_2d(params, kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'},
                            alpha=0.5, label = probes[i].split('/')[-1],
                            figsize=(7, 7))
    else:
        chains.plot_2d(ax, kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'},
                            alpha=0.5, label = probes[i].split('/')[-1])
ax.iloc[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chains_comparison.png')
plt.show()


