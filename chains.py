import matplotlib.pyplot as plt
from anesthetic import read_chains


probes = [#'data/runs_default/chains/DES',
          'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE',
          'COM_CosmoParams_fullGrid_R3.01/base/WMAP/base_WMAP',
          'COM_CosmoParams_fullGrid_R3.01/base/BAO_Cooke17/base_BAO_Cooke17',
          'COM_CosmoParams_fullGrid_R3.01/base/DES_lenspriors/base_DES_lenspriors',
          'COM_CosmoParams_fullGrid_R3.01/base/DESlens_lenspriors/base_DESlens_lenspriors',
          #'data/runs_default/chains/BAO',
          #'data/runs_default/chains/SH0ES',  
          #'data/runs_default/chains/planck', 
          #'ormondroyd/chains/desidr2/desidr2_lcdm',
          #'ormondroyd/chains/des5y/des5y_lcdm',
          #'ormondroyd/chains/pantheonplus/pantheonplus_lcdm',
          ]

for i in range(len(probes)):
    file_path = probes[i]
    chains = read_chains(file_path)
    params = ['H0', 'omegam', 'omegabh2', 'omegal', 'omegach2', 'yheused']
    chains = chains[params]
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


