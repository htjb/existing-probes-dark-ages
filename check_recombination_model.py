"""Check which recombinaiton model was used in each probe's MCMC chains."""

probes = [  # planck cmb baseline high l TT + low l  and low EE
    "COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE",
    # wmap cmb full temperature and polarisation 9 year
    "COM_CosmoParams_fullGrid_R3.01/base/WMAP/base_WMAP",
    # bao DR12, MGS, 6dF
    "COM_CosmoParams_fullGrid_R3.01/base/BAO_Cooke17/base_BAO_Cooke17",
    # lensprior is Standard base parameters with ns = 0.96 ± 0.02,
    # Ωbh2 = 0.0222 ± 0.0005, 100>H0>40, τ=0.055
    # des cosmic shear + galaxy auto + cross
    "COM_CosmoParams_fullGrid_R3.01/base/DES_lenspriors/base_DES_lenspriors",
]

rec_model = {}
for i in range(len(probes)):
    file_path = probes[i]
    with open(file_path + ".inputparams") as f:
        input_params = f.readlines()
        input_params = [
            line.strip()
            for line in input_params
            if line.strip() and not line.startswith("#")
        ]
        input_params = {
            line.split("=")[0].strip(): line.split("=")[1].strip()
            for line in input_params
        }
        print(input_params.keys())
        exit()
    rec_model[probes[i]] = input_params.get(
        "Compiled_Recombination", "unknown"
    )

print("Recombination models used in each probe:")
for probe, model in rec_model.items():
    print(f"{probe.split('/')[-1]}: {model}")
