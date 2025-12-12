# Dark Ages 21-cm Simulation

The repo includes a simple simulation of the dark ages 21-cm signal and code to explore constraints on the signal from existing probes of $$\Lambda$$ CDM.

The main results can be obtained by running `use_ionization.py`.

## uv

I used uv to manage the project so you can set up exactly the same virtual environment I have been using by installing uv (if you don't already have it) and running `uv sync` in the command line like this

```bash
pip install uv
uv init
uv sync
```

You can then run everything in the virtual environment that uv made by running things like

```bash
uv run use_ionization.py
```

in theory!

## Other codes

The code uses HYREC-2 which can be installed by running hyrec-install.sh and recfast++ which can be installed with recfast-install.sh.

Need to be careful about the file paths assumed by the wrappers for HYREC-2 and recfast++ in `darkages/hyrec.py` and `darkages/recfast.py`. You can pass a base directory which is where the ini files with the parameters should be saved and loaded from but then the HYREC-2 wrapper assumes the output is in the `HYREC-2` folder and the recfast++ wrapper assumes it is in a folder called `recfast-output` which you might have to make.

## Data

The MCMC chains are from the [Planck Legacy Archive](https://pla.esac.esa.int/#home) and can be downloaded by running

```bash
curl "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_fullGrid_R3.01.zip" -o COM_CosmoParams_fullGrid_R3.01.zip
```

but are also included in this repo.