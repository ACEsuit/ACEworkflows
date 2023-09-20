The script `fit_model.jl` fits the Silicon potential reported in `ACEpotentials.jl` (todo - update this once published).
The dataset was taken from

```
Bartók, Albert P., James Kermode, Noam Bernstein, and Gábor Csányi.
"Machine learning a general-purpose interatomic potential for silicon."
Physical Review X 8, no. 4 (2018): 041048.
```

and can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1250555.svg)](https://doi.org/10.5281/zenodo.1250555) under `models/GAP/gp_iter6_sparse9k.xml.xyz`


To fit: install a compatible Julia version, then from a shell in the current folder execute
```bash
julia --project=. -e 'using Pkg; Pkg.up()'
julia --project=. fit_model.jl
```

To run the fracture test, you need to install `matscipy`

```bash
pip3 install matscipy
cd $HOME
git clone https://github.com/libAtoms/matscipy.git
cd -
python3 $HOME/matscipy/scripts/fracture_mechanics/quasistatic_crack.py <pot_name>
```

where `<pot_name>` is the basename of the ACE model fitted above, e.g. `Si_4_20` if using the `fit_model.jl` script unmodified. The quasi-static fracture simulation parameters are taken from the file `params.py` in this directory, which needs to be the current working directory when you run the script.
