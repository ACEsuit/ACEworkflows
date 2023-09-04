The script `fit_model.jl` fits the Silicon potential reported in `ACEpotentials.jl` (todo - update this once published).
The dataset was taken from

```
Bartók, Albert P., James Kermode, Noam Bernstein, and Gábor Csányi.
"Machine learning a general-purpose interatomic potential for silicon."
Physical Review X 8, no. 4 (2018): 041048.
```

and can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1250555.svg)](https://doi.org/10.5281/zenodo.1250555) under `models/GAP/gp_iter6_sparse9k.xml.xyz`


To run: install a compatible Julia version, then from a shell in the current folder execute
```bash
julia --project=. -e 'using Pkg; Pkg.up()'
julia --project=. fit_model.jl
```
