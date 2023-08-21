This folder contains two scripts that perform some basic benchmarks on that dataset taken from. 

```quote 
Performance and Cost Assessment of Machine Learning Interatomic Potentials
Yunxing Zuo, Chi Chen, Xiangguo Li, Zhi Deng, Yiming Chen, Jörg Behler, Gábor Csányi, Alexander V. Shapeev, Aidan P. Thompson, Mitchell A. Wood, and Shyue Ping Ong
The Journal of Physical Chemistry A 2020 124 (4), 731-745
DOI: 10.1021/acs.jpca.9b08723
```

- `ACEpotentials_paper.jl` : this reproduces the results of an article in which we introduce `ACEpotentials.jl` (todo - update this once published)
- `pure.jl` : this script is an exploration of the impure, vs partially pure, vs fully purified ACE basis


To rerun those results, install a compatible Julia version, then from a shell in the current folder run
```bash
julia --project=. -e 'using Pkg; Pkg.up()' 
julia --project=. ACEpotentials_paper.jl
julia --project=. pure.jl
```
