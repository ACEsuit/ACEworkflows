
using JuLIP
using LinearAlgebra: norm

function rdf(at::Atoms, rcut::Real)
   nlist = neighbourlist(at, rcut)
   return [ norm(ijr[3]) for ijr in pairs(nlist) ]
end

function rdf(dataset::AbstractVector{<: Atoms}, rcut::Real; 
             ndat = 10_000)
   rr = rdf(dataset[1], rcut)
   for at in dataset[2:end]
      rr = vcat(rr, rdf(at, rcut))
   end
   sort!(rr)
   step = length(rr) ÷ ndat
   return rr[1:step:end]
end

function reweight_rr(rr::AbstractVector{<: Real}; f = 0.25)
   rmin = minimum(rr) 
   rmax = maximum(rr)
   rrnew = eltype(rr)[]
   for r in rr
      p_accept = 1 - (1 - f) * (r - rmin) / (rmax - rmin)
      if rand() < p_accept
         push!(rrnew, r)
      end
   end
   return rrnew
end