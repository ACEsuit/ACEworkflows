
using JuLIP
using LinearAlgebra: norm

function _rdf(at::Atoms, rcut::Real)
   nlist = neighbourlist(at, rcut)
   return [ norm(ijr[3]) for ijr in pairs(nlist) ]
end

function rdf(dataset::AbstractVector{<: Atoms}, rcut::Real; 
             r2scale = true, ndat = 10_000)
   rr = _rdf(dataset[1], rcut)
   for at in dataset[2:end]
      rr = vcat(rr, _rdf(at, rcut))
   end
   
   if r2scale 
      rr = reweight_rr(rr)
   end

   sort!(rr)
   step = length(rr) ÷ ndat
   return rr[1:step:end]
end


function reweight_rr(rr::AbstractVector{<: Real})
   rmin, rmax = extrema(rr)
   rrnew = eltype(rr)[]
   for r in rr
      p_accept = (rmin / r)^2
      if rand() < p_accept
         push!(rrnew, r)
      end
   end
   return rrnew
end