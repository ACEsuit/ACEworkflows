function isadm(bb, totaldegree)
   ord = length(bb)
   if ord > length(totaldegree)
      return false 
   end 
   tdeg = totaldegree[ord]
   if sum(b.n-1 + 2*b.l for b in bb) > tdeg
      return false
   end
   return true
end

# reconstruct the basis specification. Since the 2-body 
# doesn't have the spec_nl implemented, we have to do a bit by hand. 
spec_2 = [ [(z = AtomicNumber(:Fe), n = n-1, l = 0),] 
             for n = 1:length(model.basis.BB[1]) ]
spec_mb = ACE1.get_nl(model.basis.BB[2])
spec = [ spec_2; spec_mb ]
@assert length(spec) == length(model.basis)

## 

# some typical total-degree values
# totaldegree = [ 20, 16, 10 ]
# totaldegree = [ 24, 19, 14 ]
# totaldegree = [ 27, 22, 17 ]
# totaldegree = [ 27, 22, 17, 12 ]
# totaldegree = [ 28, 23, 18, 12 ]
# totaldegree = [ 30, 25, 20, 13 ]
# totaldegree = [ 32, 27, 22, 14 ]

## 

# by first performing a QR factorisation we can compress the linear 
# system to a much smaller one so we can then solve it many times 
# for testing ... 

qr_Ap = qr(Ap)
Q, R = qr_Ap 
Qt_Yp = R * (qr_Ap \ Yp)
Q = nothing 

##

# This test is for determining the right rtol for the RRQR solver.
# which is basically the regularisation parameter 

TDEGS = [
   # [ 20, 16, 10 ], 
   # [ 24, 19, 14 ], 
   # [ 27, 22, 17 ], 
   [ 27, 22, 17, 12 ], 
   [ 28, 23, 18, 12 ],
   [ 30, 25, 20, 13 ],
   [ 32, 27, 22, 14 ]  ]

RTOLS = [ # 1e-12, 1e-12, 1e-11, 
         1e-14, 1e-13, 1e-12, 1e-11, 1e-10 ]

TRAINERRS = []
TESTERRS = []

FErr = zeros(length(TDEGS), length(RTOLS))
EErr = zeros(length(TDEGS), length(RTOLS))

for (ideg, totaldegree) in enumerate(TDEGS), 
         (itol, rtol) in enumerate(RTOLS)
   iB = findall( bb -> isadm(bb, totaldegree), spec )
   # println(i, " : length = ", length(iB))   
   R_i = R[:, iB]
   solver = ACEfit.RRQR(rtol = rtol)
   result = ACEfit.linear_solve(solver, R_i, Qt_Yp)
   c = zeros(length(model.basis))
   c[iB] = P[iB, iB] \ result["C"]   
   ACE1pack._set_params!(model, c)

   @info("TEST ERRORS: ")
   testerr = ACE1pack.linear_errors(testing_data, model; datakeys...)

   FErr[ideg, itol] = testerr["rmse"]["set"]["F"]
   EErr[ideg, itol] = testerr["rmse"]["set"]["E"]
end

@info("Energy Error")
display(EErr)

@info("Force Error")
display(FErr)



##

TDEGS = [
   [ 20, 16, 10 ], 
   [ 24, 19, 14 ], 
   [ 27, 22, 17 ], 
   [ 27, 22, 17, 9 ], 
   [ 27, 22, 17, 12 ], 
   [ 28, 23, 18, 12 ],
   [ 30, 25, 20, 13 ],
   [ 32, 27, 22, 14 ], 
   [ 40, 40, 40, 40 ]  ]

Blen = zeros(Int, length(TDEGS))
FErr = zeros(length(TDEGS))
EErr = zeros(length(TDEGS))

for (ideg, totaldegree) in enumerate(TDEGS)
   rtol = 1e-13 
   iB = findall( bb -> isadm(bb, totaldegree), spec )
   Blen[ideg] = length(iB)
   println(ideg, " : length = ", length(iB))
   R_i = R[:, iB]
   solver = ACEfit.RRQR(rtol = rtol)
   result = ACEfit.linear_solve(solver, R_i, Qt_Yp)
   c = zeros(length(model.basis))
   c[iB] = P[iB, iB] \ result["C"]   
   ACE1pack._set_params!(model, c)

   # @info("TRAINING ERRORS: ")
   # push!(TRAINERRS, ACE1pack.linear_errors(training_data, model; datakeys...))
   @info("TEST ERRORS: ")
   testerr = ACE1pack.linear_errors(testing_data, model; datakeys...)

   FErr[ideg] = testerr["rmse"]["set"]["F"]
   EErr[ideg] = testerr["rmse"]["set"]["E"]
end

##

@info("Len / E / F ")
display(Any[Blen EErr FErr])

##
# should also double-check that the dimer looks ok. 

##

rr = range(0.1, 6.5, length=200)
yy = zeros(length(rr))
for (i, r) in enumerate(rr)
   at = Atoms(X = [[0,0,0], [r,0,0]], Z = [26, 26], pbc=false, 
              cell = [2*r 0 0; 0 1 0; 0 0 1])
   yy[i] = ACE1pack.energy(model.potential, at)
end

##

using Plots 

plot(rr, yy, label="", xlabel="r / Å", ylabel="E / eV", 
     legend=:topright, linewidth=2, 
     ylim = [-2000, 2000])
