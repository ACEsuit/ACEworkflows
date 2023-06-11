using ACE1pack
using ACE1, IPFitting
using Random
using InvertedIndices

data_file = "../train.xyz"

data = IPFitting.Data.read_xyz(data_file, energy_key="energy", force_key="force", virial_key="virial")

R = minimum(IPFitting.Aux.rdf(data, 4.0))

# split training and testing sets -> 90:10
# number of configurations
Nconfigs = length(data)
Ntest = round(Int, Nconfigs * 0.1)

#index of training and testing data
# This is for excluding cracktip config in testing set: ind_test = rand(1:Nconfigs-14,Ntest)
ind_test = rand(1:Nconfigs,Ntest)
ind_train = collect(1:Nconfigs)[Not(ind_test)]
training_data = data[ind_train]
testing_data = data[ind_test]

N=4
D=14
r_cut=6.5

r0 = rnn(:Fe)

Bsite = rpi_basis(species = [:Fe],
        N = N,       # correlation order = body-order - 1
        maxdeg = D,  # polynomial degree
        r0 = r0,     # estimate for NN distance
        rin = R,
        rcut = r_cut,   # domain for radial basis (cf documentation)
        pin = 2)                     # require smooth inner cutoff

Bpair = pair_basis(species = [:Fe],
        r0 = r0,
        maxdeg = D + 10,
        rcut = r_cut,  #+ 1.0,
        rin = 0.0,
        pin = 0 )   # pin = 0 means no inner cutoff

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite])
BN=length(Bsite)
Btwo=length(Bpair)
Ball=length(B)

println("Number of ACE, pair and all basis functions are $BN, $Btwo, and $Ball")

dB = LsqDB("", B, training_data)
# To load the DB again, one just need to do 'LsqDB(“database”)'

Vref = OneBody(:Fe => -3455.6995339)

weights = Dict(
    "default" => Dict("E" => 12.5, "F" => 1.0 , "V" => 0.0 ),
   "slice_sample_high" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),
   "phonons_54_high" => Dict("E" => 46.0, "F" => 1.0 , "V" => 0.0 ),
   "prim_random" => Dict("E" => 100.0, "F" => 0.0 , "V" => 0.01 ),    
   "phonons_128_high" => Dict("E" => 19.0, "F" => 1.0 , "V" => 0.0 )
   )

solver= Dict(
    "solver" => :ard,
    "ard_tol" => 1e-3,
    "ard_n_iter" => 3000,
    "ard_threshold_lambda" => 10000)

IP, lsqinfo = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)

@info("Training Error Table: $(N) $(D) $(r_cut)")
println("Training Error Table:")
rmse_table(lsqinfo["errors"])

# Testing the potential using testing set
add_fits!(IP, testing_data)
rmse_, rmserel_ = rmse(testing_data);
@info("Testing Error Table: $(N) $(D) $(r_cut)")
println("Testing Error Table:")
rmse_table(rmse_, rmserel_)

save_dict("./Fe_ace_N$(N)_D$(D)_R$(r_cut).json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
ACE1pack.ExportMulti.export_ACE("./Fe_ace_N$(N)_D$(D)_R$(r_cut).yace", IP)

