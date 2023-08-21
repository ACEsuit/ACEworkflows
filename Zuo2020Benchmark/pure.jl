#
# This script compares a few experimental basis functions, this is only of 
# interest to developers for now. 
#


using Distributed 
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, JuLIP, LinearAlgebra, PrettyTables

##

# the dataset is provided via ACE1pack artifacts as a convenient benchmarkset
syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]

totaldegree_sm = [ 20, 16, 12 ]   # small model: ~ 300  basis functions
totaldegree_lge = [ 25, 21, 17 ]  # large model: ~ 1000 basis functions              
totaldegree = totaldegree_lge

# In the high data regime most models behave similarly. 
# We are interested in the large basis, low data regime in these tests. 
# use data_step = 5-10 for medium size dataset, 20-50 for small dataset
data_step = 20    

## 

labels = ["dirty", "pure2b", "clean"]
_pure2b = Dict("dirty" => false, "pure2b" => true, "clean" => false)
_pure = Dict("dirty" => false, "pure2b" => false, "clean" => true)
err = Dict([b => Dict("E" => Dict(), "F" => Dict()) for b in labels]...)

for sym in syms 
   @info("---------- fitting $(sym) ----------")
   train, test, _ = ACEpotentials.example_dataset("Zuo20_$sym")
   train = train[1:data_step:end]

   # specify the models 
   for label in labels 
      model = acemodel(elements = [sym,], order = 3, totaldegree = totaldegree, 
                        pure2b = _pure2b[label], pure = _pure[label], 
                        delete2b = true) 

      @info("$sym, $label, length = $(length(model.basis))")

      # train the model 
      acefit!(model, train; solver=ACEfit.BLR())
   
      # compute and store errors for later visualisation
      err_  = ACEpotentials.linear_errors(test,  model)
      err[label]["E"][sym] =  err_["mae"]["set"]["E"] * 1000
      err[label]["F"][sym] =  err_["mae"]["set"]["F"]
   end
end


##

header = ([ "", "dirty", "pure2b", "pure"])
e_table_gap_mtp = [ 0.42  0.48; 0.46  0.41; 0.49  0.49; 2.24  2.83; 2.91  2.21; 2.06  1.79]
f_table_gap_mtp = [ 0.02  0.01; 0.01  0.01; 0.01  0.01; 0.09  0.09; 0.07  0.06; 0.05  0.05]
      
e_table = hcat(string.(syms),
         [ [round(err[label]["E"][sym], digits=3) for sym in syms]
           for label = labels ]...)     
f_table = hcat(string.(syms),
           [ [round(err[label]["F"][sym], digits=3) for sym in syms]
             for label = labels ]...)           

println("Energy Error")         
pretty_table(e_table; header = header)

println("Force Error")         
pretty_table(f_table; header = header)

