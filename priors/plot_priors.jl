
include("utils.jl")
using ACE1pack, Plots, LaTeXStrings, ACE1
using ACE1.Transforms: transform
# Plots.pythonplot()
# Plots.plotly()
Plots.pyplot()
default(fontfamily = "serif")
# Plots.rc("font",family="serif")
# Plots.gr()

##

datapath = joinpath(ACE1pack.artifact("ZuoEtAl2020"), "ZuoEtAl2020")
# syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]
sym = :Si
dataset = JuLIP.read_extxyz(joinpath(datapath, "$(sym)_train.xyz"))

r0 = rnn(sym)
rcut = 5.0

##

_trans = ACE1.Transforms.agnesi_transform(r0, 6, 6)
trans_a = r -> 1 - _trans(r)

rr = rdf(dataset, rcut; ndat=10_000)
minr = minimum(rr)
ra = 0.0; rb = rcut 
xa = -0.03; xb = 1.0 

xx = trans_a.(rr)
x0 = trans_a(r0)
xmin, xmax = extrema(xx)



histogram(rr, nbins=60, label = L"g(r)", legend = :topleft, 
          ytick = [], xtick = [], yflip=true, 
          xlims = [0, rcut], lw=0, legend_foreground_color = :white, 
          axis = false, c = 5, )
plt1 = vline!([r0,], lw=2, c=2,  label = "")
plt1 = vline!([minr, 3], lw=1, c=2,  label = "")



rp = range(0.000001, rcut, length=200)

# background lines 
xticks = [minr, r0, 3]
_xticks_lw = [1, 2, 1]
_xticks_c = [2, 2, 2]

plt2 = plot() 
for (x, lw, c) in zip(xticks, _xticks_lw, _xticks_c)
   y = trans_a(x)
   plot!(plt2, [x, x], [xa, trans_a(x)], lw=lw, c=c, label = "")
   plot!(plt2, [x, rb], [y, y], lw=lw, c=c, label = "")
end


plot!(plt2, rp, trans_a.(rp), xlim = [ra, rb], ylim = [xa, xb], lw=3, 
         ylabel = L"y~~[{\rm nondim.}]",  xlabel = L"r~[\AA]", 
         xticks = ([0, 1, minr, r0, 3, 4, 5], 
                   ["0", "1", L"r_{\min}", L"r_0", "3", "4", "5"]), 
         yticks = [], 
         label = "", # legend = :topleft, 
         grid = false, c = 1
         )


plt3 = histogram(xx, nbins=100, label = L"g(y)", legend = :bottomleft, 
            orientation=:horizontal, ylim = [xa, xb], 
            lw=0, c = 3, ytick = [],  xtick = [], 
            xlim = [0.001, 300,], legend_foreground_color = :white, 
            grid=false, axis=false, )

hline!(trans_a.(xticks), lw=1, c=2, label = "")
hline!([x0,], lw=2, c=2, label = "")


plot(plt2, plt3, plt1, 
      layout = grid(2, 2, heights=[0.8, 0.2], widths=[0.8, 0.2]),
      size = (400, 400), )

##

savefig(@__DIR__() * "/transform.pdf")   


## ------------------------------------
#  transformed potential 

f(r) = min((r0/r)^(10) - 2*(r0/r)^5, 1_000_000)

plt1 = plot(rp, f.(rp), xlabel = L"r~[\AA]", yticks = [], xlim = [ra, rb],
                   ylim = [-1.5, 2], lw=2, label = L"V(r)", 
                   legend_foreground_color = :white, grid = false)


rp_ = range(0.01, rb, length = 300) 
xp_ = trans_a.(rp_)

xp = trans_a.(rp)
plt2 = plot(xp, f.(rp), xlim = [xa, xb], ylim = [-1.5, 2], 
                  xlabel = L"y~~[{\rm nondim.}]", yticks = [], xticks = [], 
                  label = L"V(y)", lw = 2, )

e_guess(r) = min((r0/r)^(3), 1_000) + (r0/r)
plot!(plt2, xp_, f.(rp_) ./ e_guess.(rp_), 
            label = L"V(y) / f_{\mathrm{env}}(y)", lw = 2)

e_ideal(r) = min((r0/r)^(10), 1_000_000) + (r0/r)
plot!(plt2, xp_, f.(rp_) ./ e_ideal.(rp_), legend_foreground_color = :white, 
            label = L"V(y) / f_{\mathrm{env}}^{\mathrm{ideal}}(y)", lw=2)


plot(plt1, plt2, layout = grid(1, 2, widths = [0.5, 0.5]), 
            size = (400, 300),)

##

savefig(@__DIR__() * "/transformed_potential.pdf")