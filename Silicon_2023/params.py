# usage: 
#   cd /path/to/params.py
#   python ~/matscipy/scripts/fracture_mechanics/quasistatic_crack.py <pot_name>

import sys

import numpy as np

import ase.io

import ase.units as units
from ase.units import GPa, J, m

from matscipy.fracture_mechanics.clusters import diamond, set_groups
from ase.calculators.lammpslib import LAMMPSlib

import sys
pot_name = sys.argv[1]

mode = 'LAMMPS'
# mode = 'pyjulip'
assert mode in ('LAMMPS', 'pyjulip')

# via LAMMPS
if mode == 'LAMMPS':
    calc = LAMMPSlib(lmpcmds=[
                         "pair_style hybrid/overlay pace table spline 5401",
                        f"pair_coeff      * * pace {pot_name}.yace Si",
                        f"pair_coeff      1 1 table {pot_name}_pairpot.table Si_Si"
                    ],
                    atom_types={'Si': 1}, keep_alive=True, log_file='lammps.log')

# via Julia
elif mode == 'pyjulip':
    import pyjulip
    calc = pyjulip.ACE1(f"{pot_name}.json")

# Fundamental material properties
el              = 'Si'

# reuse date from diamond and surface-energy-111-relaxed tests

from ase.build import bulk
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGSLineSearch
from ase.optimize.precon import PreconLBFGS
from matscipy.elasticity import fit_elastic_constants

def find_surface_energy(symbol, calc, a0, surface, size=(8,1,1), vacuum=10, fmax=0.0001, unit='0.1J/m^2'):

    # Import required lattice builder
    if surface.startswith('bcc'):
        from ase.lattice.cubic import BodyCenteredCubic as lattice_builder
    elif surface.startswith('fcc'):
        from ase.lattice.cubic import FaceCenteredCubic as lattice_builder #untested
    elif surface.startswith('diamond'):
        from ase.lattice.cubic import Diamond as lattice_builder #untested
    ## Append other lattice builders here
    else:
        print('Error: Unsupported lattice ordering.')

    # Set orthogonal directions for cell axes
    import numpy as np
    if surface.endswith('100'):
        directions=[[1,0,0], [0,1,0], [0,0,1]] #tested for bcc
    elif surface.endswith('110'):
        directions=[[1,1,0], [-1,1,0], [0,0,1]] #tested for bcc
    elif surface.endswith('111'):
        directions=[[1,1,1], [-2,1,1],[0,-1,1]] #tested for bcc
    ## Append other cell axis options here
    else:
        print('Error: Unsupported surface orientation.')

    # Make bulk and slab with same number of atoms (size)
    bulk = lattice_builder(directions=directions, size=size, symbol=symbol, latticeconstant=a0, pbc=(1,1,1))
    if surface == 'diamond111':
        sx, sy, sz = bulk.get_cell().diagonal()       
        bulk.translate([sx/(6*size[0]), sy/(4*size[1]),  sz/(12*size[2])])
        bulk.set_scaled_positions(bulk.get_scaled_positions()%1.0)
    cell = bulk.get_cell() ; cell[0,:] *=2 # vacuum along x axis (surface normal)
    slab = bulk.copy() ; slab.set_cell(cell)
    
    # Optimize the geometries
    bulk.calc = calc ; opt_bulk = PreconLBFGS(bulk) ; opt_bulk.run(fmax=fmax)
    slab.calc = calc ; opt_slab = PreconLBFGS(slab, use_armijo=False) ; opt_slab.run(fmax=fmax)
    
    # Find surface energy
    import numpy as np
    Ebulk = bulk.get_potential_energy() ; Eslab = slab.get_potential_energy()
    area = np.linalg.norm(np.cross(slab.get_cell()[1,:],slab.get_cell()[2,:]))
    gamma_ase = (Eslab - Ebulk)/(2*area)

    # Convert to required units
    if unit == 'ASE':
        return [gamma_ase,'ase_units']
    else:
        gamma_SI = (gamma_ase / units.J ) * (units.m)**2
        if unit =='J/m^2':
            return gamma_SI
        elif unit == '0.1J/m^2':
            return 10*gamma_SI # units required for the fracture code
        else:
            print('Error: Unsupported unit of surface energy.')

at = bulk("Si", cubic=True)
at.calc = calc
ecf = ExpCellFilter(at)
opt = PreconLBFGS(ecf)
opt.run(1e-4)
a0 = at.cell[0, 0]
print('a0 =', a0, "A")

at = bulk("Si", cubic=True, a=a0)
at.calc = calc
_C, _C_err = fit_elastic_constants(at, symmetry='cubic', delta=1e-3)

C11, C12, C44 = np.array([_C[0, 0], _C[0, 1], _C[3, 3]]) / GPa
print('C11 =', C11, "GPa")
print('C12 =', C12, "GPa")
print('C44 =', C44, "GPa")

surface_energy  = find_surface_energy('Si', calc, a0=a0, surface='diamond111', fmax=1e-6)
print('Surface energy = ', surface_energy, '0.1*J/m^2')

# Crack system
n               = [ 12, 11, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]
#crack_tip       = [ 41, 56 ]
skin_x, skin_y = 1, 1

vacuum          = 6.0

# Simulation control
nsteps          = 32
# Increase stress intensity factor
k1              = np.linspace(0.6, 1.5, nsteps)
# Don't move crack tip
tip_dx          = np.zeros_like(k1)
tip_dz          = np.zeros_like(k1)

fmax            = 1e-3

center_crack_tip_on_bond = True

# Setup crack system
cryst = diamond(el, a0, n, crack_surface, crack_front)
set_groups(cryst, n, skin_x, skin_y)

ase.io.write('cryst.cfg', cryst)

# Compute crack tip position
#r0 = np.sum(cryst.get_positions()[crack_tip,:], axis=0)/len(crack_tip)
#tip_x0, tip_y0, tip_z0 = r0
