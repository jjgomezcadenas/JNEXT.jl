using Unitful
using UnitfulEquivalences
using PhysicalConstants.CODATA2018
using Interpolations

import Unitful:
    nm, μm, mm, cm, m, km,
    mg, g, kg,
    ps, ns, μs, ms, s, minute, hr, d, yr, Hz, kHz, MHz, GHz,
    eV,
    μJ, mJ, J,
	μW, mW, W,
	A, N, mol, mmol, V, L, M,
    Bq, mBq, μBq 

import PhysicalConstants.CODATA2018: N_A

"""
A Physical Cylinder shell includes:
1. A Cylinder shell shape
2. A radioactive material filling the shell
"""
struct PhysicalCylindricalShell
    shell::CylinderShell
    material::RadioactiveMaterial
end

inner_volume(c::PhysicalCylindricalShell) = inner_volume(c.shell)
shell_volume(c::PhysicalCylindricalShell) = shell_volume(c.shell)
inner_surface(c::PhysicalCylindricalShell) = inner_surface(c.shell)
outer_surface(c::PhysicalCylindricalShell) = outer_surface(c.shell)  
mass(c::PhysicalCylindricalShell) = shell_volume(c) * c.material.ρ
att_length(c::PhysicalCylindricalShell) = c.material.λ
a_bi214(c::PhysicalCylindricalShell) = c.material.a_bi214
a_tl208(c::PhysicalCylindricalShell) = c.material.a_tl208

"""
A Physical Cylinder includes:
1. A Cylinder shape
2. A radioactive material filling the cylinder
"""
struct PhysicalCylinder
    shell::Cylinder
    material::RadioactiveMaterial
end

volume(c::PhysicalCylinder) = volume(c.shell)
surface(c::PhysicalCylinder) = surface(c.shell)
endcap_surface(c::PhysicalCylinder) = endcap_surface(c.shell)
mass(c::PhysicalCylinder) = volume(c.shell) * c.material.ρ
att_length(c::PhysicalCylinder) = c.material.λ
a_bi214(c::PhysicalCylinder) = c.material.a_bi214
a_tl208(c::PhysicalCylinder) = c.material.a_tl208

"""
A NextVessel is made of:
1. A cylindrical shell shape defining the structure (e.g, steel)
2. A cylindrical shell shape defining the barrel shield (e.g, copper)
3. structure end-caps
4. shield end-caps
5. xenon
"""
struct NextVessel:
    bst::PhysicalCylindricalShell
    bsl::PhysicalCylindricalShell
    est::PhysicalCylinder
    esl::PhysicalCylinder
    gas::PhysicalCylinder
end

mass_structure(n::NextVessel) = mass(n.bst)
a_bi214_structure(n::NextVessel) = a_bi214(n.bst)
a_tl208_structure(n::NextVessel) = a_tl208(n.bst)
att_length_structure(n::NextVessel) = att_length(n.bst)
mass_shield(n::NextVessel) = mass(n.bsl)
a_bi214_shield(n::NextVessel) = a_bi214(n.bsl)
a_tl208_shield(n::NextVessel) = a_tl208(n.bsl)
att_length_shield(n::NextVessel) = att_length(n.bsl)
mass_gas(n::NextVessel) = mass(n.gas)

     