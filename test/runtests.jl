using JNEXT
using Test

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

import JNEXT: 
A_BI214_316Ti, A_TL208_316Ti, A_BI214_CU_LIM, A_TL208_CU_LIM,
A_BI214_CU_BEST, A_TL208_CU_BEST, A_BI214_PB, A_TL208_PB,
A_BI214_Poly, A_TL208_Poly, PhysicalMaterial,
RadioactiveMaterial, GXe, Cylinder, CylinderShell,
volume, surface,endcap_surface,inner_volume,shell_volume,
inner_surface,inner_endcap_surface,outer_surface,
outer_endcap_surface,thickness, mass, a_bi214, a_tl208, att_length,
PhysicalCylindricalShell, PhysicalCylinder,
mass_structure,
a_bi214_structure,
a_tl208_structure,
att_length_structure,
mass_shield,
a_bi214_shield,
a_tl208_shield,
att_length_shield,
mass_gas

pvacuum = PhysicalMaterial(1e-25 * g/cm^3, 1e-25 * cm^2/g)
vacuum = RadioactiveMaterial(pvacuum, 0.0*Bq/kg, 0.0*Bq/kg)
                            
pti316  = PhysicalMaterial(7.87 * g/cm^3, 0.039 * cm^2/g)
ti316 = RadioactiveMaterial(pti316, A_BI214_316Ti, A_TL208_316Ti)

pcu  = PhysicalMaterial(8.96 * g/cm^3, 0.039 * cm^2/g)
cu12 = RadioactiveMaterial(pcu, A_BI214_CU_LIM, A_TL208_CU_LIM)
cu03 = RadioactiveMaterial(pcu, A_BI214_CU_BEST, A_TL208_CU_BEST)

ppb  = PhysicalMaterial(11.33 * g/cm^3, 0.044 * cm^2/g)
pb = RadioactiveMaterial(ppb, A_BI214_PB, A_TL208_PB)

ppoly  = PhysicalMaterial(0.97 * g/cm^3, 1e-6 * cm^2/g)
poly = RadioactiveMaterial(ppoly, A_BI214_Poly, A_TL208_Poly)

rmdct = Dict("vacuum" => vacuum,
             "ti316" => ti316,
             "cu12" => cu12,
             "cu03" => cu03,
             "pb" => pb,
             "poly" => poly)

Rin = 1.0mm
Rout = 2.0mm
Ll = 1.0mm

cyl = Cylinder(Rin, Ll)
cyls = CylinderShell(Rin, Rout, Ll)


@testset "JNEXT.jl" begin

    for (name, mat) in rmdct
        @test mat.m.μ ≈ mat.m.μovrρ * mat.m.ρ
    end

    @test volume(cyl) ≈ π*mm^3 
    @test surface(cyl) ≈ 2π*mm^2
    @test endcap_surface(cyl) ≈ π*mm^2 
    @test inner_volume(cyls) ≈ π*mm^3
    @test shell_volume(cyls) ≈ 3π * mm^3 
    @test inner_surface(cyls) ≈ 2π*mm^2  
    @test inner_endcap_surface(cyls) ≈ π*mm^2
    @test outer_surface(cyls) ≈ 4π * mm^2     
    @test outer_endcap_surface(cyls) ≈ 4π * mm^2
    @test thickness(cyls) ≈ 1.0mm

end
