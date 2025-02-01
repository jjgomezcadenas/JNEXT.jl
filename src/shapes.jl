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
    An abstract class representing a geometrical shape.
"""
abstract type Shape end

"""
    Represents a box of dimensions (xmin,xmax), (ymin, ymax), (zmin, zmax)
"""
struct Box <: Shape
    xmin::Unitful.Length 
    xmax::Unitful.Length 
    ymin::Unitful.Length 
    ymax::Unitful.Length 
    zmin::Unitful.Length 
    zmax::Unitful.Length 
end


"""
    Represents a cylinder of radius R and length L
"""
struct Cylinder <: Shape
    R::Unitful.Length 
    L::Unitful.Length 
end



"""
    Represents a cylinder shell of internal radius Rin
    external radius Rout and length L
"""
struct CylinderShell <: Shape
    Rin::Unitful.Length 
    Rout::Unitful.Length 
    L::Unitful.Length 
end


volume(c::Cylinder) = π * c.R^2 * c.L

surface(c::Cylinder) = 2π * c.R * c.L

endcap_surface(c::Cylinder) = π * c.R^2

inner_volume(c::CylinderShell) = π * c.Rin^2 * c.L

shell_volume(c::CylinderShell) = π * (c.Rout^2 - c.Rin^2) * c.L
    
inner_surface(c::CylinderShell) = 2π * c.Rin * c.L  

inner_endcap_surface(c::CylinderShell) = π * c.Rin^2

outer_surface(c::CylinderShell) = 2π * c.Rout * c.L

outer_endcap_surface(c::CylinderShell) = π * c.Rout^2

thickness(c::CylinderShell) = c.Rout - c.Rin

    

