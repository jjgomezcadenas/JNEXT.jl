module JNEXT
using Revise
#include("materials.jl")
#include("shapes.jl")
include("laplace2D.jl")
include("laplace3D.jl")
#include("LaplaceTwoHolesXZ.jl")
end
