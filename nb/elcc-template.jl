using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

begin
	using PlutoUI
	using CSV
	using DataFrames
	using Images
	using Plots
	using Printf
	using InteractiveUtils
	using Statistics
	using StatsBase
	using StatsPlots
	using Distributions
	using Unitful 
	using UnitfulEquivalences 
	using PhysicalConstants
	using PoissonRandom
	using Interpolations
	using HDF5
	using FHist
	using NearestNeighbors
	using DataStructures
	using CategoricalArrays
end

function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(m,
        Expr(:toplevel,
                :(eval(x) = $(Expr(:core, :eval))($name, x)),
                :(include(x) = $(Expr(:top, :include))($name, x)),
                :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
                :(include($path))))
    m
end

jn = ingredients("../src/JNEXT.jl")

load("elcc.jpeg")

