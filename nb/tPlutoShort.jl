using Pkg; Pkg.activate(ENV["JNEXT"])
begin
	using PlutoUI
	using Plots
	using Random
	using Printf
	using InteractiveUtils
	using SparseArrays, LinearAlgebra
	using JLD2
end

import PyPlot

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


begin
	jn = ingredients(string(ENV["JNEXT"],"/src/JNEXT.jl"))
end