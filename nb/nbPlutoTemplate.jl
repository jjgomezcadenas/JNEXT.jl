using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

begin
	using PlutoUI
	using CSV
	using DataFrames
	using Images
	using Plots
	using Random
	using Test
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
	using SparseArrays
	using OrdinaryDiffEq
	using LinearAlgebra
	using DifferentialEquations 
	using StaticArrays
end

import Unitful:
nm, μm, mm, cm, m, km,
mg, g, kg,
fs, ps, ns, μs, ms, s, minute, hr, d, yr, Hz, kHz, MHz, GHz,
eV, keV, MeV,
μJ, mJ, J,
μW, mW, W,
A, N, mol, mmol, V, L, M

import PhysicalConstants.CODATA2018: N_A

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

PlutoUI.TableOfContents(title="", indent=true)