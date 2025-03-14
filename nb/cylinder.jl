### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 188bcb0a-fe50-11ef-3a83-db5f8c6145a5
using Pkg; Pkg.activate(ENV["JNEXT"])

# ╔═╡ 88c523e4-e523-4933-9a7c-09a151b335fa
begin
	using PlutoUI
	using Random
	using Printf
	using InteractiveUtils
	using SparseArrays, LinearAlgebra
	using Statistics
	using JLD2
end

# ╔═╡ 9231202b-301c-4099-9792-fa997e103146
begin
	import Plots 
	import PyPlot
	import PyCall
end

# ╔═╡ 061961e0-8fd1-4372-ba2e-6d0972f44f5d
np = PyCall.pyimport("numpy")

# ╔═╡ 8a68d758-7401-4fc2-b3ce-fc1203ec75c4
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

# ╔═╡ fad183d0-bb39-41d2-9bc4-b34a794a6d8e
jn = ingredients(string(ENV["JNEXT"],"/src/JNEXT.jl"))

# ╔═╡ 2d010d6c-3f62-4037-9751-c4d53977a81d
"""meshgrid implementation"""
function meshgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(reshape(x, 1, length(x)), length(y), 1)
    Y = repeat(reshape(y, length(y), 1), 1, length(x))
    return X, Y
end

# ╔═╡ e476ac13-3fe5-4a28-a2b5-5a409d8eb4cb
begin
	struct Cylinder
	    r::Float64      # radius
		x0::Float64     # cylinder center
		y0::Float64     # cylinder center
	    zmin::Float64   # lower z-boundary
	    zmax::Float64   # upper z-boundary
	    p0::Vector{Float64}  # point on one end (computed)
	    p1::Vector{Float64}  # point on the other end (computed)
	   
	end
	"""Outer constructor computes additional fields."""
	function Cylinder(r::Float64, x0::Float64, y0::Float64, 
		               zmin::Float64, zmax::Float64)
	    p0 = [x0, y0, zmin]   # point at one end
	    p1 = [x0, y0, zmax]   # point at the other end
	    cyl = Cylinder(r, x0, y0, zmin, zmax, p0, p1)
	    return cyl
	end
end

# ╔═╡ 6b367049-6a8d-49be-8a97-5e2d0b124bff
function unit_vectors(c::Cylinder)
	    v = c.p1 .- c.p0
	    mag = norm(v)
	    v = v / mag
	    not_v = [1.0, 0.0, 0.0]
	    # Check if v is approximately (1,0,0)
	    if all(isapprox.(v, not_v))
	        not_v = [0.0, 1.0, 0.0]
	    end
	    n1 = cross(v, not_v)
	    n1 /= norm(n1)
	    n2 = cross(v, n1)
	    return mag, v, n1, n2
	end

# ╔═╡ 9bcb8d53-5f19-4923-9930-6362f33badfc
begin
	clength(c::Cylinder) = c.zmax - c.zmin
	perimeter(c::Cylinder) = 2 * π * c.r
	area_barrel(c::Cylinder) = 2 * π * c.r * length(c)
	area_endcap(c::Cylinder) = π * c.r^2
	area(c::Cylinder) = area_barrel(c) + 2 * area_endcap(c)
	volume(c::Cylinder) = π * c.r^2 * length(c)
end

# ╔═╡ b0458771-0f96-4c39-a546-0370e0096da4
c2 = Cylinder(1.5, 3.0, 3.0, -5.0, 0.0)

# ╔═╡ 363759b1-5870-48d7-9b41-b85e01b3e34d
c2.p0

# ╔═╡ 8347b858-13b8-4091-a8f0-a384abebc503
unit_vectors(c2)

# ╔═╡ efdcf82b-eaef-48ae-9b66-5a67b1e7acf7
function surfaces(c::Cylinder)

	mag, v, n1, n2 = unit_vectors(c)
	
	# Create parameter ranges:
	t = collect(range(0, stop=mag, length=2))   # for the axis (2 sample points)
	theta = collect(range(0, stop=2*pi, length=100))    # angular parameter
	rsample = collect(range(0, stop=c.r, length=2))     # for endcaps

	# Create meshgrid arrays.
	T, Theta2 = meshgrid(t, theta)        # for the barrel
	R, Theta  = meshgrid(rsample, theta)   # for the endcaps


	# Barrel ("tube"): generate coordinates over the lateral surface.
	    X = c.p0[1] .+ v[1]*T .+ c.r .* sin.(Theta2) .* n1[1] .+ c.r .* cos.(Theta2) .* n2[1]
	    Y = c.p0[2] .+ v[2]*T .+ c.r .* sin.(Theta2) .* n1[2] .+ c.r .* cos.(Theta2) .* n2[2]
	    Z = c.p0[3] .+ v[3]*T .+ c.r .* sin.(Theta2) .* n1[3] .+ c.r .* cos.(Theta2) .* n2[3]
	    
	    # Bottom endcap (at zmin)
	    X2 = c.p0[1] .+ R .* sin.(Theta) .* n1[1] .+ R .* cos.(Theta) .* n2[1]
	    Y2 = c.p0[2] .+ R .* sin.(Theta) .* n1[2] .+ R .* cos.(Theta) .* n2[2]
	    Z2 = c.p0[3] .+ R .* sin.(Theta) .* n1[3] .+ R .* cos.(Theta) .* n2[3]
	    
	    # Top endcap (at zmax)
	    X3 = c.p0[1] .+ v[1]*mag .+ R .* sin.(Theta) .* n1[1] .+ R .* cos.(Theta) .* n2[1]
	    Y3 = c.p0[2] .+ v[2]*mag .+ R .* sin.(Theta) .* n1[2] .+ R .* cos.(Theta) .* n2[2]
	    Z3 = c.p0[3] .+ v[3]*mag .+ R .* sin.(Theta) .* n1[3] .+ R .* cos.(Theta) .* n2[3]
	    
	    return (X, Y, Z), (X2, Y2, Z2), (X3, Y3, Z3)
end

# ╔═╡ 05e078e1-dbff-424b-925b-d8e61aceb452
function draw_cylinder2(c; alpha=0.2, figsize=(16,16), 
                        DWORLD=false, 
                        WDIM=((0.0,4.5), (0.0,4.5), (-10.0,0)),
                        barrelColor="blue", cupColor="red")
	
	fig = PyPlot.figure(figsize=figsize)
    ax = PyPlot.subplot(111, projection="3d")

	P, P2, P3 = surfaces(c)

	if DWORLD
        ax.set_xlim3d(WDIM[1][1], WDIM[1][2])
        ax.set_ylim3d(WDIM[2][1], WDIM[2][2])
        ax.set_zlim3d(WDIM[3][1], WDIM[3][2])
    end

	ax.plot_surface(P[1], P[2], P[3], color=barrelColor, alpha=alpha)
	ax.plot_surface(P2[1], P2[2], P2[3], color=cupColor, alpha=alpha)
    ax.plot_surface(P3[1], P3[2], P3[3], color=cupColor, alpha=alpha)

	
	fig
end

# ╔═╡ 4249d51a-cc48-4537-91af-1173e86f1596
draw_cylinder2(c2; alpha=0.2, figsize=(16,16),
                        barrelColor="blue", cupColor="red")

# ╔═╡ Cell order:
# ╠═188bcb0a-fe50-11ef-3a83-db5f8c6145a5
# ╠═88c523e4-e523-4933-9a7c-09a151b335fa
# ╠═9231202b-301c-4099-9792-fa997e103146
# ╠═061961e0-8fd1-4372-ba2e-6d0972f44f5d
# ╠═8a68d758-7401-4fc2-b3ce-fc1203ec75c4
# ╠═fad183d0-bb39-41d2-9bc4-b34a794a6d8e
# ╠═2d010d6c-3f62-4037-9751-c4d53977a81d
# ╠═e476ac13-3fe5-4a28-a2b5-5a409d8eb4cb
# ╠═6b367049-6a8d-49be-8a97-5e2d0b124bff
# ╠═9bcb8d53-5f19-4923-9930-6362f33badfc
# ╠═b0458771-0f96-4c39-a546-0370e0096da4
# ╠═363759b1-5870-48d7-9b41-b85e01b3e34d
# ╠═8347b858-13b8-4091-a8f0-a384abebc503
# ╠═efdcf82b-eaef-48ae-9b66-5a67b1e7acf7
# ╠═05e078e1-dbff-424b-925b-d8e61aceb452
# ╠═4249d51a-cc48-4537-91af-1173e86f1596
