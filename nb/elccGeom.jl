### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7884ab66-f9e2-11ef-03ea-f10faf671dba
using Pkg; Pkg.activate(ENV["JNEXT"])


# ╔═╡ 108bd997-6d2d-416d-b2df-ec034273d62e
begin
	using PlutoUI
	using CSV
	using DataFrames
	using Plots
	using Printf
	using InteractiveUtils
	using Statistics
	using LinearAlgebra
	using JLD2
end

# ╔═╡ a333497c-c5fc-49a7-a8ca-d82a7dcd27ad
begin
	import PyPlot
end

# ╔═╡ 57248e63-9e36-4644-a6c4-5a3aa1808e29
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

# ╔═╡ eada2802-68a7-4663-a2c0-c1ca41b74601
jn = ingredients(string(ENV["JNEXT"],"/src/JNEXT.jl"))

# ╔═╡ 279bd076-8960-4f17-b719-f3d985ffdd5c
md"""
## Logger
"""

# ╔═╡ 1046cd03-a4b0-411f-8107-a2d3df10a386
begin
	const DEBUG = 10
	const INFO  = 20
	const WARN  = 30
	const ERROR = 40
	global_log_level = INFO
end

# ╔═╡ aa3b92ad-7fbf-4b3e-9eae-b1e9dcaf2eb4
begin
		"Internal function to print message if the given level is >= global level."
	function log_message(level::Int, level_str::String, msg)
	    if level >= global_log_level
	        # Print with a timestamp if desired.
	        println("[$(level_str)] ", msg)
	    end
	end

	"Log a debug message."
	debug(msg) = log_message(DEBUG, "DEBUG", msg)
	"Log an info message."
	info(msg)  = log_message(INFO,  "INFO", msg)
	"Log a warning message."
	warn(msg)  = log_message(WARN,  "WARN", msg)
	"Log an error message."
	error(msg) = log_message(ERROR, "ERROR", msg)

end

# ╔═╡ 2617ae05-e1db-473a-9b0f-befeea6d0e12
md"""
# Simulation
"""

# ╔═╡ a891cff0-6910-4f78-8fc5-ff4e90163a7e
begin
	kV = 1e+3
	mm = 1.0
	cm = 10.0
end

# ╔═╡ 3b1d7427-73ca-4dca-99f9-93b2cb6df9a8
struct ELCCGeometry
	X::Float64         # ELCC total x dimension (mm)
	Y::Float64         # ELCC total y dimension (mm)
	Zc::Float64        # Z posiiton of collector (mm)
	Zg::Float64        # Z posiiton of gate (mm)
	Za::Float64        # Z posiiton of anode (mm)
	Zs::Float64        # Z posiiton of SiPM plane (mm)
	Vg::Float64        # potential at gate 
	Va::Float64        # potential at anode 
	d_hole::Float64    # Hole diameter in each dice (mm)
	pitch::Float64     # pitch (mm)	    
end

# ╔═╡ af54e98e-15fb-4ad0-a990-66e183265867
struct SiPMGeometry
	sipmSize::Float64  # Side length of SiPM active area (mm); assumed square.
	pitch::Float64     # Pitch (center-to-center distance) between SiPMs (mm)
	X::Float64         # Overall SiPM panel x dimension (mm)
	Y::Float64         # Overall SiPM panel y dimension (mm)
end

# ╔═╡ 6097b338-4107-4d8e-9ee3-3f806f73c45b
begin
	ndicex(elcc::ELCCGeometry) = Int(floor(elcc.X/elcc.pitch))
	ndicey(elcc::ELCCGeometry) = Int(floor(elcc.Y/elcc.pitch))
	nsipmx(sipm::SiPMGeometry) = Int(floor(sipm.X/sipm.pitch))
	nsipmy(sipm::SiPMGeometry) = Int(floor(sipm.Y/sipm.pitch))
end

# ╔═╡ ebe9308e-87f5-410e-a17d-deb59b59a62d


# ╔═╡ ac79ab2e-af61-499a-94e7-964a8f04b111
begin

	# ELCSS geometry. The collector is located at 10 mm, the gate at 0 mm and the anode at -5 mm (the SiPMs at -10 mm)
	X = 120.0mm      # ELCC total x dimension (mm)
	Y = 120.0mm      # ELCC total y dimension (mm)
	Zc = 10.0mm     # Z posiiton of collector (mm)
	Zg = 0.0mm      # Z posiiton of gate (mm)
	Za = -5.0mm     # Z posiiton of anode (mm)
	Zs = -10.0mm    # Z posiiton of SiPM plane (mm)
	Vg = -10.0kV  # potential at gate 
	Va = 1kV      # potential at anode 
	d_hole = 3.0mm  # Hole diameter in each dice (mm)
	pitch = 6mm     # pitch
	trfile ="trj_d_3_p_6.jld2"

	R = d_hole/2
	dd = d_hole
	pp = (pitch - d_hole)/2
	x0 = y0 = pp + R
	
	
	md"""
	- Size of ELCC $(X) x $(Y) mm
	- hole center in ($(x0), $(y0)) mm
	- hole diameter $(dd) mm, pitch $(pitch) mm
	- Zc = $(Zc), Zg = $(Zg), Za = $(Za), Zs = $(Zs)
	-
	"""
end

# ╔═╡ 2c67d2f7-4d20-4ff6-ba81-c6902980478d
function yield_mm_p_10b(zg, za, vg, va) 
	vv = (vg-va)/kV
	zz = (zg-za)/cm
	eovp10b = -vv/zz
	ycm = 140 * eovp10b - 116 * 10
	ycm/cm
end

# ╔═╡ b2c37cf5-3a69-408c-9324-7ec1cdef6d18
ymm = yield_mm_p_10b(Zg, Za, Vg, Va)

# ╔═╡ ce43b3cb-69b9-43d3-beba-d83ac5f0f1a6
140 * 20 - 1160

# ╔═╡ 321fb432-4464-47b8-94ac-30d466670224
md"""
## ELCC geometry
"""

# ╔═╡ 154133b1-81bd-4bfe-86dc-cb3ccfec48f0
elcc = ELCCGeometry(X, Y, Zc, Zg, Za, Zs, Vg, Va, d_hole, pitch)

# ╔═╡ a0dfd610-50ca-4a75-9ab8-8c3937f31c33
sipm = SiPMGeometry(6.0, 10.0, 120.0, 120.0)

# ╔═╡ 16e4221a-4dd8-4571-8ce2-ef259400562a
md"""
- ELCC structure created with $(ndicex(elcc)) dices and $(nsipmx(sipm)) sipms.
"""

# ╔═╡ a340f566-c9c0-4293-988e-11b7e69e8e4a
md"""
## Trajectories
"""

# ╔═╡ c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
@load  trfile ftrj btrj

# ╔═╡ 7c38062b-0671-451c-911e-f88272f97937
begin
	zftrj = [ftrj[i][1,1:2] for i in range(1, length(ftrj))]
	xmax = maximum([z[1] for z in zftrj])
	xmin = minimum([z[1] for z in zftrj])
	ymax = maximum([z[2] for z in zftrj])
	ymin = minimum([z[2] for z in zftrj])
	md"""
	Trajectories:
	- in X, between $(xmin) and $(xmax)
	- in Y, between $(ymin) and $(ymax)
	
	"""
end

# ╔═╡ c5924aa7-a04b-4820-aafb-2c71a5bb289d
@bind SimulatePhotons CheckBox()

# ╔═╡ b4bec083-392c-45f3-b440-91edd3b5e5fc
#ntop, nbot, sij, gammas = simulate_photons_along_trajectory(electron_pos, tr, elcc, sipm; p1=1.00, p2=0.7, ymm=ymm, samplet=2, keepg=true)

# ╔═╡ 42392447-15b4-48ed-ad52-cf00aefc435f
sqrt((3-2.09)^2 +(3-2.9)^2)

# ╔═╡ 604a8022-d687-4684-8fd2-0178a3192452
sqrt(0.5^2 +0.6^2 + 0.62^2)

# ╔═╡ 1dad2fcb-836e-46a8-bb2b-8d43f25c4767


# ╔═╡ 02ffafb2-8ca6-4fcd-a267-c5e7528137cc
PyPlot.close("all")

# ╔═╡ e8f9953e-4983-4825-abfa-829653aa0d26
open_figs = PyPlot.get_fignums()

# ╔═╡ 23d039f4-b1db-4778-be0b-2fa01075a1a2
#plot_trajectory(tr, elcc, sipm)

# ╔═╡ 0e31a7b1-e95c-477a-9212-a5a1726370e5
# Example usage:
# Assume that your extended trajectory from simulate_electron_transport3D is stored in `traj_ext`
# and its last row is the impact point on the SiPM plane. Then:
# photon_impact = (traj_ext[end,1], traj_ext[end,3])
# p2 = plot_trajectory_xz(traj_ext, photon_impact)
# display(p2)

# ╔═╡ b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
md"""
# Functions
"""

# ╔═╡ f6e1dced-e962-4884-83aa-9f181a65982e
function meshgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(reshape(x, 1, length(x)), length(y), 1)
    Y = repeat(reshape(y, length(y), 1), 1, length(x))
    return X, Y
end

# ╔═╡ 5f2f631e-f305-49c2-88f4-dbf9be2c97a5
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

# ╔═╡ 657f8596-decb-41fc-bb83-f23839d8de32
c = Cylinder(R, x0, y0, Za, Zg)
	

# ╔═╡ ba7cb219-b68d-4414-9da0-e0c113db5c24
begin
	clength(c::Cylinder) = c.zmax - c.zmin
	perimeter(c::Cylinder) = 2 * π * c.r
	area_barrel(c::Cylinder) = 2 * π * c.r * length(c)
	area_endcap(c::Cylinder) = π * c.r^2
	area(c::Cylinder) = area_barrel(c) + 2 * area_endcap(c)
	volume(c::Cylinder) = π * c.r^2 * length(c)
end

# ╔═╡ d6f88078-ee18-4ff0-a2a1-67e6005d0b39
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

# ╔═╡ f400c977-03df-4ea3-b93a-c66bd386ab04
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

# ╔═╡ 57dec276-2cd4-4f12-9595-8cb42cbf08d9
function draw_cylinder(c; alpha=0.2, figsize=(16,16), 
                        DWORLD=false, 
                        WDIM=((0.0,4.5), (0.0,4.5), (-10.0,0)),
                        barrelColor="blue", cupColor="red")
	
	fig = PyPlot.figure(figsize=figsize)
    ax = PyPlot.subplot(111, projection="3d")

	P, P2, P3 = surfaces(c)

	if DWORLD
        ax.set_xlim3d(0.0, 10.0)
        ax.set_ylim3d(0.0, 10.0)
        ax.set_zlim3d(-10.0, 0.0)
    end

	ax.plot_surface(P[1], P[2], P[3], color=barrelColor, alpha=alpha)
	ax.plot_surface(P2[1], P2[2], P2[3], color=cupColor, alpha=alpha)
    ax.plot_surface(P3[1], P3[2], P3[3], color=cupColor, alpha=alpha)

	
	ax, fig
end

# ╔═╡ 578639df-4bec-42cf-97c4-0b510cd32b26
function draw_cylinder2(c, c2; alpha=0.3, figsize=(16,16), 
                        DWORLD=false, 
                        WDIM=((0.0,4.5), (0.0,4.5), (-10.0,0)),
                        barrelColor="blue", cupColor="red")
	
	fig = PyPlot.figure(figsize=figsize)
    ax = PyPlot.subplot(111, projection="3d")

	P, P2, P3 = surfaces(c2)

	if DWORLD
        ax.set_xlim3d(0.0, 10.0)
        ax.set_ylim3d(0.0, 10.0)
        ax.set_zlim3d(-10.0, 0.0)
    end

	ax.plot_surface(P[1], P[2], P[3], color=barrelColor, alpha=alpha)
	ax.plot_surface(P2[1], P2[2], P2[3], color=cupColor, alpha=alpha)
    ax.plot_surface(P3[1], P3[2], P3[3], color=cupColor, alpha=alpha)

	PP, _, _ = surfaces(c)
	ax.plot_surface(PP[1], PP[2], PP[3], color="orange", alpha=0.1)


	
	ax, fig
end

# ╔═╡ 8c7035f0-82fd-4c86-ab63-5cdd3b4d7539
function plot_cylinder(gammas, jsteps, x0, y0, zg, zb, za, r; 
                       num_plot=5, figsize=(16,16))

	c = Cylinder(r, x0, y0, za, zg)
	c2 = Cylinder(r, x0, y0, zb, zg)
	ax, fig = draw_cylinder2(c, c2; alpha=0.2, figsize=figsize, DWORLD=false)

	# Define a set of colors to cycle through for different gammas.
	colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow]

	# Extract x, y, z coordinates for each step in the jsteps.
	xs = [step[1] for step in jsteps]
	ys = [step[2] for step in jsteps]
	zs = [step[3] for step in jsteps]
	ax.plot(xs, ys, zs, linestyle="--", color=:cyan, linewidth=1)
	
	# Loop over the selected gammas.
	#for i in 1:min(num_plot, length(gammas))
	i = num_plot
	gamma = gammas[i]
	# Extract x, y, z coordinates for each step in the gamma.
	xs = [step[1] for step in gamma]
	ys = [step[2] for step in gamma]
	zs = [step[3] for step in gamma]
	
	# Choose a color for this gamma (cycling if needed).
	#col = colors[mod1(i, length(colors))]
	
	ax.scatter(xs, ys, zs, s=25, color=:red)

	#println("xs =",xs)
	#println("ys =",ys)
	#println("zs =",zs)

	# Connect the points with a dashed line.
	 ax.plot(xs, ys, zs, linestyle="--", color=:red, linewidth=1)
	#end

	fig
end

# ╔═╡ ef38565b-5a8d-41c2-a4f1-21cfbe0cb3aa
function draw_cylinder_proj(jsteps, gammas, r, x0, y0, Za, Zg, Zs, pitch; 
                            alpha=0.3, figsize=(8,8))


	c = Cylinder(r, x0, y0, Za, Zg)
	P, _, _ = surfaces(c)
	
	X = vec(P[1])
	Y = vec(P[1])
    Z = vec(P[3])

	# Extract x, y, z coordinates for each step in the jsteps.
	xs = [step[1] for step in jsteps]
	ys = [step[2] for step in jsteps]
	zs = [step[3] for step in jsteps]

	
	
	# Choose a color for this gamma (cycling if needed).
	#col = colors[mod1(i, length(colors))]
	
	
	
	fig, axs = PyPlot.subplots(1, 2, figsize=figsize)
	# Draw a vertical line at x = 5
	axs[1].axhline(y=Zg, color="black", linestyle="--", linewidth=2, label=false)
	axs[1].axhline(y=Za, color="black", linestyle="--", linewidth=2, label=false)
	axs[1].axhline(y=Zs, color="black", linestyle="solid", linewidth=2, label=false)

	axs[1].axvline(x=x0-r, color="black", linestyle="--", linewidth=2, label=false)
	axs[1].axvline(x=x0+r, color="black", linestyle="--", linewidth=2, label=false)
	
	# X-Z projection: plot X vs Z.
    #axs[1].scatter(X, Z, s=1, color="blue")
	axs[1].plot(xs, zs, linestyle=:solid, color=:red, linewidth=3)
	
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Z (mm)")
    axs[1].set_title("X–Z Projection")
	axs[1].set_xlim(0, pitch)
	axs[1].set_ylim(Zs, Zg)
	for gamma in gammas
		# Extract x, y, z coordinates for each step in the gamma.
		xgs = [step[1] for step in gamma]
		zgs = [step[3] for step in gamma]

		axs[1].scatter(xgs, zgs, s=1, color="cyan")
		axs[1].plot(xgs, zgs, linestyle="--", color=:cyan, linewidth=1)
	end
    
    # Y-Z projection: plot Y vs Z.
    axs[2].scatter(Y, Z, s=1, color="blue")
	axs[2].plot(ys, zs, linestyle=:solid, color=:cyan, linewidth=1)
    axs[2].set_xlabel("Y (mm)")
    axs[2].set_ylabel("Z (mm)")
    axs[2].set_title("Y–Z Projection")
    axs[2].set_xlim(0, pitch)
	axs[2].set_ylim(Zs, Zg)
    PyPlot.tight_layout()
	fig
end

# ╔═╡ 84704333-6927-4b78-be39-37e32c90ef15
P, _, _ = surfaces(c)

# ╔═╡ dbc021f7-9b18-4aa2-83d3-ad44bb887f58
length(vec(P[1]))

# ╔═╡ baadcc6b-36cb-49ae-aa06-686b267b4b4c
length(vec(P[3]))

# ╔═╡ 94a48200-aca1-4167-9e45-581e53cfdad5
"""Normal to the cylinder barrel"""
function normal_to_barrel(c::Cylinder, P::Vector{Float64})    
    [P[1], P[2], 0] ./ c.r
end

# ╔═╡ c2f6d14e-cfb7-4f53-88cc-9da2676f1ecb
"""
Uses equation of cylynder: 

F(x,y,z) = x^2 + y^2 - r^2 = 0
"""
function cylinder_equation(c::Cylinder, P::Vector{Float64})
    P[1]^2 + P[2]^2 - c.r^2
end

# ╔═╡ 3c238213-30c7-41a7-bd7b-50f8c09b7adf
"""
Get coordinates and yield
"""
function get_coord_and_yield(tr::AbstractMatrix, zg::Float64, ymm::Float64)
		iz0 = 0
		z0 = zg
		YL = Vector{Int64}()
		XC = Vector{Float64}()
		YC = Vector{Float64}()
		ZC = Vector{Float64}()
		for i in range(1, size(tr)[1])
			z = tr[i, 3]
			if z >zg
				continue
			elseif iz0 == 0
				iz0 = i
				z0 = tr[i, 3]
				#println(z0)
			end
			if z < z0
				dz = z0 - z
				yl = Int(floor(dz * ymm))
				push!(YL, yl)
				push!(XC, tr[i, 1])
				push!(YC, tr[i, 2])
				push!(ZC, tr[i, 3])
				z0 = z
			end
		end
		YL, XC, YC, ZC
	end

# ╔═╡ 05ac2255-16ed-4bdd-a4c4-c9e611cda5d0
"""
	Solve the intersection with the cylinder wall 
	
"""
function solve_t_barrel(x, y, x0, y0, vx, vy, R; eps=1e-16)
	a = vx^2 + vy^2
	b = 2 * ((x-x0) * vx + (y-y0) * vy)
	c = (x - x0)^2 + (y - y0)^2 - R^2

	debug("####solve_t_barrel: x-x0 = $(x-x0), y-y0=$(y-y0), vx = $(vx), vy=$(vy),R = $(R)")
	debug("####a = $(a), b = $(b), c = $(c)")
	if abs(a) < eps
		return nothing
	end

	disc = b^2 - 4 * a * c
	debug("####disc = $(disc)")
	if disc < 0
		return nothing
	end

	sqrt_disc = sqrt(disc)
	t1 = (-b + sqrt_disc) / (2 * a)
	t2 = (-b - sqrt_disc) / (2 * a)

	debug("####t1 = $(t1), t2 = $(t2)")

	ts_candidates = Float64[]
	#if t1 > 0
	#	push!(ts_candidates, t1)
	#elseif abs(t1) < eps
	#	push!(ts_candidates, 0.0)
	#end
	#if t2 > 0
	#	push!(ts_candidates, t2)
	#elseif abs(t2) < eps
	#	push!(ts_candidates, 0.0)
	#end

	if t1 > 0 && t2 > 0
		push!(ts_candidates, t1)
		push!(ts_candidates, t2)
	elseif t1 > 0 && t2 <= 0
		push!(ts_candidates, t1)
	elseif t2 >0 && t1 <= 0
		push!(ts_candidates, t2)
	elseif t1 >0 && abs(t2) <= eps
		push!(ts_candidates, t1)
	elseif t2 >0 && abs(t1) <= eps
		push!(ts_candidates, t2)
	elseif abs(t1) < eps || abs(t2) < eps
		push!(ts_candidates, 0.0)
	end

	if !isempty(ts_candidates)
		return minimum(ts_candidates)
	else
		return nothing
	end
end


# ╔═╡ e361a98f-f008-48f5-a4cb-94a100702460
solve_t_barrel(3.1989419320430073, 4.486748838128013, 3.0, 3.0, -0.7038555940531903, -0.2260637209182603, 1.5; eps=1e-16)

# ╔═╡ 9e8d1efe-0f54-4afc-a956-3664cf972d8a
"""
 Solve for time to reach the top (z = ztop).
"""
function solve_t_top(z, vz, ztop; eps=1e-10)
	debug("####solve_t_top: z = $(z), vz=$(vz), ztop=$(ztop)")
    if vz > eps
        dt = (ztop - z) / vz
		debug("####solve_t_top: dt = $(dt)")
        if dt > eps
            return dt
        end
    end
    return nothing
end



# ╔═╡ 9ebb47ac-b76c-4be4-8336-6da26f26c6c5

"""
 Solve for time to reach the bottom (z = zb).
"""
function solve_t_bottom(z, vz, zb; eps=1e-10)
	debug("####solve_t_bottom: z = $(z), vz=$(vz), zb=$(zb)")
    if vz < -eps
        dtb = (abs(zb) -abs(z)) / abs(vz)
		debug("####solve_t_bottom: dt = $(dtb)")
        if dtb > eps
            return dtb
        end
    end
    return nothing
end

# ╔═╡ 5b913924-1f00-4fc7-9c6a-0ede1d400f5c



# ╔═╡ b501a1f1-2c96-4680-b08a-982d2877603d
"""
Generate random direction
"""
function generate_direction()
    cost = 2.0 * rand() - 1.0  # cosine(theta) uniformly in [-1, 1]
    theta = acos(cost)         # theta in [0, π]
    phi = 2.0 * pi * rand()      # phi in [0, 2π]
    sinth = sqrt(1 - cost^2)
    vx = sinth * cos(phi)
    vy = sinth * sin(phi)
    vz = cost
    return (vx, vy, vz)
end


# ╔═╡ c7704f94-2ab5-4111-ac7c-63f978c7ee4c
function float_to_str(number, fmt::String)
    io = IOBuffer()
    Printf.format(io, Printf.Format(fmt), number)
    return String(take!(io))
end

# ╔═╡ fcbf9e5a-b7f2-400d-87af-7448fd348071
function vect_to_str(vec::AbstractVector, fmt::String)
    # Helper function to format a single element
    function format_element(x, fmt)
        if x isa AbstractFloat
            float_to_str(x, fmt)
        else
            return string(x)
        end
    end
	formatted_elements = [format_element(x, fmt) for x in vec]
	return "[" * join(formatted_elements, ", ") * "]"
end

# ╔═╡ c1fec8e5-bf38-4d39-8224-bf0051fc08eb
function generate_electron_positions_random(N::Int, 
                                            x_min::Float64, x_max::Float64, 
                                            y_min::Float64, y_max::Float64)
    # Generate N random x positions uniformly distributed between x_min and x_max
    xs = x_min .+ (x_max - x_min) .* rand(N)
    
    # Generate N random y positions uniformly distributed between y_min and y_max
    ys = y_min .+ (y_max - y_min) .* rand(N)
    
    # Combine the x and y positions into an N×2 matrix
    electron_xy0 = hcat(xs, ys)
    
    return electron_xy0
end

# Example usage:
# positions = generate_electron_positions_random(100, 0.0, 10.0, 0.0, 5.0)
# This returns a 100×2 matrix where each row is a random (x, y) coordinate.


# ╔═╡ 2273c136-4709-43a6-bf68-1184493fbb70
begin
	electron_xy0 = generate_electron_positions_random(20, 0.0, elcc.X, 0.0, elcc.Y)
	i = 1 # electron number to run 
	electron_pos = electron_xy0[i, :]  
	md"""
	Generated electron at position $(vect_to_str(electron_pos, "%.2f"))
	"""
end

# ╔═╡ af2b2805-9e6c-4078-9427-02f787212f19
function generate_electron_positions(N::Int, 
	                                 x_min::Float64, x_max::Float64,
	                                 y_min::Float64, y_max::Float64)
    # Choose numbers of electrons in x and y directions so that n_x*n_y >= N.
    n_x = floor(Int, sqrt(N))
    n_y = ceil(Int, N / n_x)
    
    # Create uniform grids in x and y
    xs = collect(range(x_min, x_max, length=n_x))
    ys = collect(range(y_min, y_max, length=n_y))
    
    # Create the full grid of (x,y) pairs.
    pos = [(x,y) for y in ys, x in xs]  # Note: yields an n_y×n_x array of tuples.
    pos = vec(pos)  # Flatten to a 1D vector
    
    # Take the first N positions
    pos = pos[1:N]
    
    # Convert to a matrix with N rows and 2 columns.
    electron_xy0 = reduce(vcat, [reshape([p[1], p[2]], 1, 2) for p in pos])
    
    return electron_xy0
end

# ╔═╡ d1aec6ee-f530-4a15-9bf9-3081c7f55f4a
"""
Given an electron absolute (x,y) position on the ELCC surface,
find the dice to which it is assigned and compute local (x_e,y_e).
"""
function find_dice(xpos::Vector{Float64}, elcc::ELCCGeometry)
    # Compute dice indices (starting at 1)
	
    i = clamp(floor(Int, (xpos[1]) / elcc.pitch) + 1, 1, ndicex(elcc))
    j = clamp(floor(Int, (xpos[2]) / elcc.pitch) + 1, 1, ndicey(elcc))
    # Dice lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*elcc.pitch, (j-1)*elcc.pitch )
    # Local coordinates inside dice
    xe = xpos[1] - dice_origin[1]
    ye = xpos[2] - dice_origin[2]
    return (i, j), dice_origin, (xe, ye)
end

# ╔═╡ 232fab2c-fd22-449c-be78-f4e55c7021e8
"""
Given an electron absolute (x,y) position on the ELCC surface,
find the sipm to which it is assigned 
"""
function find_sipm(xpos::Vector{Float64}, sipm::SiPMGeometry)
    # Compute dice indices (starting at 1)
    i = clamp(floor(Int, (xpos[1]) / sipm.pitch) + 1, 1, nsipmx(sipm))
    j = clamp(floor(Int, (xpos[2]) / sipm.pitch) + 1, 1, nsipmy(sipm))
    # Dice lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*sipm.pitch, (j-1)*sipm.pitch )
   
	return (i, j), dice_origin
end

# ╔═╡ ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
begin 
	dice_indices, dice_origin, xlocal = find_dice(electron_pos, elcc)
	sipm_indices, sipm_origin =find_sipm(electron_pos, sipm)
	xl = collect(xlocal)
	izfr = argmin([norm(xl -zftrj[i]) for i in 1:length(zftrj)])
	md"""
	- dice indices = $(dice_indices), sipm indices = $(sipm_indices)
	- local coordinates = $(vect_to_str(xl, "%.2f"))
	- Closer trajectory number $(izfr), with coordinates $(vect_to_str(ftrj[izfr][1,:], "%.2f"))
	"""
end

# ╔═╡ 951ea570-d232-47a3-bbe8-b216de1469a8
begin 
	trjf = ftrj[izfr]
	trjb = btrj[izfr]
	tr = vcat(trjf, trjb)
	md"""
	- Lenght of trajectory $(size(tr))
	"""
end

# ╔═╡ d474342a-81ca-4504-86a9-52925211b685
tr

# ╔═╡ 1ef1b221-da82-4852-bfb3-ffe1b2b50600
typeof(tr)

# ╔═╡ 5bee9446-a537-4012-95d2-77c91f27c83a
"""
Given an electron absolute (x,y) position on the ELCC surface,
find the sipm to which it is assigned 
"""
function find_abspos(xr::Tuple{Float64, Float64}, sipmIJ::Tuple{Int64, Int64}, sipm::SiPMGeometry)
    # Compute dice indices (starting at 1)
    i = sipmIJ[1]
    j = sipmIJ[2]
    # SiPM lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*sipm.pitch, (j-1)*sipm.pitch )
   	xabs = xr[1] + dice_origin[1]
    yabs = xr[2] + dice_origin[2]
	return  xabs, yabs
end

# ╔═╡ 7ec27076-edd8-4d3f-b691-8cf877144f98
"""
Simulate photons along the trajectory.

At each step along the trajectory, generate a number of photons.
Propagate them along.
Count photons that hit a SiPM (if the impact falls within a sensor active area).

- electron_post: the absolute position of the electron
- trj: A matrix holding a large number of trajectories, starting in the collector plane (where the electron_pos is defined) and ending in the anode.
- The ELCC structure
- The SiPM structure
- ymm: yield/mm in the EL region
- p1: probability of the photon to be absorbed in the first interaction
- p2: probability to be absorbed in further interactions.
- samplet: sampling of trayectory (to make it shorter than the frozen ones)
- maxgam: allows to generate less gammas than stipulated by the yield.
- saveg: allows to save gammas for plotting.
- savet: allows to save trajectory for plotting.
- ncmax: max number of bounces allowd for a given photon
- eps: tolerance factor
"""
function simulate_photons_along_trajectory(electron_pos::Vector{Float64},                                                       trj::AbstractMatrix, 
	                                       elcc::ELCCGeometry,
	                                       sipm::SiPMGeometry; 
										   ymm=10, # yield per mm
										   p1=1.00, # Prob abs first interaction
										   p2=0.5,  # Prob abs >1 interaction
										   samplet=1e+6, # sample trajectory
										   maxgam=1e6, # max number of gammas
										   saveg = false, # save gammas for plotting
										   savet = false, # save trj for plotting
                                           ncmax=20, # max number of bounces
										   eps=1e-16) # tolerance
	
	sipm_indices, sipm_origin =find_sipm(electron_pos, sipm)
	dice_indices, dice_origin, xlocal = find_dice(electron_pos, elcc)
	xl = collect(xlocal)
	
	dd = elcc.d_hole
	R = dd/2
	pp = (elcc.pitch - dd)/2
	x0 = y0 = pp + R
	ztop = elcc.Zg
	zbot = elcc.Za
	
	println("--simulate_photons_along_trajectory--")
	println("r hole = $(R) position of hole x0=$(x0) y0=$(y0)")
	println("->absolute electron position $(electron_pos)")
	println("->electron position relative to dice $(xl)")
	println("->dice indices $(dice_indices), origin =$(dice_origin)")
	println("->sipm indices $(sipm_indices), origin =$(sipm_origin)")

	c2 = Cylinder(R, x0, y0, zbot, ztop)
	
	YL, XC, YC, ZC = get_coord_and_yield(trj, ztop, ymm)

	gammas = Vector{Vector{Vector{Float64}}}()
	jsteps =Vector{Vector{Float64}}()
	NTOP = Vector{Int}()
	NBOT = Vector{Int}()
	SIJ = Vector{Vector{Tuple{Int, Int}}}()
	
	# Trajectory steps
	ltrj = min(length(YL), samplet)
	for ri in range(1, stop=length(YL), length=ltrj)
	
		i =Int(floor(ri))
		xe = XC[i]
		ye = YC[i]
		ze = ZC[i]
		yy = YL[i]

		if savet
			push!(jsteps, [xe, ye, ze])
		end

		yield = min(yy, maxgam)
		count_top = 0
		count_bot = 0
		println("--------000------------\n")
		println("-->for step =$(i), xe = $(xe), y=$(ye), z=$(ze), yield=$(yield)")
		println("--------000-----------\n")

		GIJ = Vector{Tuple{Int, Int}}()
		
		for ng in range(1, yield) # number of photons per step
			vx, vy, vz= generate_direction() # generate random direction
			
			println("------xxxx-----------\n")
			println("--->for ng =$(ng), vx =$(vx), vy =$(vy), vz = $(vz)")
			println("------xxxx-----------\n")

			n_collisions = 0
			
        	alive = true
			steps = Vector{Vector{Float64}}()
			push!(steps, [xe, ye, ze])
			x = xe
			y = ye
			z = ze

	        while alive
				if n_collisions > ncmax
					println("exceeds number of collisions =", n_collisions)
					break
				end
	            t_barrel = solve_t_barrel(x, y, x0, y0, vx, vy, R; eps=1e-16)
	            t_top    = solve_t_top(z, vz, ztop; eps=eps)
	            t_bottom = solve_t_bottom(z, vz, zbot; eps=eps)
				
				println("--->t_barrel =$(t_barrel), t_top =$(t_top), t_bottom =$(t_bottom)")
	
	            # Gather valid intersection times and corresponding surface labels.
	            possible_times = Float64[]
	            labels = String[]
	
	            if t_barrel !== nothing
	                push!(possible_times, t_barrel)
	                push!(labels, "barrel")
	            end
	            if t_top !== nothing
	                push!(possible_times, t_top)
	                push!(labels, "top")
	            end
	            if t_bottom !== nothing
	                push!(possible_times, t_bottom)
	                push!(labels, "bottom")
	            end
	
	            # If no intersection is found, the photon is lost.
	            if isempty(possible_times)
	                alive = false
	                break
	            end
	
	            # Choose the intersection that happens first.
	            idx_min = argmin(possible_times)
	            t_min = possible_times[idx_min]
	            surf = labels[idx_min]

				println("--->t_min =$(t_min), surf =$(surf)")
	
	            # Move the photon.
	            x_new = x + t_min * vx
	            y_new = y + t_min * vy
	            z_new = z + t_min * vz

				println("--->x_new =$(x_new), y_new =$(y_new), z_new =$(z_new)")
				push!(steps, [x_new, y_new, z_new])
	
	            if surf == "bottom" # anode 
	                alive = false
	                count_bot += 1
					if saveg
						push!(gammas, steps)
					end
					
					println("--->photon hits bottom at x =$(x_new), y = $(y_new), count =$(count_bot)")
					
					# keep position of sipm
					xabs, yabs = find_abspos((x_new, y_new), sipm_indices, sipm)
					println("--->xabs =$(xabs), yabs = $(yabs)")
					
					sipmij, _ = find_sipm(collect((xabs, yabs)), sipm)
					println("--->sipm(i,j) =($(sipmij[1]), $(sipmij[2]))")

					push!(GIJ, (sipmij[1], sipmij[2]))
						
	            elseif surf == "top"
	                alive = false
					count_top += 1
					if saveg
						push!(gammas, steps)
					end
					
					println("--->photon hits top at x =$(x_new), y = $(y_new), count =$(count_top)")

	            else  # "barrel"
	                n_collisions += 1
					
					println("--->photon hits barrel at x =$(x_new), y = $(y_new), n_collisions =$(n_collisions)")
					
	                # Determine re-emission probability.
	                p = (n_collisions == 1) ? p1 : p2
	                
	                # Photon is re-emitted 
	                x, y, z = x_new, y_new, z_new
					vx0, vy0, vz0 = vx, vy, vz
	                    
					cteta = 1.0
					while cteta >= 0 #photon goes against the wall
						p = (n_collisions == 1) ? p1 : p2
						if rand() < p # Photon is re-emitted 
							vx, vy, vz = generate_direction()
							cteta = dot([vx,vy,vz],[vx0,vy0,vx0])
							println("---->photon reemited in barrel")
							println("vx = $(vx), vy=$(vy), vz=$(vz)")
							println("cos(theta) =", cteta)

							if cteta >=0
								println("---->hit wall, n_collisions=", n_collisions)
							else
								println("---->photon goes away from wall")
							end
							
						else   # Photon is absorbed
							println("---->photon absorbed in barrel")
							alive = false	
							if saveg
								push!(gammas, steps)
							end
							
							break
						end
						 n_collisions += 1
					end   
					println("x = $(x), y=$(y), z=$(z)")
					println("vx = $(vx), vy=$(vy), vz=$(vz)")
					
            	end
			end
		end
		push!(NTOP, count_top)
		push!(NBOT, count_bot)
		push!(SIJ, GIJ)
	end
	NTOP, NBOT, SIJ, gammas, jsteps
end

# ╔═╡ 27669660-d21b-4e10-904d-b8142e8447dd
if SimulatePhotons
	ntop, nbot, sij, gammas, jsteps = simulate_photons_along_trajectory(electron_pos,
		                                                        tr, 
	                                                            elcc,
	                                                            sipm; 
										                        ymm=ymm,
										                        p1=1.00, 
										                        p2=0.5,  
										                        samplet=500, 
										                        maxgam=2,
	                                                            saveg=true,
	                                                            savet = true)
	
end

# ╔═╡ 8009f2f9-8931-45f4-9f66-41ea283563a5
jsteps[end]

# ╔═╡ 23167ba6-4ddc-49b8-9aec-b9148d09befc
gammas

# ╔═╡ ae488bf8-706a-4d57-8ac7-412f0a43bd08
gamma=gammas[1]

# ╔═╡ 8f537416-301f-427f-a584-b91cbd83450d
begin
 	fig = plot_cylinder(gammas, jsteps, x0, y0, Zg, Za, Zs, R; figsize=(6,6), num_plot=50)
	
end

# ╔═╡ a5d8d56f-641c-4a45-88e3-737f992802e2
draw_cylinder_proj(jsteps, gammas, R, x0, y0, Za, Zg, Zs, pitch; alpha=0.3, figsize=(8,8))

# ╔═╡ e97ec2de-accb-4336-bf0e-2247bbaeb3e2
gammas

# ╔═╡ 82882beb-98b0-4a53-9f0d-9d16bcbc6c09
"""
Graphical representation of the full structure.
"""
function pstructure(elcc::ELCCGeometry, sipm::SiPMGeometry)
    # ELCC: Represented as a rectangle in the xy-plane.
    # Coordinates of ELCC (we assume its top face is at z = elcc.Z)
    elcc_rect = Shape([0, elcc.X, elcc.X, 0], [0, 0, elcc.Y, elcc.Y])
    
    # Plot ELCC outline.
    p1 = plot(elcc_rect, fillcolor=:lightblue, alpha=0.3, label="ELCC", 
		      aspect_ratio=1,
              xlabel="x (mm)", ylabel="y (mm)", title="ELCC and SiPM Layout")
    
    # Draw dice boundaries and holes.
    for i in 0:ndicex(elcc)
        plot!([i*elcc.pitch, i*elcc.pitch], [0, elcc.Y], 
			   lc=:gray, lw=0.5, label=false)
    end
    for j in 0:ndicey(elcc)
        plot!([0, elcc.X], [j*elcc.pitch, j*elcc.pitch], 
			  lc=:gray, lw=0.5, label=false)
    end
	
    # Draw holes: each hole is a circle at the center of its dice.
    for i in 0:(ndicex(elcc)-1)
        for j in 0:(ndicey(elcc)-1)
            cx = i*elcc.pitch + elcc.pitch/2
            cy = j*elcc.pitch + elcc.pitch/2
            circle = Shape(cx .+ (elcc.d_hole/2)*cos.(LinRange(0,2π,50)),
                           cy .+ (elcc.d_hole/2)*sin.(LinRange(0,2π,50)))
            plot!(circle, fillcolor=:white, linecolor=:black, lw=0.5, label=false)
        end
    end

    # Plot the SiPM plane: assume it is located below the ELCC, at a distance elcc.Zsipm from the bottom.
    # For visualization, we show the SiPM panel as a rectangle.
    sipm_rect = Shape([0, sipm.X, sipm.X, 0], [0, 0, sipm.Y, sipm.Y])
    p2 = plot(sipm_rect, fillcolor=:lightgreen, alpha=0.3, label="SiPM Panel", 
		      aspect_ratio=1,
              xlabel="x (mm)", ylabel="y (mm)", title="SiPM Panel Layout")
    # Draw SiPM boundaries (sensors are squares of side sipm.sipmSize centered in cells of size sipm.pitch).
    n_sipm_x = round(Int, sipm.X / sipm.pitch)
    n_sipm_y = round(Int, sipm.Y / sipm.pitch)
    for i in 0:n_sipm_x-1
        for j in 0:n_sipm_y-1
            sx = i*sipm.pitch + (sipm.pitch - sipm.sipmSize)/2
            sy = j*sipm.pitch + (sipm.pitch - sipm.sipmSize)/2
            sensor = Shape([sx, sx+sipm.sipmSize, sx+sipm.sipmSize, sx],
                           [sy, sy, sy+sipm.sipmSize, sy+sipm.sipmSize])
            plot!(sensor, fillcolor=:white, linecolor=:black, lw=1.0, label=false)
        end
    end

    # Layout plots side by side.
   p1, p2
end

# ╔═╡ 79a51e0b-60ec-4ce7-b15b-1f4d88c6aa28
"""
Graphical representation of the full structure.
"""
function plot_structure(elcc::ELCCGeometry, sipm::SiPMGeometry)
	p1, p2 = pstructure(elcc, sipm)
    plot(p1, p2, layout = (1,2))
end

# ╔═╡ dfd7cbf8-adaa-454f-957e-ecc6eee905d3
plot_structure(elcc, sipm)

# ╔═╡ 2b780346-122b-463f-ac4b-498e45dfa84f
function plot_impact_xy(elcc::ELCCGeometry, 
	                sipm::SiPMGeometry, 
	                electron_impact_xy::Vector{Float64})
	p1, p2 = pstructure(elcc, sipm)
	# Plot the electron impact point in red.
    p1 = scatter!(p1, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
		          mc=:red, label=false)
	p2 = scatter!(p2, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
		          mc=:red, label=false)
	plot(p1, p2, layout = (1,2))
end

# ╔═╡ 9eb86c8c-4347-46c4-9111-793f921fac56
plot_impact_xy(elcc, sipm, electron_pos)

# ╔═╡ 60269ab6-d610-409e-90de-48022143ef1e
function pt_lgeom(epos::Vector{Float64}, elcc::ELCCGeometry, sipm::SiPMGeometry)
	
	dice_i, dice_o, xlocal = find_dice(electron_pos, elcc)
	sipm_i, sipm_o =find_sipm(electron_pos, sipm)

	xsipml = sipm_o[1]
	ysipmd = sipm_o[2]
	xdicel = dice_o[1]
	ydiced = dice_o[2]
	xl = minimum([xsipml, xdicel])
	yd = minimum([ysipmd, ydiced])
	zmax = elcc.Zc
	zmin = elcc.Zs 
	
	println("xsipml= $(xsipml), ysipmd = $(ysipmd), xdicel =$(xdicel), ydiced =$(ydiced)")
	println("xl= $(xl), yd = $(yd)")

	println("epos= $(epos)")

	six0l = xsipml
	six0r = six0l + sipm.sipmSize
	six1l = six0l - sipm.pitch
	six1r = six1l + sipm.sipmSize
	six2r = six0r + sipm.pitch
	six2l = six2r - sipm.sipmSize

	siy0l = ysipmd
	siy0r = siy0l + sipm.sipmSize
	siy1l = siy0l - sipm.pitch
	siy1r = siy1l + sipm.sipmSize
	siy2r = siy0r + sipm.pitch
	siy2l = siy2r - sipm.sipmSize

	dcx0l = xdicel
	dcx0r = dcx0l + elcc.d_hole
	dcx1l = dcx0l - elcc.pitch
	dcx1r = dcx1l + elcc.d_hole
	dcx2r = dcx0r + elcc.pitch
	dcx2l = dcx2r - elcc.d_hole

	dcy0l = ydiced
	dcy0r = dcy0l + elcc.d_hole
	dcy1l = dcy0l - elcc.pitch
	dcy1r = dcy1l + elcc.d_hole
	dcy2r = dcy0r + elcc.pitch
	dcy2l = dcy2r - elcc.d_hole

	xmin = minimum([six1l, dcx1l])
	xmax = maximum([six2r, dcx2r])
	ymin = minimum([siy1l, dcy1l])
	ymax = maximum([siy2r, dcy2r])
	

	println("six0l= $(six0l), six0r = $(six0r), six1l =$(six1l), six1r =$(six1r), six2l =$(six2l), six2r =$(six2r)")

	println("dcy0l= $(dcy0l), dcy0r = $(dcy0r), dcy1l =$(dcy1l), dcy1r =$(dcy1r), dcy2l =$(dcy2l), dcy2r =$(dcy2r)")

	println("xmin= $(xmin), xmax = $(xmax), ymin =$(ymin), ymax =$(ymax)")
	
	p1 = plot(xlims=(xmin, xmax), ylims = (zmin, zmax),
		      xlabel="x (mm)", ylabel="z (mm)", title="XZ")
	p1 = scatter!(p1, [epos[1]], [zmax - 0.5], ms=2,
		          mc=:red, label=false)

	p1 = plot!(p1, [six0l, six0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p1 = plot!(p1, [six1l, six1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p1 = plot!(p1, [six2l, six2r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)

	p1 = plot!(p1, [dcx0l, dcx0l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx0r, dcx0r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx1l, dcx1l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx1r, dcx1r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx2l, dcx2l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx2r, dcx2r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	
	p2 = plot(xlims=(ymin, ymax), ylims = (zmin, zmax),
		      xlabel="y (mm)", ylabel="z (mm)", title="YZ")
	p2 = scatter!(p2, [epos[2]], [zmax - 0.5], ms=2,
		          mc=:red, label=false)

	p2 = plot!(p2, [siy0l, siy0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p2 = plot!(p2, [siy1l, siy1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p2 = plot!(p2, [siy2l, siy2r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)

	p2 = plot!(p2, [dcy0l, dcy0l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy0r, dcy0r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy1l, dcy1l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy1r, dcy1r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy2l, dcy2l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy2r, dcy2r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	
	return p1, p2
end
	                     
	

# ╔═╡ 90d67f13-a612-42ec-9a21-fd40d822c17c
function p_trajectory(epos::Vector{Float64}, elcc::ELCCGeometry, 
	                  sipm::SiPMGeometry, tr::AbstractMatrix)
	
	function plott!(i, p, traj)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		plot!(p, traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false)

	
	end
	dice_indices, dice_origin, _ = find_dice(epos, elcc)
	posd = [dice_origin[1], dice_origin[2], elcc.Zc]
	xtrj = [tr[i,:] + [dice_origin[1], dice_origin[2], 0.0] for i in range(1, size(tr)[1]) ] 
	# Concatenate columns then transpose to get an n×3 matrix
	trj = hcat(xtrj...)'   
	p1, p2 = pt_lgeom(epos, elcc, sipm)
	p1 = plott!(1, p1, trj)
	p2 = plott!(2, p2, trj)
	
    plot(p1,p2)
end

# ╔═╡ 2144e49f-1505-4c19-a047-7733c7cfc0c1
p_trajectory(electron_pos, elcc, sipm, tr) 

# ╔═╡ 734dc505-9e93-4dd9-b939-6eba4317cb27


# ╔═╡ 9d86737d-1009-4847-84ee-b5073408acde
"""

Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
point (shown in blue) on the SiPM plane.
- `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
- `photon_impact` is a tuple (x, z) on the SiPM plane.
"""
function plt_trajectory(traj::AbstractMatrix, 
	                     xyl ::ELCCGeometry, sipm::SiPMGeometry)
	
	function plott(i, title)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		p = plot(traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false,
			  #xlims=(xymin, xymax), ylims = (zmin, zmax),
			  xlims=(sixy1l, sixy2r), ylims = (zmin, zmax),
		      xlabel="x (mm)", ylabel="z (mm)", title=title)

		p = plot!(p, [xyl, xyr], [elcc.Zg, elcc.Zg], 
			       lw=1, lc=:red, linestyle=:dash, label=false)
		p = plot!(p, [xyl, xyr], [elcc.Za, elcc.Za], 
			       lw=1, lc=:red, linestyle=:dash, label=false)

		p = plot!(p, [sixy1l, sixy1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)
		
		p = plot!(p, [sixy2l, sixy2r], [elcc.Zs + 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)

		p = plot!(p, [sixy0l, sixy0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)
	
		p = plot!(p, [xyl, xyl], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:dash, label=false)
		
		p = plot!(p, [xyr, xyr], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:dash, label=false)
	end
		
	xymin = 0
	xyl = (elcc.pitch - elcc.d_hole)/2
	xy0 = xyl +  elcc.d_hole/2
	xyr = xyl + elcc.d_hole
	xymax = elcc.pitch
	zmax = elcc.Zc
	zmin = elcc.Zs 

	sixy0l = 0
	sixy0r = sixy0l + sipm.sipmSize
	sixy1l = sixy0l - sipm.pitch
	sixy1r = sixy1l + sipm.sipmSize
	sixy2r = sixy0r + sipm.pitch
	sixy2l = sixy2r - sipm.sipmSize
	
	
	println("xymin= $(xymin), xymax = $(xymax), zmin =$(zmin), zmax =$(zmax)")
	println("xyl= $(xyl), xyr = $(xyr), xy0 =$(xy0)")
	println("sixy0l= $(sixy0l), sixy0r = $(sixy0r), sixy1l= $(sixy1l), sixy1r = $(sixy1r), sixy2l =$(sixy2l), sixy2r =$(sixy2r)")
	
	p1 = plott(1, "Trajectory (x,z)")
	p2 = plott(2, "Trajectory (y,z)")
	
    p1,p2
end

# ╔═╡ d586338c-ad5a-42b4-aa12-ec2a59b583f8
"""

Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
point (shown in blue) on the SiPM plane.
- `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
- `photon_impact` is a tuple (x, z) on the SiPM plane.
"""
function plot_gammas(gammas::Vector{Vector{Vector{Float64}}}, 
	                 traj::AbstractMatrix, 
	                 xyl::ELCCGeometry, sipm::SiPMGeometry;
                     num_to_plot::Int=1)
	
	p1,p2 = plt_trajectory(traj, xyl, sipm)

	
	# Create a 3D plot with axis labels and a title.
	#plt = plot(title="Gamma Trajectories", xlabel="x", ylabel="y", zlabel="z", legend=:outertopright)

	# Use a built-in palette for distinct colors.
	colors = palette(:tab10)

	for i in 1:num_to_plot
    	gamma = gammas[i]
   	 # Extract x, y, z coordinates from each step in the gamma.
    	xs = [step[1] for step in gamma]
    	ys = [step[2] for step in gamma]
    	zs = [step[3] for step in gamma]
    
    	# Select color for this gamma.
    	col = colors[(i - 1) % length(colors) + 1]
    
    	# Plot points for each step.
    	p1 = scatter!(p1, xs, ys, zs, label=false, marker=:circle, markersize=1, 
			     color=col)
    	# Connect the points with a dashed line.
    	p1 =plot!(p1, xs, ys, zs, label="", linestyle=:dash, linewidth=1, color=col)
	end

    plot(p1,p2)
end

# ╔═╡ 0aa7a4ef-5136-4126-a486-c40eaad99557
plot_gammas(gammas,tr,  elcc, sipm; num_to_plot=5)

# ╔═╡ 36c8c1d9-689b-446e-a8d0-83c7b5115944
"""

Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
point (shown in blue) on the SiPM plane.
- `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
- `photon_impact` is a tuple (x, z) on the SiPM plane.
"""
function plot_trajectory(traj::AbstractMatrix, 
	                     xyl ::ELCCGeometry, sipm::SiPMGeometry)
	
	p1,p2 = plt_trajectory(traj, xyl, sipm)
	
    plot(p1,p2)
end

# ╔═╡ a91f3f93-ffec-47fc-819e-e4f43bee7f95
"""
Simulation 
    
electron_pos: absolute (x,y) position where the electron arrives on the ELCC.

Determine dice assignment and local coordinates.
"""
function run_simulation(elcc::ELCCGeometry, sipm::SiPMGeometry, electron_pos::Tuple{Float64,Float64})

    dice_indices, dice_origin, xlocal = find_dice(electron_pos[1], electron_pos[2], elcc)
    println("Electron at $(electron_pos) assigned to dice $(dice_indices) with dice origin $(dice_origin) and local coords $(xlocal)")
    
    # Get a trajectory for the electron (assuming it follows a straight line through the hole).
    traj = get_trajectory_in_hole(dice_origin, elcc)
    
    # Simulate photon generation along the trajectory.
    photon_info = simulate_photons_along_trajectory(traj, sipm; N_photons_per_step=10)
    if photon_info !== nothing
        println("Photon hit on SiPM at index $(photon_info[1]), $(photon_info[2]) with total photons = $(photon_info[3])")
    else
        println("Photon impact was lost (did not hit an active SiPM).")
    end
    
    return (traj, dice_indices, dice_origin, xlocal, traj, photon_info)
end

# ╔═╡ 42d12a7c-e067-4342-860e-ad3530913094
md"""
### Intersection of a Line with a Finite Cylinder

#### 1. Equation of the Line
The line is given in parametric form as:

``
\begin{aligned}
x(t) &= x + v_x\,t,\\,
y(t) &= y + v_y\,t,\\,
z(t) &= z + v_z\,t,
\end{aligned}
``

where ``(x,y,z)`` is a point on the line and ``(v_x,v_y,v_z)`` are the direction cosines.

#### 2. Description of the Cylinder

The cylinder is defined by:

- **Lateral Surface:** All points satisfying

  ``
  (x - x_0)^2 + (y - y_0)^2 = R^2,
  ``

  where ``(x_0, y_0)`` is the center of the cylinder’s circular cross-section in the ``xy``-plane and ``R`` is the cylinder’s radius.

- **End-Caps:** Two horizontal planes at:

  ``
  z = z_a \quad (\text{top cap}) \quad \text{and} \quad z = z_b \quad (\text{bottom cap}),
  ``

  with ``z_a > z_b``.

#### 3. Intersection with the Lateral Surface
Substitute the parametric equations of the line into the cylinder’s equation:

``
\bigl(x + v_x\,t - x_0\bigr)^2 + \bigl(y + v_y\,t - y_0\bigr)^2 = R^2.
``

Expanding, we obtain a quadratic in \(t\):

``
(v_x^2 + v_y^2)t^2 + 2\bigl[v_x(x-x_0) + v_y(y-y_0)\bigr]t + \Bigl[(x-x_0)^2+(y-y_0)^2 - R^2\Bigr] = 0.
``

Let

``
a = v_x^2 + v_y^2,\quad b = 2\bigl[v_x(x-x_0) + v_y(y-y_0)\bigr],\quad c = (x-x_0)^2 + (y-y_0)^2 - R^2.
``

Then the quadratic equation is:

``
a\,t^2 + b\,t + c = 0.
``

The solutions are:

``
t = \frac{-b \pm \sqrt{b^2-4ac}}{2a}.
``

For real intersections, the discriminant \(D = b^2 - 4ac\) must be non-negative. For each valid \(t\), the \(z\)-coordinate is given by:

``
z(t) = z + v_z\,t,
``

and must satisfy ``z_b \le z(t) \le z_a`` to lie within the finite cylinder.

#### 4. Intersection with the End-Caps

**Top End-Cap (at ``z = z_a``):**

- Set the ``z`` equation equal to ``z_a``:

``
  z + v_z\,t = z_a \quad \Longrightarrow \quad t = \frac{z_a - z}{v_z}\quad (v_z \neq 0).
``

- The corresponding ``x`` and ``y`` coordinates are:

  ``
  x(t) = x + v_x\,t,\quad y(t) = y + v_y\,t.
  ``

- This intersection is valid if:

  ``
  (x(t)-x_0)^2 + (y(t)-y_0)^2 \le R^2.
  ``


**Bottom End-Cap (at ``z = z_b``):**
- Similarly, set:

  ``
  z + v_z\,t = z_b \quad \Longrightarrow \quad t = \frac{z_b - z}{v_z}\quad (v_z \neq 0).
  ``

- The intersection is valid if:

  ``
  (x(t)-x_0)^2 + (y(t)-y_0)^2 \le R^2.
  ``


"""

# ╔═╡ 569026e1-b82f-4230-b8f5-4fe60afd2cb7


# ╔═╡ Cell order:
# ╠═7884ab66-f9e2-11ef-03ea-f10faf671dba
# ╠═108bd997-6d2d-416d-b2df-ec034273d62e
# ╠═a333497c-c5fc-49a7-a8ca-d82a7dcd27ad
# ╠═57248e63-9e36-4644-a6c4-5a3aa1808e29
# ╠═eada2802-68a7-4663-a2c0-c1ca41b74601
# ╠═279bd076-8960-4f17-b719-f3d985ffdd5c
# ╠═1046cd03-a4b0-411f-8107-a2d3df10a386
# ╠═aa3b92ad-7fbf-4b3e-9eae-b1e9dcaf2eb4
# ╠═2617ae05-e1db-473a-9b0f-befeea6d0e12
# ╠═a891cff0-6910-4f78-8fc5-ff4e90163a7e
# ╠═3b1d7427-73ca-4dca-99f9-93b2cb6df9a8
# ╠═6097b338-4107-4d8e-9ee3-3f806f73c45b
# ╠═af54e98e-15fb-4ad0-a990-66e183265867
# ╠═ebe9308e-87f5-410e-a17d-deb59b59a62d
# ╠═ac79ab2e-af61-499a-94e7-964a8f04b111
# ╠═2c67d2f7-4d20-4ff6-ba81-c6902980478d
# ╠═b2c37cf5-3a69-408c-9324-7ec1cdef6d18
# ╠═ce43b3cb-69b9-43d3-beba-d83ac5f0f1a6
# ╠═321fb432-4464-47b8-94ac-30d466670224
# ╠═154133b1-81bd-4bfe-86dc-cb3ccfec48f0
# ╠═a0dfd610-50ca-4a75-9ab8-8c3937f31c33
# ╠═dfd7cbf8-adaa-454f-957e-ecc6eee905d3
# ╠═16e4221a-4dd8-4571-8ce2-ef259400562a
# ╠═a340f566-c9c0-4293-988e-11b7e69e8e4a
# ╠═c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
# ╠═7c38062b-0671-451c-911e-f88272f97937
# ╠═2273c136-4709-43a6-bf68-1184493fbb70
# ╠═9eb86c8c-4347-46c4-9111-793f921fac56
# ╠═ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
# ╠═951ea570-d232-47a3-bbe8-b216de1469a8
# ╠═2144e49f-1505-4c19-a047-7733c7cfc0c1
# ╠═d474342a-81ca-4504-86a9-52925211b685
# ╠═1ef1b221-da82-4852-bfb3-ffe1b2b50600
# ╠═c5924aa7-a04b-4820-aafb-2c71a5bb289d
# ╠═27669660-d21b-4e10-904d-b8142e8447dd
# ╠═b4bec083-392c-45f3-b440-91edd3b5e5fc
# ╠═8009f2f9-8931-45f4-9f66-41ea283563a5
# ╠═42392447-15b4-48ed-ad52-cf00aefc435f
# ╠═604a8022-d687-4684-8fd2-0178a3192452
# ╠═e361a98f-f008-48f5-a4cb-94a100702460
# ╠═7ec27076-edd8-4d3f-b691-8cf877144f98
# ╠═1dad2fcb-836e-46a8-bb2b-8d43f25c4767
# ╠═d586338c-ad5a-42b4-aa12-ec2a59b583f8
# ╠═23167ba6-4ddc-49b8-9aec-b9148d09befc
# ╠═57dec276-2cd4-4f12-9595-8cb42cbf08d9
# ╠═578639df-4bec-42cf-97c4-0b510cd32b26
# ╠═ef38565b-5a8d-41c2-a4f1-21cfbe0cb3aa
# ╠═8c7035f0-82fd-4c86-ab63-5cdd3b4d7539
# ╠═ae488bf8-706a-4d57-8ac7-412f0a43bd08
# ╠═8f537416-301f-427f-a584-b91cbd83450d
# ╠═657f8596-decb-41fc-bb83-f23839d8de32
# ╠═84704333-6927-4b78-be39-37e32c90ef15
# ╠═dbc021f7-9b18-4aa2-83d3-ad44bb887f58
# ╠═baadcc6b-36cb-49ae-aa06-686b267b4b4c
# ╠═a5d8d56f-641c-4a45-88e3-737f992802e2
# ╠═02ffafb2-8ca6-4fcd-a267-c5e7528137cc
# ╠═e8f9953e-4983-4825-abfa-829653aa0d26
# ╠═e97ec2de-accb-4336-bf0e-2247bbaeb3e2
# ╠═0aa7a4ef-5136-4126-a486-c40eaad99557
# ╠═23d039f4-b1db-4778-be0b-2fa01075a1a2
# ╠═0e31a7b1-e95c-477a-9212-a5a1726370e5
# ╠═b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
# ╠═f6e1dced-e962-4884-83aa-9f181a65982e
# ╠═5f2f631e-f305-49c2-88f4-dbf9be2c97a5
# ╠═ba7cb219-b68d-4414-9da0-e0c113db5c24
# ╠═f400c977-03df-4ea3-b93a-c66bd386ab04
# ╠═d6f88078-ee18-4ff0-a2a1-67e6005d0b39
# ╠═94a48200-aca1-4167-9e45-581e53cfdad5
# ╠═c2f6d14e-cfb7-4f53-88cc-9da2676f1ecb
# ╠═3c238213-30c7-41a7-bd7b-50f8c09b7adf
# ╠═05ac2255-16ed-4bdd-a4c4-c9e611cda5d0
# ╠═9e8d1efe-0f54-4afc-a956-3664cf972d8a
# ╠═9ebb47ac-b76c-4be4-8336-6da26f26c6c5
# ╠═5b913924-1f00-4fc7-9c6a-0ede1d400f5c
# ╠═b501a1f1-2c96-4680-b08a-982d2877603d
# ╠═c7704f94-2ab5-4111-ac7c-63f978c7ee4c
# ╠═fcbf9e5a-b7f2-400d-87af-7448fd348071
# ╠═c1fec8e5-bf38-4d39-8224-bf0051fc08eb
# ╠═af2b2805-9e6c-4078-9427-02f787212f19
# ╠═d1aec6ee-f530-4a15-9bf9-3081c7f55f4a
# ╠═232fab2c-fd22-449c-be78-f4e55c7021e8
# ╠═5bee9446-a537-4012-95d2-77c91f27c83a
# ╠═82882beb-98b0-4a53-9f0d-9d16bcbc6c09
# ╠═79a51e0b-60ec-4ce7-b15b-1f4d88c6aa28
# ╠═2b780346-122b-463f-ac4b-498e45dfa84f
# ╠═60269ab6-d610-409e-90de-48022143ef1e
# ╠═90d67f13-a612-42ec-9a21-fd40d822c17c
# ╠═36c8c1d9-689b-446e-a8d0-83c7b5115944
# ╠═734dc505-9e93-4dd9-b939-6eba4317cb27
# ╠═9d86737d-1009-4847-84ee-b5073408acde
# ╠═a91f3f93-ffec-47fc-819e-e4f43bee7f95
# ╠═42d12a7c-e067-4342-860e-ad3530913094
# ╠═569026e1-b82f-4230-b8f5-4fe60afd2cb7
