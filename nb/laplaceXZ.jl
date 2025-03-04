### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 74d371f0-f6b8-11ef-09b9-0b4b37b3c7fc
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")


# ╔═╡ 4d79a667-8ed7-4562-adc2-ccf59071efa2
begin
	using PlutoUI
	using Plots
	using Random
	using Printf
	using InteractiveUtils
	using SparseArrays, LinearAlgebra
end

# ╔═╡ 18c68f2f-23d8-4e61-b20c-b2cdc15dd48d
import PyPlot

# ╔═╡ d9d86bea-afba-4532-beff-fbcc0b662930
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

# ╔═╡ b47f446c-db95-4c45-a399-2c1b5f751bca
begin
	cd("/Users/jjgomezcadenas/Projects/JNEXT/nb")
	jn = ingredients("../src/JNEXT.jl")
end

# ╔═╡ d08bb2f0-3714-48ec-93ec-58863ce119b8
begin
	V=1e+3
	kV=1e+3V
	l0 = -10.0      # ground, z=0, potential = 0 V
	l1 = -5.0      # anode, z=l1, potential = Va
	l2 = 0.0     # gate, z=l2, potential = Vg
	l3 = 10.0     # reference electrode, z=l3, potential = V0

	V0 = -10.5kV  # potential at z=l3
	Vg = -10.0kV  # potential at gate (only in hole)
	Va = 1kV    # potential at anode (only in hole)


	# Hole geometry in x
	x1_center = 0.0    # center of hole 1 (mm)
	p = 5.0             # pitch between holes (mm)
	d1 = 2.5            # hole diameter (mm)

	# x-domain: from x1 - p/2 to x2 + p/2
	x_min = x1_center - p/2
	x_max = x1_center + p/2

	params = jn.JNEXT.GalaParams(x1_center, d1, p, l0, l1 , l2, l3, Va, Vg, V0)

	Nx = 100
	Nz = 500
	
	md"""
	Gala structure: 
	- Ground plane located at $(l0) mm, V0 = 0 V
	- Anode located at $(l1) mm, Va = $(Va/V) V
	- Gate located at $(l2) mm, Va = $(Vg/kV) kV
	- Drift located at $(l3) mm, Vd = $(V0/kV) kV
	- Hole located at $(x1_center) mm, between $(x_min) and $(x_max) with diameter $(d1)
	"""
	
end

# ╔═╡ 631febe1-dacf-4728-98e3-6b2de9a3d902
x, z, phi = jn.JNEXT.phi2d(params, Nx, Nz)

# ╔═╡ f54385bf-676f-4598-a956-8ca2e9f920c3
E_x, E_z = jn.JNEXT.exz(x, z, phi, Nx, Nz)

# ╔═╡ 76c008ef-632b-4f58-be36-ac8d8f29f788
begin
	n_electrons = 100
	electron_x0 = collect(LinRange(x_min, x_max, n_electrons))
	electron_z0 = params.l3
	electron_zend = params.l1
	ftrj, btrj = jn.JNEXT.simulate_electron_transport(x, z, E_x, E_z, electron_x0, params, dt=0.1, max_steps=5000)
end

# ╔═╡ 100b15ac-5a04-4d79-b1da-855602532136


# ╔═╡ b0e6f483-a15c-46ac-92eb-bcf82c54dbd9
size(ftrj)

# ╔═╡ 23ff4ced-4a63-4e38-a411-05d61b6eb506
heatmap(x,z,phi)

# ╔═╡ 9a2f127e-ad48-40b5-a2ca-1299e8b412f4
pyplot()

# ╔═╡ 5ad9b2ed-4fa6-4188-b553-9f9aa7ab4dcc
xphi = fill(0.0, 3, 3, 3)

# ╔═╡ 7713a836-8a96-4862-8cf4-cea535d53097
function lt5(x)
	if x < 5
		true
	else
		false
	end
end

# ╔═╡ 65429cc0-0654-4315-9573-5a99bd25934e
!lt5(3)

# ╔═╡ e0dc020b-d422-41b0-8362-820f1491a36e
md"""
## Local functions
"""

# ╔═╡ 886229eb-2bdc-4396-9360-a2dedcf12d0a
function meshgrid(x::AbstractVector, y::AbstractVector)
        X = repeat(reshape(x, 1, length(x)), length(y), 1)
        Y = repeat(reshape(y, length(y), 1), 1, length(x))
        return X, Y
end

# ╔═╡ 109a0354-66e9-474b-8bd2-b7a8e2546d78
function contour_phi(x, z, phi; size=(800,600))
	contour(x, z, phi,
            levels = 20,
            linewidth = 0.5,
            c = :viridis,
            xlabel = "x (mm)",
            ylabel = "z (mm)",
            title = "Equipotential lines and electron trajectories",
            size = size)
end

# ╔═╡ 7cff5452-ba50-486c-a20f-409ee12dd2a3
function contour_phi_py(x, z, phi; figsize=(8,6))
    # Create meshgrid arrays from x and z
    X, Z = meshgrid(x, z)
    
    # Create figure with the desired size
    fig = PyPlot.figure(figsize=figsize)
    
    # Plot the contour lines
    cs = PyPlot.contour(X, Z, phi, levels=20, linewidths=0.5, cmap="viridis")
    
    # Add contour labels
	label_levels = cs.levels[1:3:end]
    PyPlot.clabel(cs, levels=label_levels, inline=true, fontsize=8)
    
    PyPlot.xlabel("x (mm)")
    PyPlot.ylabel("z (mm)")
    PyPlot.title("Equipotential lines and electron trajectories")
    return fig
end

# ╔═╡ 5daea9a9-91cb-484c-9eeb-21b9803e6e50

contour_phi_py(x, z, phi; figsize=(8,6))

# ╔═╡ dc448199-e877-4a4e-8ef7-495490a6c691


function plot_traj(x, z, phi, trajectories, btr, params; figsize=(8,6))
    # Extract parameters
    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p  = params.p           # pitch between holes (mm)
    
    start_z = params.l3
    gate    = params.l2
    anode   = params.l1

    # Create meshgrid arrays (X and Z will have dimensions (length(z), length(x)))
    X, Z = meshgrid(x, z)

    # Create figure and axis
    fig, ax = PyPlot.subplots(figsize=figsize)

    # Plot equipotential contours
    contours = ax.contour(X, Z, phi, levels=20, linewidths=1.0, cmap="viridis")
    ax.clabel(contours, inline=true, fontsize=8)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Equipotential lines and electron trajectories")

    # Plot electron trajectories (first set in green)
    for traj in trajectories
        # In Julia, traj[:,1] is the x-coordinate and traj[:,2] is the z-coordinate.
        ax.plot(traj[:, 1], traj[:, 2], "g", lw=0.3)
    end

    # Plot secondary trajectories (btr in red)
    for traj in btr
        ax.plot(traj[:, 1], traj[:, 2], "r", lw=0.3)
    end

    # Plot vertical lines for the holes
    PyPlot.axvline(x = x1_center - d1/2, color="black", linestyle="--", linewidth=1)
    PyPlot.axvline(x = x1_center + d1/2, color="black", linestyle="--", linewidth=1)
    PyPlot.axvline(x = x1_center - p/2, color="black", linestyle="--", linewidth=1)
    PyPlot.axvline(x = x1_center + p/2, color="black", linestyle="--", linewidth=1)
    # Plot horizontal lines for start, gate and anode positions
    PyPlot.axhline(y = start_z, color="blue", linestyle="--", linewidth=1)
    PyPlot.axhline(y = gate, color="blue", linestyle="--", linewidth=1)
    PyPlot.axhline(y = anode, color="blue", linestyle="--", linewidth=1)

    PyPlot.tight_layout()
    return fig 
end

# Example usage:
# Assuming x, z, phi, trajectories, btr, and params are defined appropriately:
# 

# ╔═╡ 15a8058c-2a2c-4331-ae66-788ac60c48d6
plot_traj(x, z, phi, ftrj, btrj, params, figsize=(8,6))

# ╔═╡ Cell order:
# ╠═74d371f0-f6b8-11ef-09b9-0b4b37b3c7fc
# ╠═4d79a667-8ed7-4562-adc2-ccf59071efa2
# ╠═18c68f2f-23d8-4e61-b20c-b2cdc15dd48d
# ╠═d9d86bea-afba-4532-beff-fbcc0b662930
# ╠═b47f446c-db95-4c45-a399-2c1b5f751bca
# ╠═d08bb2f0-3714-48ec-93ec-58863ce119b8
# ╠═631febe1-dacf-4728-98e3-6b2de9a3d902
# ╠═f54385bf-676f-4598-a956-8ca2e9f920c3
# ╠═76c008ef-632b-4f58-be36-ac8d8f29f788
# ╠═100b15ac-5a04-4d79-b1da-855602532136
# ╠═b0e6f483-a15c-46ac-92eb-bcf82c54dbd9
# ╠═23ff4ced-4a63-4e38-a411-05d61b6eb506
# ╠═9a2f127e-ad48-40b5-a2ca-1299e8b412f4
# ╠═5daea9a9-91cb-484c-9eeb-21b9803e6e50
# ╠═15a8058c-2a2c-4331-ae66-788ac60c48d6
# ╠═5ad9b2ed-4fa6-4188-b553-9f9aa7ab4dcc
# ╠═7713a836-8a96-4862-8cf4-cea535d53097
# ╠═65429cc0-0654-4315-9573-5a99bd25934e
# ╠═e0dc020b-d422-41b0-8362-820f1491a36e
# ╠═886229eb-2bdc-4396-9360-a2dedcf12d0a
# ╠═109a0354-66e9-474b-8bd2-b7a8e2546d78
# ╠═7cff5452-ba50-486c-a20f-409ee12dd2a3
# ╠═dc448199-e877-4a4e-8ef7-495490a6c691
