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
	using JLD2
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


	x0 = 3.0    # center in x
    y0 = 3.0    # center in y
    d1 = 3.0    # hole diameter (in x-y plane)
    p  = 6.0    # pitch (used here to define domain size in x and y)
    z0 = -10.0  # ground position
    za= -5.0    # anode position
    zg= 0.0     # gate position
    zc= 10.0    # collector position

	Vc = -10.5kV  # potential at the collector
	Vg = -10.0kV  # potential at gate 
	Va = 1kV      # potential at anode 
	V0 = 0.0      # potential at ground

	
	par = jn.JNEXT.ParH3D(x0, y0, d1, p, z0, za, zg, zc, V0, Va, Vg, Vc)
    
	Nx = 90
	Ny = 90
	Nz = 100

	x_min = par.x0 - par.p/2 
    x_max = par.x0 + par.p/2 
    y_min = par.y0 - par.p/2 
    y_max = par.y0 + par.p/2 
   
    # Compute grid spacings
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    dz = (par.zc - par.z0) / (Nz - 1)

	file = string("phi3d_d", string(Int(d1)), "_p_", string(Int(p)),".jld2")
	
	md"""
	Gala3D structure: 
	- Ground plane located at $(z0) mm, V0 = 0 V
	- Anode located at $(za) mm, Va = $(Va/V) V
	- Gate located at $(zg) mm, Vg = $(Vg/kV) kV
	- Collector located at $(zc) mm, Vc = $(Vc/kV) kV
	- Hole located at ($(x0), $(y0)) with diameter $(d1) and pitch $(p)
	- Domain in x between $(x_min) and $(x_max) 
	- Domain in y between $(y_min) and $(y_max)
	- Grid spacing: dx =$(dx), dy = $(dy), dz = $(dz)
	- file name = $(file)
	"""
	
end

# ╔═╡ ca8eeeca-615d-46bd-b785-82ff7719a1de
md"""
- Click in box to recompute potential
"""

# ╔═╡ 3b780ca8-3e52-4b46-bbd5-dc6f82d02bf6
 @bind ComputePhi3d CheckBox()

# ╔═╡ 344d26c6-6977-4578-9a4e-dfdd30fd76ba
if ComputePhi3d
	let
		phi_full, xd, yd, zd = jn.JNEXT.phi3D(par, Nx, Ny, Nz)
		@save file phi_full xd yd zd
	end
end
	

# ╔═╡ 717fc78b-6fa1-4251-ac69-d268b7cca214
@load file phi_full xd yd zd

# ╔═╡ 53f9bd98-de0e-45b4-a1fd-48b4ebdbffdf
#phi3ds, xs, ys, zs = jn.JNEXT.phi3D_sor(par, Nx, Ny, Nz; ω=1.5, tol=1e-6, max_iter=10000)

# ╔═╡ 8b677dce-ba87-4ccd-9234-91328b3fd5b9
#cpxzs = contour_xz(phi3ds, xs, ys, zs, y_idx=round(Int, length(ys)/2))

# ╔═╡ 2999819c-77b4-461b-9f4b-a74d8a7201bf
#contour_yz(phi_full, xd, yd, zd, x_idx=round(Int, length(xd)/2))

# ╔═╡ 7e7ae9ed-dc26-409c-aedf-3d27a797dc13
E_x, E_y, E_z = jn.JNEXT.exyz(xd, yd, zd, phi_full, Nx, Ny, Nz)

# ╔═╡ bc5b571d-f658-4a30-b884-ac11ebabedf2


# ╔═╡ e0dc020b-d422-41b0-8362-820f1491a36e
md"""
## Local functions
"""

# ╔═╡ 05692202-4939-406e-ab7d-4a91daa10478
function generate_electron_positions(N::Int, xg::AbstractVector, yg::AbstractVector)
    # Determine the domain limits from the grid vectors:
    x_min, x_max = minimum(xg), maximum(xg)
    y_min, y_max = minimum(yg), maximum(yg)
    
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

# ╔═╡ 76c008ef-632b-4f58-be36-ac8d8f29f788
begin
	n_electrons = 300
	electron_xy0 =generate_electron_positions(n_electrons, xd, yd)
	#electron_x0 = collect(LinRange(x_min, x_max, n_electrons))
	#electron_z0 = params.l3
	#electron_zend = params.l1
	#ftrj, btrj = jn.JNEXT.simulate_electron_transport(x, z, E_x, E_z, electron_x0, params, dt=0.1, max_steps=5000)
	ftrj, btrj = jn.JNEXT.simulate_electron_transport3Dx(xd, yd, zd, E_x, E_y, E_z, 
		                                          electron_xy0, par;
								                  dt=0.1, max_steps=2000)
end

# ╔═╡ 97fd6c0b-cf19-4e0b-87b5-113a8a9da7be
ftrj

# ╔═╡ ce25e8f5-2616-42a3-bd35-dbd8d69bd37f
btrj

# ╔═╡ 1b19e24d-5d81-4125-a3bf-6441654f05b8
@save "trajectories_data.jld2" ftrj btrj

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

# ╔═╡ 1e25597c-4442-489e-a1ae-240cbfbff055
function plot_trajectories_3D(trajs, btrjs; figsize=(8,6))
    fig = PyPlot.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for traj in trajs
        xs = traj[:, 1]
        ys = traj[:, 2]
        zs = traj[:, 3]
        ax.plot(xs, ys, zs, "b", lw=0.3)
    end

	 # Plot secondary trajectories (btr in red)
    for traj in btrjs
		xs = traj[:, 1]
        ys = traj[:, 2]
        zs = traj[:, 3]
        ax.plot(xs, ys, zs, "r", lw=0.3)
    end
	
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D Electron Trajectories")
    return fig
end

# ╔═╡ b0e6f483-a15c-46ac-92eb-bcf82c54dbd9
plot_trajectories_3D(ftrj, btrj, figsize=(8,6))

# ╔═╡ 17a120a1-ca91-46cc-b1ea-87d8ab4ea29c
# 2D Projection: (x, z)
function plot_trajectories_xyz(trajs, btrjs, par; figsize=(8,6))

	x1_center = par.x0    # center of hole 1 (mm)
	y1_center = par.y0    # center of hole 1 (mm)
    d1 = par.d1
    p  = par.p           # pitch between holes (mm)
    
    start_z = par.zc
    gate    = par.zg
    anode   = par.za
	
    fig, ax = PyPlot.subplots(1, 2, figsize=figsize)
    for traj in trajs
        xs = traj[:, 1]
		ys = traj[:, 2]
        zs = traj[:, 3]
        ax[1].plot(xs, zs, "g", lw=0.3)
		ax[2].plot(ys, zs, "g", lw=0.3)
    end
	for traj in btrjs
        xs = traj[:, 1]
		ys = traj[:, 2]
        zs = traj[:, 3]
        ax[1].plot(xs, zs, "r", lw=0.3)
		ax[2].plot(ys, zs, "r", lw=0.3)
    end
    ax[1].set_xlabel("x (mm)")
    ax[1].set_ylabel("z (mm)")
    ax[1].set_title("Trajectories: x-z Projection")
	
	ax[1].axvline(x = x1_center - d1/2, color="black", linestyle="--", linewidth=1)
    ax[1].axvline(x = x1_center + d1/2, color="black", linestyle="--", linewidth=1)
    ax[1].axvline(x = x1_center - p/2, color="black", linestyle="--", linewidth=1)
    ax[1].axvline(x = x1_center + p/2, color="black", linestyle="--", linewidth=1)
    # Plot horizontal lines for start, gate and anode positions
    ax[1].axhline(y = start_z, color="blue", linestyle="--", linewidth=1)
    ax[1].axhline(y = gate, color="blue", linestyle="--", linewidth=1)
    ax[1].axhline(y = anode, color="blue", linestyle="--", linewidth=1)
	
	
	ax[2].set_xlabel("y (mm)")
    ax[2].set_ylabel("z (mm)")
    ax[2].set_title("Trajectories: y-z Projection")

	ax[2].axvline(x = y1_center - d1/2, color="black", linestyle="--", linewidth=1)
    ax[2].axvline(x = y1_center + d1/2, color="black", linestyle="--", linewidth=1)
    ax[2].axvline(x = y1_center - p/2, color="black", linestyle="--", linewidth=1)
    ax[2].axvline(x = y1_center + p/2, color="black", linestyle="--", linewidth=1)
    # Plot horizontal lines for start, gate and anode positions
    ax[2].axhline(y = start_z, color="blue", linestyle="--", linewidth=1)
    ax[2].axhline(y = gate, color="blue", linestyle="--", linewidth=1)
    ax[2].axhline(y = anode, color="blue", linestyle="--", linewidth=1)

	# Plot vertical lines for the holes
    
    PyPlot.tight_layout()
    return fig
end



# ╔═╡ 5c65025b-be1e-4e49-a81d-3c017f5292a1
plot_trajectories_xyz(ftrj, btrj, par; figsize=(8,6))

# ╔═╡ a47c4275-f3fb-4212-97c8-b08889980615
# 2D Projection: (y, z)
function plot_trajectories_yz(trajs, figsize=(8,6))
    fig, ax = subplots(figsize=figsize)
    for traj in trajs
        ys = traj[:, 2]
        zs = traj[:, 3]
        ax.plot(ys, zs, lw=0.5)
    end
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Trajectories: y-z Projection")
    return fig
end

# ╔═╡ dc448199-e877-4a4e-8ef7-495490a6c691


function plot_traj(x, z, phi, trajectories, btr, params; figsize=(8,6))
    # Extract parameters
    x1 = params.x1    # center of hole 1 (mm)
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

# ╔═╡ 4f181d98-b2e3-4325-a75d-f43f1f1d3f3c
function contour_xz(phi, xg, yg, zg; y_idx::Int=round(Int, length(phi[1,:,1])/2), figsize=(8,6))
    # Extract a slice: fix y = y_idx. phi has dims (Nx,Ny,Nz).
    # We want a 2D array with x along columns and z along rows.
    # Here, we take phi[:, y_idx, :] which has size (Nx, Nz). We then transpose it.
	
	ysl = phi[:, y_idx, :]
    phi_xz = permutedims(ysl, (2,1))
    X, Z = meshgrid(xg, zg)
    fig = PyPlot.figure(figsize=figsize)
    cs = PyPlot.contour(X, Z, phi_xz, levels=20, linewidths=0.5, cmap="viridis")
    # Label only every third level, for clarity
    label_levels = cs[:levels][1:3:end]
    PyPlot.clabel(cs, levels=label_levels, inline=true, fontsize=8)
    PyPlot.xlabel("x (mm)")
    PyPlot.ylabel("z (mm)")
    PyPlot.title("Contour Plot (x-z projection) at y-index = $y_idx (y = $(round(yg[y_idx], digits=2)) mm)")
    return fig
end

# ╔═╡ cbd15a01-e896-4563-ac6b-d7fca29c0e5a
begin 
	#fig, ax = PyPlot.subplots(figsize=figsize)
	cpxzf = contour_xz(phi_full, xd, yd, zd, y_idx=round(Int, length(yd)/2))
	##cpxzs = contour_xz(phi3ds, xs, ys, zs, y_idx=round(Int, length(ys)/2))
	#plot(cpxzf, cpxzs)
end
	

# ╔═╡ fe3220e8-1b76-4600-ba18-3f74ab10ef56
function contour_yz(phi, xg, yg, zg; x_idx::Int=round(Int, length(phi[:,1,1])/2), figsize=(8,6))
    # Extract slice: fix x = x_idx. Then phi[x_idx, :, :] is of size (Ny, Nz).
    # Transpose it so rows become z and columns become y.
    phi_yz = permutedims(phi[x_idx, :, :], (2,1))
    Y, Z = meshgrid(yg, zg)
    fig = PyPlot.figure(figsize=figsize)
    cs = PyPlot.contour(Y, Z, phi_yz, levels=20, linewidths=0.5, cmap="viridis")
    label_levels = cs[:levels][1:3:end]
    PyPlot.clabel(cs, levels=label_levels, inline=true, fontsize=8)
    PyPlot.xlabel("y (mm)")
    PyPlot.ylabel("z (mm)")
    PyPlot.title("Contour Plot (y-z projection) at x-index = $x_idx (x = $(round(xg[x_idx], digits=2)) mm)")
    return fig
end

# ╔═╡ 508d19ae-3cce-4859-93af-cfe5e74956be
begin



# 1. 3D Surface Plot for a given z-slice
function plot_surface_z(phi, xg, yg, zg; z_idx::Int=round(Int, length(zg)/2), figsize=(8,6))
    # Extract the 2D potential slice at a given z-index.
    phi_slice = phi[:,:,z_idx]
    # Create meshgrid for x and y (note: PyPlot.meshgrid produces arrays with dims (ny, nx))
    X, Y = meshgrid(xg, yg)
    fig = PyPlot.figure(figsize=figsize)
    ax = fig[:add_subplot](111, projection="3d")
    # Create a surface plot; rstride and cstride can be tuned
    surf = ax[:plot_surface](X, Y, phi_slice; rstride=1, cstride=1, cmap="viridis")
    ax[:set_xlabel]("x (mm)")
    ax[:set_ylabel]("y (mm)")
    ax[:set_zlabel]("Potential (V)")
    ax[:set_title]("3D Surface Plot at z = $(round(zg[z_idx], digits=2)) mm")
    fig[:colorbar](surf, ax=ax)
    return fig
end

# 2. Heatmap for a given z-slice
function heatmap_slice(phi, xg, yg, zg; z_idx::Int=round(Int, length(zg)/2), figsize=(8,6))
    phi_slice = phi[:,:,z_idx]
    fig = PyPlot.figure(figsize=figsize)
    # Use imshow; extent maps the grid to x and y values, origin set to "lower" so y increases upward.
    im = PyPlot.imshow(phi_slice, extent=(first(xg), last(xg), first(yg), last(yg)), origin="lower", cmap="viridis")
    colorbar(im)
    xlabel("x (mm)")
    ylabel("y (mm)")
    title("Heatmap of Potential at z = $(round(zg[z_idx], digits=2)) mm")
    return fig
end


end
# Example usage:
# Assuming phi, xg, yg, zg are already computed (for example, via phi3D or phi3D_opt/sor)
# fig1 = plot_surface_z(phi, xg, yg, zg, z_idx=round(Int, length(zg)/2))
# fig2 = heatmap_slice(phi, xg, yg, zg, z_idx=round(Int, length(zg)/2))
# fig3 = contour_xz(phi, xg, zg, y_idx=round(Int, length(yg)/2))
# fig4 = contour_yz(phi, yg, zg, x_idx=round(Int, length(xg)/2))

# ╔═╡ ab886a0e-7f24-470b-8f55-cd05b8767380
ps = plot_surface_z(phi_full, xd, yd, zd; z_idx=round(Int, 30), figsize=(8,6))

# ╔═╡ Cell order:
# ╠═74d371f0-f6b8-11ef-09b9-0b4b37b3c7fc
# ╠═4d79a667-8ed7-4562-adc2-ccf59071efa2
# ╠═18c68f2f-23d8-4e61-b20c-b2cdc15dd48d
# ╠═d9d86bea-afba-4532-beff-fbcc0b662930
# ╠═b47f446c-db95-4c45-a399-2c1b5f751bca
# ╠═d08bb2f0-3714-48ec-93ec-58863ce119b8
# ╠═ca8eeeca-615d-46bd-b785-82ff7719a1de
# ╠═3b780ca8-3e52-4b46-bbd5-dc6f82d02bf6
# ╠═344d26c6-6977-4578-9a4e-dfdd30fd76ba
# ╠═717fc78b-6fa1-4251-ac69-d268b7cca214
# ╠═53f9bd98-de0e-45b4-a1fd-48b4ebdbffdf
# ╠═cbd15a01-e896-4563-ac6b-d7fca29c0e5a
# ╠═8b677dce-ba87-4ccd-9234-91328b3fd5b9
# ╠═2999819c-77b4-461b-9f4b-a74d8a7201bf
# ╠═ab886a0e-7f24-470b-8f55-cd05b8767380
# ╠═7e7ae9ed-dc26-409c-aedf-3d27a797dc13
# ╠═76c008ef-632b-4f58-be36-ac8d8f29f788
# ╠═b0e6f483-a15c-46ac-92eb-bcf82c54dbd9
# ╠═97fd6c0b-cf19-4e0b-87b5-113a8a9da7be
# ╠═ce25e8f5-2616-42a3-bd35-dbd8d69bd37f
# ╠═5c65025b-be1e-4e49-a81d-3c017f5292a1
# ╠═1b19e24d-5d81-4125-a3bf-6441654f05b8
# ╠═bc5b571d-f658-4a30-b884-ac11ebabedf2
# ╠═e0dc020b-d422-41b0-8362-820f1491a36e
# ╠═05692202-4939-406e-ab7d-4a91daa10478
# ╠═886229eb-2bdc-4396-9360-a2dedcf12d0a
# ╠═109a0354-66e9-474b-8bd2-b7a8e2546d78
# ╠═7cff5452-ba50-486c-a20f-409ee12dd2a3
# ╠═1e25597c-4442-489e-a1ae-240cbfbff055
# ╠═17a120a1-ca91-46cc-b1ea-87d8ab4ea29c
# ╠═a47c4275-f3fb-4212-97c8-b08889980615
# ╠═dc448199-e877-4a4e-8ef7-495490a6c691
# ╠═4f181d98-b2e3-4325-a75d-f43f1f1d3f3c
# ╠═fe3220e8-1b76-4600-ba18-3f74ab10ef56
# ╠═508d19ae-3cce-4859-93af-cfe5e74956be
