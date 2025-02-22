### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 21e618d1-f690-4fd1-b81c-fe7c5161a612
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

# ╔═╡ 977e4fb7-8f7d-4fc9-a005-a4cd58629cde
begin
	using PlutoUI
	using Plots
	using Random
	using Test
	using Printf
	using InteractiveUtils
	using SparseArrays, LinearAlgebra
	using Unitful 
end

# ╔═╡ 41ae9712-e9f4-11ef-0520-730cc79c0144
PlutoUI.TableOfContents(title="", indent=true)

# ╔═╡ 0dac543d-dbc7-4512-b6a2-7beda4ae32a7
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

# ╔═╡ 886c9529-939c-40c8-8bd2-118a9cbf2dc8
# ╠═╡ disabled = true
#=╠═╡
begin
	cd("/Users/jjgomezcadenas/Projects/JNEXT/nb")
	jn = ingredients("../src/JNEXT.jl")
end
  ╠═╡ =#

# ╔═╡ 28440d12-d42b-48e7-837b-6b01bc66745e
begin
	cd("/Users/jjgomezcadenas/Projects/JNEXT/nb")
	jn = ingredients("../src/JNEXT.jl")
end

# ╔═╡ ab10130d-5557-41e3-ac5a-52f77a1a7594
begin
	# Define parameters for the 2D problem with two holes.
	# Hole 1: centered at x1; Hole 2: centered at x1+p.
	# Domain in x: [x1 - p/2, x1 + p + p/2]

	x1 = -2.5      # position of first hole
	d1 = 2.5       # hole diameter in mm 
	p = 5.0       # pitach 
	l1 = 2.0      # anode position
	l2 = 7.0      # gate position
	l3 = 10.0      # top electrode position
	Va = 1e+5     # anode potential (V) : V scaled up so that V/m makes sense
	#Va = 0.0     # anode potential (V) : V scaled up so that V/m makes sense
	Vg = -15e+6   # gate potential (V)
	V0 = -10e+6 # top electrode potential (V)
	
	
	gparams = jn.JNEXT.GALAParams2D(x1, d1, p, l1, l2, l3, Va, Vg, V0)
		#(A_mat, b_vec, rgrid, zgrid), (phi_mat, rgrid, zgrid)
	
	Nx=511 
	Nz=511
end


# ╔═╡ c381fcc3-2e08-43cb-bc90-11a36306db77
begin
	# Domain in x is from x1 - p/2 to (x1+p) + p/2.
	xmin = gparams.x1 - gparams.p/2
	xmax = (gparams.x1 + gparams.p) + gparams.p/2

	md"""
	
	- xmin= $(xmin)
	- xmax= $(xmax)
	"""

end

# ╔═╡ a30a68c6-b1f8-4bf6-aa99-2d7a35f9bf34
t = @elapsed begin
	phi_mat, xgrid, zgrid = jn.JNEXT.solve_laplace_gala_2d(gparams; Nx, Nz)
end


# ╔═╡ 295a107f-57f4-4928-ae22-8658d86a64a0
println("Nx = $(Nx), Nz = $(Nz): Elapsed time (Julia): ", t, " seconds")


# ╔═╡ 1c0ae936-9796-4712-9447-62a3f0903e7e
begin
	c1 = contour(xgrid, zgrid, phi_mat', xlabel="x [m]", ylabel="z [m]", 
        title="Potential Contour Plot", fill=true, color=:viridis, size=(1000,400))
	c2 = contour(xgrid, zgrid, phi_mat'; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Potential Contour Plot",
        size=(800,600))
	plot(c1, c2, layout=(1,2), size=(1400,800), margin=5Plots.mm)
end


# ╔═╡ 01610b8e-3047-4b20-bcbf-be156148f57b
surface(xgrid, zgrid, phi_mat'; xlabel="x [m]", ylabel="z [m]", zlabel="Potential [V]", camera=(0, 90), size=(1400,800),
        title="3D Surface Plot of the Potential", legend=false)

# ╔═╡ 04c12676-6f76-455d-b37d-1ec10d718446
t2 = @elapsed begin
E_x, E_z =jn.JNEXT.compute_E_field_2d(phi_mat, xgrid, zgrid)
end

# ╔═╡ 15e73bee-87e3-4209-93da-5986ceae5223
println("Elapsed time : ", t2, " seconds")


# ╔═╡ 3e5f7188-088f-423f-a0ba-d9a401aace9e
begin

	step =5  # Plot every 5th point to avoid clutter
	x_sub = xgrid[1:step:end]
	z_sub = zgrid[1:step:end]
	E_x_sub = E_x[1:step:end, 1:step:end]
	E_z_sub = E_z[1:step:end, 1:step:end]
	normE = sqrt(norm(E_x_sub) + norm(E_z_sub))
	ex = E_x_sub/norm(E_x_sub)
	ey = E_z_sub/norm(E_z_sub)
	# Create meshgrid for the subset (ensure matching dimensions)
	X_sub = [r for r in x_sub, _ in z_sub]
	Z_sub = [z for _ in x_sub, z in z_sub]
	quiver(X_sub, Z_sub, quiver=(ex, ey), xlabel="r [m]", ylabel="z [m]",
	       title="Electric Field Quiver Plot", legend=false)
end

# ╔═╡ 8f7306a1-9d7e-47d7-b266-8a2f79326b5c
begin
	c3 = contour(xgrid, zgrid, E_x; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Ex Contour Plot",
        size=(800,600))
	c4 = contour(xgrid, zgrid, E_z; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Ez Contour Plot",
        size=(800,600))
	plot(c3, c4, layout=(1,2), size=(1400,800), margin=5Plots.mm)
end

# ╔═╡ e1d9aec2-0e07-4d09-8f44-ea2feed51dec
phi_mat

# ╔═╡ 8dc1d8b3-f8a3-4465-9b7f-0cb89138c2b1
t3 = @elapsed begin
ftrj = jn.JNEXT.transport_electrons_2d(phi_mat, xgrid, zgrid, gparams.l1;           									   N_electrons=100, ds=1e-1)
end

# ╔═╡ 6121fd8e-bdd6-4dd0-a48b-352c96d5ae7a
ftrj

# ╔═╡ 2d9cdc9b-6310-4696-b604-ae8bba3fbc33
println("Elapsed time : ", t3, " seconds")

# ╔═╡ 85d4e577-f3d2-49fd-8f11-2cd65e7c7d85
md"""
## Plots
"""

# ╔═╡ c32b21ac-3d91-4936-84e3-dd1ab086ce78
function plot_electron_trajectories(ftrj; params)

	xmin = params.x1 - params.p/2
	xmax = (params.x1 + params.p) + params.p/2

	x1 = -2.5      # position of first hole
	d1 = 2.5       # hole diameter in mm 
	p = 5.0  

	x1l = params.x1 - params.d1/2
	x1r = params.x1 + params.d1/2
	x2l = params.x1 + params.p - params.d1/2
	x2r = params.x1 + params.p + params.d1/2
	
    plt = plot(legend=false,
               xlabel="r [m]", ylabel="z [m]",
               title="Electron Trajectories",
               aspect_ratio=:equal,
               xlims=(xmin, xmax), ylims=(0, params.l3))


	# Add a vertical line at x = 5.
	hline!(plt, [params.l1], lw=2, ls=:dash, color=:red, label="anode")
	hline!(plt, [params.l2], lw=2, ls=:dash, color=:red, label="gate")
	hline!(plt, [params.l3], lw=2, ls=:dash, color=:red, label="end-tpc")

	# Add a horizontal line at y = 0.5.
	vline!(plt, [x1l], lw=2, ls=:dot, color=:red, label="diameter hole")
	vline!(plt, [x1r], lw=2, ls=:dot, color=:red, label="diameter hole")
	vline!(plt, [x2l], lw=2, ls=:dot, color=:red, label="diameter hole")
	vline!(plt, [x2r], lw=2, ls=:dot, color=:red, label="diameter hole")
    
    for traj in ftrj
        # Extract r and z coordinates from each trajectory.
        r_traj = [p[1] for p in traj]
        z_traj = [p[2] for p in traj]
        plot!(plt, r_traj, z_traj, lw=1, color=:blue)
    end

	#for traj in btrj
        # Extract r and z coordinates from each trajectory.
    #    r_traj = [p[1] for p in traj]
    #    plot!(plt, r_traj, z_traj, lw=1, color=:green)
    #    z_traj = [p[2] for p in traj]
    #end
    
    return plt
end

# ╔═╡ bd6d4471-7aee-46af-8a0d-86b75a7b5629
plot_electron_trajectories(ftrj; params=gparams)

# ╔═╡ ed6b9d7a-3111-4e9e-87ad-320b4392477f
md"""
# GALA Problem in 2D (x,z) with Two Holes

The GALA problem is an electrostatic design problem in which we want to compute the potential distribution in a device having several electrodes separated by a dielectric layer. In the configuration considered here, the electrodes are arranged in the vertical (z) direction and in one longitudinal dimension (x). 

The electrodes are:

- The reference: An infinite metal electrode, in position ``l_3`` held at V0 voltage. Example: ``l_3 = 10 mm, V0=-12500 V``.

- The gate: An infinite metal electrode in position ``l_2`` held at ``Vg > V0``.
Exmple: ``l_2 = 8 mm, V0=-12000 V``.

- The anode: An infinite metal electrode in position ``l_1`` held at ``Va > Vg``.
Exmple: ``l_1 = 3 mm, Va=5000 V``.

- The Ground: An infinite metal electrode in position ``l_0 = 0 mm`` held at ``Vg =0``.

Then, the GALA geometry is created by inserting a dielectric (e.g., metacrilate) between the gate and the anode, and opening two holes that connect the gate with the anode. 

### Device Geometry and Boundary Conditions

- **Domain:**  
  The computational domain is defined in the (x,z) half‐plane:

In the horizontal (x) direction the device contains two holes:
- **Hole 1** is centered at ``x_1 `` and has a diameter ``d_1``. 
Example, ``x:1 = -2.5 mm``. 

- **Hole 2** is centered at ``x_2 = x_1 + p ``, where ``p`` is the pitch between holes. Example: ``p = 5 mm``.

- The region of interest in ``x`` is defined as
 ``
  x \in \left[x_1 - \frac{p}{2},\; x_2 + \frac{p}{2}\right].
  ``

  
**Boundary conditions in x:**  
At ``x = x_{min} = x_1 - p/2`` and ``x = x_{max} = x_2 + p/2``, Neumann conditions (zero normal derivative) are imposed.


**Electrode Conditions in z:**  
- At ``l_0=0, z = 0 ``: ``\phi(x,0)=0``.  
- At ``z=l_3 ``: ``\phi(x,l_3)=V_0``.  
- At ``z=l_1`` (anode) and ``z=l_2`` (gate), Dirichlet conditions are imposed **only within the holes**. A point ``(x,z)`` on one of these planes is inside a hole if:
  - For Hole 1: ``x \in \left[x_1 - \frac{d_1}{2},\; x_1 + \frac{d_1}{2}\right] ``.
  - For Hole 2: ``x \in \left[x_2 - \frac{d_1}{2},\; x_2 + \frac{d_1}{2}\right] ``.


**Finite Difference Algorithm:**
1. **Grid Generation:**  
   Uniform grids are created in ``x`` (from ``x_{min}`` to ``x_{max}`` and in ``z`` (from ``0`` to ``l_3``).

2. **Assembly:**  
   For interior points, central differences approximate the second derivatives:
   - In ``z``:  
     ``\frac{\partial^2 \phi}{\partial z^2} \approx \frac{\phi(x,z+\Delta z) - 2\phi(x,z) + \phi(x,z-\Delta z)}{\Delta z^2}``.
   - In ``x``:  
     ``\frac{\partial^2 \phi}{\partial x^2} \approx \frac{\phi(x+\Delta x,z) - 2\phi(x,z) + \phi(x-\Delta x,z)}{\Delta x^2}``.
   
   At boundaries in ``x``, one-sided differences are used to enforce Neumann (zero derivative) conditions; at ``z=0`` and ``z=l_3`` the potential is fixed. At ``z=l_1`` and ``z=l_2`` the potential is fixed only for grid points inside the holes.
   
   This yields a sparse linear system ``A\phi = b``, which is solved to get ``\phi(x,z)``.

3. **Electric Field:**  
   The electric field is given by  
   ``
   \mathbf{E}(x,z) = -\nabla \phi(x,z),
   ``
   and its components are approximated by finite differences (forward/backward at boundaries, central in the interior).

4. **Electron Transport:**  
   Electrons are initialized at `` = l_3`` at uniformly spaced ``x``-positions (within the domain). Then, using Euler integration, each electron is transported along the negative gradient of the potential (i.e. opposite to the electric field, since electrons are negatively charged) until the electron reaches the anode (i.e. ``z \le l_1``).

---
"""

# ╔═╡ Cell order:
# ╠═21e618d1-f690-4fd1-b81c-fe7c5161a612
# ╠═977e4fb7-8f7d-4fc9-a005-a4cd58629cde
# ╠═41ae9712-e9f4-11ef-0520-730cc79c0144
# ╠═0dac543d-dbc7-4512-b6a2-7beda4ae32a7
# ╟─886c9529-939c-40c8-8bd2-118a9cbf2dc8
# ╠═28440d12-d42b-48e7-837b-6b01bc66745e
# ╠═ab10130d-5557-41e3-ac5a-52f77a1a7594
# ╠═c381fcc3-2e08-43cb-bc90-11a36306db77
# ╠═a30a68c6-b1f8-4bf6-aa99-2d7a35f9bf34
# ╠═295a107f-57f4-4928-ae22-8658d86a64a0
# ╠═1c0ae936-9796-4712-9447-62a3f0903e7e
# ╠═01610b8e-3047-4b20-bcbf-be156148f57b
# ╠═04c12676-6f76-455d-b37d-1ec10d718446
# ╠═15e73bee-87e3-4209-93da-5986ceae5223
# ╠═3e5f7188-088f-423f-a0ba-d9a401aace9e
# ╠═8f7306a1-9d7e-47d7-b266-8a2f79326b5c
# ╠═e1d9aec2-0e07-4d09-8f44-ea2feed51dec
# ╠═8dc1d8b3-f8a3-4465-9b7f-0cb89138c2b1
# ╠═6121fd8e-bdd6-4dd0-a48b-352c96d5ae7a
# ╠═2d9cdc9b-6310-4696-b604-ae8bba3fbc33
# ╠═bd6d4471-7aee-46af-8a0d-86b75a7b5629
# ╠═85d4e577-f3d2-49fd-8f11-2cd65e7c7d85
# ╠═c32b21ac-3d91-4936-84e3-dd1ab086ce78
# ╠═ed6b9d7a-3111-4e9e-87ad-320b4392477f
