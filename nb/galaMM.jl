### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3e0354b1-fef0-4e17-8b14-8e128f5b55af
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")


# ╔═╡ 675394ae-bfec-412c-b408-5bbc83a500f3
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

# ╔═╡ 3766fec6-e89f-11ef-1716-1f64cc6c7d76
md"""
# The GALA Problem in Cylindrical Coordinates (One Hole)

The GALA problem is an electrostatic design problem in which we want to control the potential distribution in a device having several electrodes separated by a dielectric layer. In the configuration considered here, the electrodes are arranged in the vertical (z) direction and the problem is assumed to be axisymmetric. Thus, the spatial coordinates are \( r \) (radial distance) and \( z \) (vertical coordinate), with the azimuthal angle \(\phi\) being irrelevant because of symmetry.

## Device Geometry and Boundary Conditions

- **Domain:**  
  The computational domain is defined in the (r,z) half‐plane with  

  ``r \in [0, r_{\text{max}}] \quad \text{and} \quad z \in [0, l_3]``

  where ``r_{\text{max}}`` is chosen to include the region of interest (typically larger than the hole diameter (d) and ``l_3`` is the height of the top electrode.

- **Electrodes and Potentials:**

  - **Ground (Bottom Electrode):** At ``z = 0, \phi(r, 0) = 0``

  - **Top Electrode:** At ``z = l_3, \phi(r, l_3) = V_0``

  - **Anode and Gate:**  

    - At intermediate levels ``z = l_1`` (anode) and ``z = l_2`` (gate) the device has a small circular opening (a hole) of diameter d.  

    - For ``r \le d/2`` (inside the hole), the potential is forced to a prescribed value:
      - At ``z = l_1,  \phi(r, l_1) = V_a`` (anode potential).
      - At ``z = l_2, \phi(r, l_2) = V_g`` (gate potential).  
    - For ``r > d/2`` (outside the hole), the potential is determined by the Laplace equation.

- **Boundary Conditions in the Radial Direction:**  
  - At ``r = 0``, symmetry imposes a Neumann condition:  
    ``\partial_r \phi(0,z) = 0.``
  - At ``r = r_{\text{max}}``, a Neumann condition (zero normal derivative) is imposed as well.

## Governing Equation and Numerical Method

The potential ``\phi(r,z)`` in the charge-free region satisfies the axisymmetric Laplace equation:
``
\frac{1}{r} \frac{\partial}{\partial r}\left( r \frac{\partial \phi}{\partial r} \right) + \frac{\partial^2 \phi}{\partial z^2} = 0.
``
This equation is discretized on a grid in ``r`` and ``z`` using finite differences. In the interior of the domain, central differences are used for the second derivatives, and at the boundaries (for both ``r`` and ``z`` appropriate one-sided differences or symmetry conditions are imposed.

### Numerical Algorithm

1. **Grid Generation:**  
   Create uniform grids in r (from 0 to ``r_{\text{max}}`` and in z (from 0 to ``l_3`` ).

2. **Assembly of the Finite Difference Equations:**  
   - For interior points, approximate the radial and vertical derivatives using central differences.
   - At ``r=0``, enforce the symmetry condition by setting ``\phi(0,z)=\phi(dr,z)``.
   - At ``r=r_{\text{max}}``, enforce a Neumann condition (e.g. ``\phi(r_{\text{max}},z)=\phi(r_{\text{max}}-dr,z)``.
   - Impose Dirichlet conditions on the z boundaries:
     - At z=0 and ``z=l_3``, assign the prescribed potentials.
     - At ``z=l_1`` and ``z=l_2``, assign the electrode potentials only for ``r \le d/2``.

3. **Solve the Linear System:**  
   The discretization leads to a sparse linear system ``A \phi = b`` that is solved (using a direct solver) to obtain the potential distribution ``\phi``.

Once the potential field is computed, one can later compute the electric field as
``
\mathbf{E} = -\nabla \phi,
``
but for now we focus solely on solving for the potential.

---
"""


# ╔═╡ eec95346-2dc6-4b60-8b3a-7c70e7ed49db
md"""
# Computing the Electric Field for the GALA Structure

Once the potential φ(r,z) has been computed (using, for example, the `solve_laplace_gala` function), the electric field **E** is obtained as the negative gradient of φ:

``
\mathbf{E}(r,z) = -\nabla \phi(r,z).
``

For an axisymmetric (cylindrical) problem in the (r,z) plane, the components of the electric field are:

- **Radial Component:**  
  ``
  E_r(r,z) = -\frac{\partial \phi(r,z)}{\partial r}
  ``
- **Vertical Component:**  
 ``
  E_z(r,z) = -\frac{\partial \phi(r,z)}{\partial z}
 ``

### Algorithm
1. **Input:**  
   - The potential matrix `phi_mat` of size (Nr × Nz)  
   - The grid vectors `rgrid` and `zgrid` (with uniform spacings `dr` and `dz`).

2. **Finite Difference Approximation:**  
   - **For the r-derivative:**  
     - At interior points (i = 2 to Nr-1), approximate using a central difference:
      ``
       \frac{\partial \phi}{\partial r}(r_i,z_j) \approx \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2\,dr}
      ``
     - At the left boundary (r = 0, i = 1), use a forward difference:
       ``
       \frac{\partial \phi}{\partial r}(r_1,z_j) \approx \frac{\phi_{2,j} - \phi_{1,j}}{dr}
       ``
       (By symmetry, this should be near zero.)
     - At the right boundary (r = rmax, i = Nr), use a backward difference:
       ``
       \frac{\partial \phi}{\partial r}(r_{Nr},z_j) \approx \frac{\phi_{Nr,j} - \phi_{Nr-1,j}}{dr}
       ``
   - **For the z-derivative:**  
     - At interior points (j = 2 to Nz-1), approximate using a central difference:
       ``
       \frac{\partial \phi}{\partial z}(r_i,z_j) \approx \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2\,dz}
       ``
     - At the bottom boundary (z = 0, j = 1), use a forward difference:
       ``
       \frac{\partial \phi}{\partial z}(r_i,z_1) \approx \frac{\phi_{i,2} - \phi_{i,1}}{dz}
       ``
     - At the top boundary (z = l3, j = Nz), use a backward difference:
       ``
       \frac{\partial \phi}{\partial z}(r_i,z_{Nz}) \approx \frac{\phi_{i,Nz} - \phi_{i,Nz-1}}{dz}
       ``

3. **Compute the Electric Field:**  
   - ``E_r(r_i,z_j) = -`` (computed r-derivative)
   - ``E_z(r_i,z_j) = -`` (computed z-derivative)

4. **Output:**  
   - Two matrices `E_r` and `E_z` representing the radial and vertical components of the electric field.

This approach uses standard finite differences and handles boundaries with forward or backward differences.
"""

# ╔═╡ 80b301b1-01e7-4de3-8f7e-a3dcd0033155
md"""
# Electron Transport Function: Boundary Check Explanation

When transporting electrons along field lines, the interpolation objects (built with `interpolate`) expect evaluation points that lie within the grid domain. If an electron’s position goes outside the grid bounds, a BoundsError is thrown. To avoid this, we add a boundary check inside the integration loop. If the electron’s position falls outside the ranges of `rgrid` or `zgrid`, we break out of the loop.

The algorithm is as follows:
1. Start electrons at uniformly spaced r‑positions on the top electrode (z = l₃).
2. At each integration step:
   - Check that the current position is within [min(rgrid), max(rgrid)] and [min(zgrid), max(zgrid)].
   - If not, break the loop.
   - Otherwise, compute the interpolated electric field and update the position.
3. Stop when the electron reaches the anode (z ≤ l₁) or leaves the domain.
"""

# ╔═╡ 95f7fdba-9f68-4d64-a6ef-8f21be265e74


# ╔═╡ 8f8f694b-974c-4f9b-a8d2-df1d73114127
PlutoUI.TableOfContents(title="", indent=true)

# ╔═╡ 99c41eb9-4446-4d39-808e-6bb1e24fa1c8
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




# ╔═╡ 888415cd-7989-40ee-951c-8b08ca232238
begin
	cd("/Users/jjgomezcadenas/Projects/JNEXT/nb")
	jn = ingredients("../src/JNEXT.jl")
end


# ╔═╡ 81ab992f-607c-45bc-9aa5-ad709e6a4b79


# ╔═╡ ba8d6008-a346-4823-b502-d02ab15df560
begin

	d = 2.5      # hole diameter in mm 
	rmax = 5.0    # simulation domain in r: 
	l1 = 2.0    # anode position
	l2 = 7.0    # gate position
	l3 = 8.0    # top electrode position
	Va = 5e+6    # anode potential (V) : V scaled up so that V/m makes sense
	#Va = 0.0    # anode potential (V) : V scaled up so that V/m makes sense
	Vg = -12e+6  # gate potential (V)
	V0 = -12.5e+6   # top electrode potential (V)
	
	
	gparams = jn.JNEXT.GALAParams(d, rmax, l1, l2, l3, Va, Vg, V0)
		#(A_mat, b_vec, rgrid, zgrid), (phi_mat, rgrid, zgrid)
	
	Nr=301 
	Nz=301
end


# ╔═╡ 4100138d-3fc1-4af3-927a-78cb8530df7a
begin 
	la = 0.0
	if Va > 0
		la = l1
	end
end

# ╔═╡ 89affbbe-8eed-47ff-a245-7b257b961230

t = @elapsed begin
    #phi_mat, rgrid, zgrid = solve_laplace_gala(params; Nr=51, Nz=101)
	#AA, PHI = jn.JNEXT.solve_laplace_gala(gparams; Nr=Nr, Nz=Nz, returnMatrix=false)
	phi_mat, rgrid, zgrid = jn.JNEXT.solve_laplace_gala(gparams; Nr=Nr, Nz=Nz, returnMatrix=false)
end


# ╔═╡ 57614361-da39-4cfc-8615-cac89129b5e1
println("Nr = $(Nr), Nz = $(Nz): Elapsed time (Julia): ", t, " seconds")

# ╔═╡ 8b7364b5-a1b0-4197-b7a9-91506bd277ae
println("Froebius norm of phi_mat = $(norm(phi_mat))")

# ╔═╡ f4a35a52-e574-490b-9680-c8b3736eb1b6
#AJ = Matrix(AA[1])

# ╔═╡ c1761c02-6162-4450-b61e-c25de171ffb3
# For inspection, print the dense matrix (or a norm)
#println("Frobenius norm of Julia matrix: ", norm(AJ))

# ╔═╡ 53c4fdca-1646-46fa-87d8-d7c07c5d2f30
contour(rgrid, zgrid, phi_mat, xlabel="r [m]", ylabel="z [m]", 
        title="Potential Contour Plot", fill=true, color=:viridis, size=(1000,600))

# ╔═╡ a816c4b5-50bf-4fb6-b59c-cac2cf142f08
contour(rgrid, zgrid, phi_mat; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Potential Contour Plot",
        size=(800,600))

# ╔═╡ 74282ce2-74cc-4084-8533-70c230f04455
# Example usage (assuming you have already computed phi_mat, rgrid, zgrid):
# (phi_mat, rgrid, zgrid) = solve_laplace_gala(params; Nr=51, Nz=101)
t2 = @elapsed begin
	E_r, E_z = jn.JNEXT.compute_E_field(phi_mat, rgrid, zgrid)
end



# ╔═╡ 498c3fac-e5a8-444e-8b85-453559103d30
println("norm(Er) = $(norm(E_r)), norm(Ez) = $(norm(E_z)): Elapsed time (Julia): ", t2, " seconds")

# ╔═╡ 70aca53d-d8f0-4d03-a021-cc4721268422
# Create a quiver plot (for a subset of points if the grid is dense)
begin

	step = 1  # Plot every 5th point to avoid clutter
	r_sub = rgrid[1:step:end]
	z_sub = zgrid[1:step:end]
	E_r_sub = E_r[1:step:end, 1:step:end]
	E_z_sub = E_z[1:step:end, 1:step:end]
	# Create meshgrid for the subset (ensure matching dimensions)
	R_sub = [r for r in r_sub, _ in z_sub]
	Z_sub = [z for _ in r_sub, z in z_sub]
	quiver(R_sub, Z_sub, quiver=(E_r_sub, E_z_sub), xlabel="r [m]", ylabel="z [m]",
	       title="Electric Field Quiver Plot", legend=false)
end

# ╔═╡ b1d2abe7-094f-4e7a-a186-6864423763ce
contour(rgrid, zgrid, E_r; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Potential Contour Plot",
        size=(400,400))

# ╔═╡ 906d2667-031f-4ecd-b3a4-e736da3ef146
contour(rgrid, zgrid, E_z; levels=20, c=:viridis,
        xlabel="r [m]", ylabel="z [m]",
        title="GALA Potential Contour Plot",
        size=(400,400))

# ╔═╡ bcff425c-ba90-4d4a-9770-06e17790f14e
begin
ftrj, btrj = jn.JNEXT.transport_electrons(phi_mat, rgrid, zgrid, la; N_electrons=40, ds=1e-1)
end

# ╔═╡ 87a9b642-5415-4e67-954c-77d6c31efd47
btrj

# ╔═╡ e5eeb5ca-7e87-477a-8f2b-77ba862e3f6c
plot_electron_trajectories(ftrj, btrj, params=gparams)

# ╔═╡ 4dbb9707-a9e5-4ed5-bb25-0969cc716623
gparams

# ╔═╡ 83fcf651-777c-454c-b817-baecf2ff9cb0


# ╔═╡ 36f1421e-b4a4-476e-b8c2-a5a3fe1b63aa
md"""

## Plots 
"""

# ╔═╡ 075b479d-379f-4273-85f8-f681ecc07f69
"""
    plot_electron_trajectories(trajectories)

Plots the electron trajectories on an (r,z) plane.

-Arguments
- trajectories: A vector of trajectories, where each trajectory is a vector of points [r, z].

-Returns
- A plot displaying the trajectories.
"""
function plot_electron_trajectories(ftrj, btrj; params)
    plt = plot(legend=false,
               xlabel="r [m]", ylabel="z [m]",
               title="Electron Trajectories",
               aspect_ratio=:equal,
               xlims=(0.0, params.rmax), ylims=(0, params.l3))


	# Add a vertical line at x = 5.
	hline!(plt, [params.l1], lw=2, ls=:dash, color=:red, label="anode")
	hline!(plt, [params.l2], lw=2, ls=:dash, color=:red, label="gate")
	hline!(plt, [params.l3], lw=2, ls=:dash, color=:red, label="end-tpc")

	# Add a horizontal line at y = 0.5.
	vline!(plt, [d], lw=2, ls=:dot, color=:red, label="diameter hole")
    
    for traj in ftrj
        # Extract r and z coordinates from each trajectory.
        r_traj = [p[1] for p in traj]
        z_traj = [p[2] for p in traj]
        plot!(plt, r_traj, z_traj, lw=1, color=:blue)
    end

	for traj in btrj
        # Extract r and z coordinates from each trajectory.
        r_traj = [p[1] for p in traj]
        z_traj = [p[2] for p in traj]
        plot!(plt, r_traj, z_traj, lw=1, color=:green)
    end
    
    return plt
end


# ╔═╡ 42c99bde-ff81-4aad-a9d3-3c26e54d8c65


# ╔═╡ Cell order:
# ╠═3766fec6-e89f-11ef-1716-1f64cc6c7d76
# ╟─eec95346-2dc6-4b60-8b3a-7c70e7ed49db
# ╟─80b301b1-01e7-4de3-8f7e-a3dcd0033155
# ╠═95f7fdba-9f68-4d64-a6ef-8f21be265e74
# ╠═8f8f694b-974c-4f9b-a8d2-df1d73114127
# ╠═3e0354b1-fef0-4e17-8b14-8e128f5b55af
# ╠═675394ae-bfec-412c-b408-5bbc83a500f3
# ╠═99c41eb9-4446-4d39-808e-6bb1e24fa1c8
# ╠═888415cd-7989-40ee-951c-8b08ca232238
# ╠═81ab992f-607c-45bc-9aa5-ad709e6a4b79
# ╠═ba8d6008-a346-4823-b502-d02ab15df560
# ╠═4100138d-3fc1-4af3-927a-78cb8530df7a
# ╠═89affbbe-8eed-47ff-a245-7b257b961230
# ╠═57614361-da39-4cfc-8615-cac89129b5e1
# ╠═8b7364b5-a1b0-4197-b7a9-91506bd277ae
# ╠═f4a35a52-e574-490b-9680-c8b3736eb1b6
# ╠═c1761c02-6162-4450-b61e-c25de171ffb3
# ╠═53c4fdca-1646-46fa-87d8-d7c07c5d2f30
# ╠═a816c4b5-50bf-4fb6-b59c-cac2cf142f08
# ╠═74282ce2-74cc-4084-8533-70c230f04455
# ╠═498c3fac-e5a8-444e-8b85-453559103d30
# ╠═70aca53d-d8f0-4d03-a021-cc4721268422
# ╠═b1d2abe7-094f-4e7a-a186-6864423763ce
# ╠═906d2667-031f-4ecd-b3a4-e736da3ef146
# ╠═bcff425c-ba90-4d4a-9770-06e17790f14e
# ╠═87a9b642-5415-4e67-954c-77d6c31efd47
# ╠═e5eeb5ca-7e87-477a-8f2b-77ba862e3f6c
# ╠═4dbb9707-a9e5-4ed5-bb25-0969cc716623
# ╠═83fcf651-777c-454c-b817-baecf2ff9cb0
# ╠═36f1421e-b4a4-476e-b8c2-a5a3fe1b63aa
# ╠═075b479d-379f-4273-85f8-f681ecc07f69
# ╠═42c99bde-ff81-4aad-a9d3-3c26e54d8c65
