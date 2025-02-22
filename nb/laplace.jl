### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 2a6b84b4-e730-11ef-0930-255b8575d196
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

# ╔═╡ e699b1e6-ef3b-430a-865e-40977e134323
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

# ╔═╡ 2011a9c4-13f8-41bf-b2aa-93657000fb88
""" GALA parameters struct"""
struct GALAParams
    d::Float64      # hole diameter
    rmax::Float64   # maximum radial extent of the simulation domain (e.g., rmax = d or larger)
    l1::Float64     # z-position of anode (potential Va)
    l2::Float64     # z-position of gate (potential Vg)
    l3::Float64     # z-position of top conductor (potential V0)
    Va::Float64     # potential of anode
    Vg::Float64     # potential of gate
    V0::Float64     # potential of top electrode
end

# ╔═╡ 2b388872-f0a1-4ef5-bd3f-8b6a83dde773
"""function to find the grid index closest to a given z value"""
    function find_z_index(zval, zgrid)
        diffs = abs.(zgrid .- zval)
        return argmin(diffs)
    end

# ╔═╡ 10d5f968-b409-4897-aba7-8bdcb2928cca
""" Helper: find the index of a grid vector zgrid corresponding to a given value."""
function find_z_index2(z_val, zgrid; tol=1e-6)
    for (j, z) in enumerate(zgrid)
        if isapprox(z, z_val; atol=tol)
            return j
        end
    end
    error("z value $z_val not found in zgrid")
end

# ╔═╡ 7cb4c032-be03-4425-a2c5-87aaa0d90104
"""
    solve_laplace_gala(params::GALAParams; Nr=51, Nz=101)

Solve Laplace’s equation in the (r,z) domain:
  r ∈ [0, params.rmax]   (with Neumann BC at r=0 and r=params.rmax)
  z ∈ [0, params.l3]

The boundary conditions are:
  - At z = 0: φ = 0 (ground) for all r.
  - At z = l3: φ = V0 for all r.
  - At z = l1 (anode): impose φ = Va only for r ≤ d/2, 
      while for r > d/2 the equation is solved (modeling the dielectric-covered region).
  - At z = l2 (gate): impose φ = Vg only for r ≤ d/2, 
      while for r > d/2 the equation is solved.
  
The axisymmetric Laplace equation

    (1/r)(r φ_r)_r + φ_zz = 0

is discretized using central differences. At r = 0 the Neumann condition φ_r=0 is imposed by symmetry.

Returns (phi_mat, rgrid, zgrid) where phi_mat is an Nr×Nz matrix.
"""
function solve_laplace_gala(params::GALAParams; Nr=51, Nz=101)
    # Domain in r and z.
    rmin = 0.0
    rmax = params.rmax    # Choose params.rmax ≥ d so that the dielectric region is included.
    zmin = 0.0
    zmax = params.l3

    dr = (rmax - rmin)/(Nr - 1)
    dz = (zmax - zmin)/(Nz - 1)
    rgrid = collect(range(rmin, rmax, length=Nr))
    zgrid = collect(range(zmin, zmax, length=Nz))
    
    Ntot = Nr * Nz
    A_mat = spzeros(Float64, Ntot, Ntot)
    b_vec = zeros(Float64, Ntot)
    
    # mapping: index(i,j) = (j-1)*Nr + i, with i = 1:Nr, j = 1:Nz.
    index(i,j) = (j - 1)*Nr + i

    # Find the z indices corresponding to the electrode planes.
    i_z0  = find_z_index(0.0,   zgrid)
    i_zl1 = find_z_index(params.l1, zgrid)
    i_zl2 = find_z_index(params.l2, zgrid)
    i_zl3 = find_z_index(params.l3, zgrid)
    
    # Loop over grid points.
    for j in 1:Nz
        for i in 1:Nr
            idx = index(i, j)
            r_val = rgrid[i]
            z_val = zgrid[j]
            
            # Fixed potentials on the full z=0 and z=l3 planes.
            if j == i_z0
                A_mat[idx, idx] = 1.0
                b_vec[idx] = 0.0
            elseif j == i_zl3
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.V0
            # For the anode (z = l1) and gate (z = l2), impose Dirichlet only inside the hole.
            elseif (j == i_zl1) && (r_val <= params.d/2)
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Va
            elseif (j == i_zl2) && (r_val <= params.d/2)
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Vg
            else
                # For all other nodes, assemble the discretized Laplace operator.
                # In the z–direction (common to all nodes, assuming interior in z):
                #    (φ_{i,j+1} - 2 φ_{i,j} + φ_{i,j-1})/dz^2.
                coeff_z = 1/dz^2
                # Add contributions from the z–neighbors.
                A_mat[idx, index(i, j+1)] += coeff_z
                A_mat[idx, index(i, j-1)] += coeff_z
                A_mat[idx, idx]         += -2*coeff_z
                
                # For the r–direction, we use the standard axisymmetric discretization.
                if i == 1
                    # At r = 0 use symmetry: φ(1,j) = φ(2,j)  →  (φ_1 - φ_2) = 0.
                    A_mat[idx, index(1, j)] = 1.0
                    A_mat[idx, index(2, j)] = -1.0
                elseif i == Nr
                    # At r = rmax, impose a Neumann condition: φ(Nr,j) = φ(Nr-1,j).
                    A_mat[idx, index(Nr, j)]     = 1.0
                    A_mat[idx, index(Nr-1, j)]   = -1.0
                else
                    # For interior r points (with r_val > 0):
                    # Discretize:
                    #   φ_rr ≈ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j})/dr^2,
                    #   φ_r   ≈ (φ_{i+1,j} - φ_{i-1,j})/(2dr),
                    # so that
                    #   (1/r)*(φ_r) ≈ (φ_{i+1,j} - φ_{i-1,j})/(2dr*r_val).
                    coeff_r_plus  = 1/dr^2 + 1/(2*dr*r_val)
                    coeff_r_minus = 1/dr^2 - 1/(2*dr*r_val)
                    A_mat[idx, index(i+1, j)] += coeff_r_plus
                    A_mat[idx, index(i-1, j)] += coeff_r_minus
                    A_mat[idx, idx]           += -2/dr^2
                end
            end
        end
    end
    
    # Solve the sparse linear system.
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nr, Nz)
    return (phi_mat, rgrid, zgrid)
end

# ╔═╡ cfc3c94a-275d-4457-a4b2-d926ecb08dbb
"""
    compute_E_field(phi_mat, rgrid, zgrid)

Compute the electric field from a potential matrix φ defined on a grid.
Assume:
  - rgrid is a vector of radial grid points (uniform spacing dr),
  - zgrid is a vector of longitudinal grid points (uniform spacing dz),
  - phi_mat is a matrix of size (Nr × Nz) where the first index corresponds to r and the second to z.
  
We use central differences in the interior and one-sided differences at the boundaries.
Returns two matrices, E_r and E_z, each of size (Nr × Nz).
"""
function compute_E_field(phi_mat::AbstractMatrix, rgrid::AbstractVector, zgrid::AbstractVector)
    Nr, Nz = size(phi_mat)
    dr = rgrid[2] - rgrid[1]
    dz = zgrid[2] - zgrid[1]
    E_r = zeros(Nr, Nz)
    E_z = zeros(Nr, Nz)
    
    # Loop over grid points
    for j in 1:Nz
        for i in 1:Nr
            # --- radial derivative: ∂φ/∂r ---
            if i == 1
                # At r = 0: use forward difference
                E_r[i,j] = - (phi_mat[i+1,j] - phi_mat[i,j]) / dr
            elseif i == Nr
                # At r = rmax: backward difference
                E_r[i,j] = - (phi_mat[i,j] - phi_mat[i-1,j]) / dr
            else
                # Central difference
                E_r[i,j] = - (phi_mat[i+1,j] - phi_mat[i-1,j]) / (2*dr)
            end
            
            # --- longitudinal derivative: ∂φ/∂z ---
            if j == 1
                # At z = zmin: forward difference
                E_z[i,j] = - (phi_mat[i,j+1] - phi_mat[i,j]) / dz
            elseif j == Nz
                # At z = zmax: backward difference
                E_z[i,j] = - (phi_mat[i,j] - phi_mat[i,j-1]) / dz
            else
                # Central difference
                E_z[i,j] = - (phi_mat[i,j+1] - phi_mat[i,j-1]) / (2*dz)
            end
        end
    end
    return E_r, E_z
end

# ╔═╡ 0991d40d-0a3e-41c8-973e-677b8b026fdd




# ╔═╡ 7516a730-dd72-44a0-9508-3597ced3782c
"""
    electron_trajectory(r0, z0, interp_Er, interp_Ez, rgrid, zgrid; ds=1e-6, params, rmax, max_steps=1000000)

Integrate a single electron trajectory following the electric field.

The electron is assumed to move in the direction of –E (since electrons are negatively charged). The
integration stops when either:
  - The electron reaches (or passes) the anode plane (z ≤ params.l1),
  - The electron leaves the (r,z) grid domain, or
  - The number of integration steps exceeds `max_steps`.

# Arguments
- `r0, z0`: Initial coordinates of the electron (typically, z0 = l3).
- `interp_Er, interp_Ez`: Interpolation objects for the electric field components.
- `rgrid, zgrid`: The coordinate vectors defining the (r,z) grid.
- `ds`: The integration step size (default is 1e-6).
- `params`: A GALAParams structure containing the electrode positions (l1, l3, etc.).
- `rmax`: Maximum radial coordinate (should correspond to last(rgrid)).
- `max_steps`: Maximum number of integration steps (default is 1e6).

# Returns
A tuple `(r_traj, z_traj)` containing the arrays of r and z positions along the trajectory.
"""
function electron_trajectory(r0, z0, interp_Er, interp_Ez, rgrid, zgrid; ds=1e-6, params, rmax, max_steps=1000000)
    r = r0
    z = z0
    r_traj = [r]
    z_traj = [z]

    for step in 1:max_steps
        # Check if the current position is within the grid bounds.
        if r < first(rgrid) || r > last(rgrid) || z < first(zgrid) || z > last(zgrid)
            break
        end

        # Evaluate the local electric field via the (possibly extrapolated) interpolants.
        Er = interp_Er(r, z)
        Ez = interp_Ez(r, z)
        normE = sqrt(Er^2 + Ez^2)
        if normE < 1e-12
            # Field is too small to define a direction.
            break
        end

        # Electrons move opposite to the electric field.
        drds = -Er / normE
        dzds = -Ez / normE

        # Take one Euler integration step.
        r_new = r + ds * drds
        z_new = z + ds * dzds

        # Check if the new point is still within the grid.
        if r_new < first(rgrid) || r_new > last(rgrid) || z_new < first(zgrid) || z_new > last(zgrid)
            break
        end

        push!(r_traj, r_new)
        push!(z_traj, z_new)
        r, z = r_new, z_new

        # Terminate if the electron has reached the anode plane.
        if z <= params.l1
            break
        end
    end

    return r_traj, z_traj
end



# ╔═╡ 199b2d71-3352-451d-b647-7f9c7d742441
"""
    compute_electron_trajectories(phi_mat, rgrid, zgrid, params; N_electrons=10, ds=1e-6, max_steps=1000000)

Compute trajectories for multiple electrons injected at the top electrode (z = l3).

Electrons are injected uniformly in r ∈ [0, rmax] at z = l3 and then integrated along the
electric field lines using `electron_trajectory`.

# Arguments
- `phi_mat`: Matrix of the potential (computed from solve_laplace_gala).
- `rgrid, zgrid`: Vectors defining the spatial grid.
- `params`: A GALAParams structure containing electrode positions and potentials.
- `N_electrons`: Number of electrons to simulate (default is 10).
- `ds`: Integration step size (default is 1e-6).
- `max_steps`: Maximum number of integration steps for each electron (default is 1e6).

# Returns
A vector of trajectories, where each trajectory is a tuple `(r_traj, z_traj)`.
"""
function compute_electron_trajectories(phi_mat, rgrid, zgrid, params; N_electrons=10, ds=1e-6, max_steps=1000000)
    # Compute the electric field on the grid.
    E_r, E_z = compute_E_field(phi_mat, rgrid, zgrid)
    
    # Create interpolation objects for the electric field and wrap with extrapolation.
    interp_Er = extrapolate(interpolate((rgrid, zgrid), E_r, Gridded(Linear())), Flat())
    interp_Ez = extrapolate(interpolate((rgrid, zgrid), E_z, Gridded(Linear())), Flat())
    
    trajectories = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    rmax_grid = last(rgrid)  # Maximum radial coordinate from the grid.
    
    # Inject electrons at z = l3 with r uniformly in [0, rmax_grid].
    for n in 1:N_electrons
        r0 = rand() * rmax_grid
        z0 = last(zgrid)  # Should be equal to params.l3.
        traj = electron_trajectory(r0, z0, interp_Er, interp_Ez, rgrid, zgrid;
                                   ds=ds, params=params, rmax=rmax_grid, max_steps=max_steps)
        push!(trajectories, traj)
    end
    
    return trajectories
end



# ╔═╡ e5b02c54-8c5a-46fb-b4fe-8c41a293d2cc
"""
    plot_phi_3D(phi_mat, rgrid, zgrid)

Plot a 3D surface of φ(r,z).
"""
function plot_phi_3D(phi_mat, rgrid, zgrid)
    surface(rgrid, zgrid, phi_mat',
        xlabel="r", ylabel="z", zlabel="φ(r,z)",
        title="3D Surface of φ(r,z)")
end



# ╔═╡ 81e7087f-c403-4e52-ae21-215a9d654ea0
"""
    plot_phi_contour(phi_mat, rgrid, zgrid)

Plot a contour (filled) of φ(r,z) in the (r,z) plane.
"""
function plot_phi_contour(phi_mat, rgrid, zgrid)
    contour(rgrid, zgrid, phi_mat',
        xlabel="r", ylabel="z", title="Contour of φ(r,z)", fill=true, c=:viridis)
end

# ╔═╡ c91d7363-7081-4282-896c-39837ddac50e
function test_phi_boundaries(params)
    @testset "φ(r,z) Boundary Conditions" begin
        (phi_mat, rgrid, zgrid) = solve_laplace_gala(params; Nr=51, Nz=101)
        
        i0 = find_z_index(0, zgrid)
        il1 = find_z_index(params.l1, zgrid)
        il2 = find_z_index(params.l2, zgrid)
        il3 = find_z_index(params.l3, zgrid)
        
        for i in 1:length(rgrid)
            @test isapprox(phi_mat[i, i0], 0.0; atol=1e-5)
            @test isapprox(phi_mat[i, il1], params.Va; atol=1e-5)
            @test isapprox(phi_mat[i, il2], params.Vg; atol=1e-5)
            @test isapprox(phi_mat[i, il3], params.V0; atol=1e-5)
        end
    end
end

# ╔═╡ 06051d9d-b188-4f33-9a94-ac94c27e35f8
"""
    test_E_field()

Run tests on the computed electric field.
In our geometry (axial symmetry) we expect that at r=0 the radial field vanishes.
"""
function test_E_field(params)
    
    # Here we call the Laplace solver to get phi(r,z) on the grid.
    # (Replace solve_laplace_gala with your actual solver function if necessary.)
    (phi_mat, rgrid, zgrid) = solve_laplace_gala(params; Nr=51, Nz=101)
    E_r, E_z = compute_E_field(phi_mat, rgrid, zgrid)
    
    # Test: For all z, at r=0 the radial derivative should be near zero.
    for j in 1:length(zgrid)
        @test isapprox(E_r[1,j], 0.0; atol=1e-5)
    end
    println("Electric field tests passed: E_r(r=0, z) ≈ 0 for all z.")
end

# ╔═╡ 1d40d40d-8c45-4925-84d3-4a1b28f03f89
"""
    plot_E_field_quiver(phi_mat, rgrid, zgrid; skip_r=2, skip_z=2, scale=0.1)

Compute the electric field from phi\\_mat and plot it as a quiver (arrow) plot.
- skip_r, skip_z: integer step to skip grid points for clarity.
- scale: a scaling factor for arrow lengths.
"""
function plot_E_field_quiver(phi_mat, rgrid, zgrid; skip_r=2, skip_z=2, scale=0.1)
    E_r, E_z = compute_E_field(phi_mat, rgrid, zgrid)
    
    # Create arrays for the quiver plot. We sample every skip_r-th and skip_z-th point.
    r_points = Float64[]
    z_points = Float64[]
    U = Float64[]   # component in r-direction
    V = Float64[]   # component in z-direction
    
    for j in 1:skip_z:length(zgrid)
        for i in 1:skip_r:length(rgrid)
            push!(r_points, rgrid[i])
            push!(z_points, zgrid[j])
            # scale the field for display purposes
            push!(U, scale*E_r[i,j])
            push!(V, scale*E_z[i,j])
        end
    end
    
    plt = quiver(r_points, z_points, quiver=(U, V),
                 xlabel="r", ylabel="z", title="Electric Field in (r,z) Plane",
                 aspect_ratio=:equal, legend=false)
    #display(plt)
    return plt
end

# ╔═╡ 04cd4641-8629-4386-806c-f379c71430e2
"""
    plot_trajectories(trajectories; params=nothing)

Plot the electron trajectories.

# Arguments
- `trajectories::Vector{Tuple{AbstractVector,AbstractVector}}`: A vector where each element is a tuple `(r_traj, z_traj)` containing the electron's (r,z) positions along its path.
- `params` (optional): A `GALAParams` instance. If provided, horizontal lines will be added to mark the positions of the electrodes (ground at z=0, anode at z = l1, gate at z = l2, and the top electrode at z = l3) and a vertical line is drawn at r = d/2 (the hole boundary).

# Returns
A Plots.jl plot object.
"""
function plot_trajectories(trajectories; params=nothing)
    plt = plot(legend=false, xlabel="r [m]", ylabel="z [m]", title="Electron Trajectories")
    # Plot each trajectory
    for (r_traj, z_traj) in trajectories
        plot!(plt, r_traj, z_traj, lw=2)
    end

    if params !== nothing
        # Mark electrode positions as horizontal dashed lines.
        plot!(plt, [0, params.rmax], [params.l3, params.l3], ls=:dash, lw=1, color=:black, label="Top Electrode")
        plot!(plt, [0, params.rmax], [params.l2, params.l2], ls=:dash, lw=1, color=:gray, label="Gate")
        plot!(plt, [0, params.rmax], [params.l1, params.l1], ls=:dash, lw=1, color=:red, label="Anode")
        plot!(plt, [0, params.rmax], [0, 0],       ls=:dash, lw=1, color=:blue, label="Ground")
        # Mark the hole boundary (r = d/2) as a vertical dotted line.
        vline!(plt, [params.d/2], ls=:dot, lw=1, color=:purple, label="Hole Boundary")
    end

    return plt
end



# ╔═╡ 9ebfa33b-392e-4ab7-acbf-08cb51149c75
"""
    plot_potential_and_trajectories(phi_mat, rgrid, zgrid, trajectories; params=nothing)

Overlay the electron trajectories on a contour plot of the potential field.

# Arguments
- `phi_mat`: A matrix (Nr×Nz) of the computed potential.
- `rgrid`: The vector of r coordinates.
- `zgrid`: The vector of z coordinates.
- `trajectories`: A vector of electron trajectories as produced by `compute_electron_trajectories`.
- `params` (optional): A `GALAParams` instance, which (if provided) is used to mark the electrode positions and hole boundary.

# Returns
A Plots.jl plot object.
"""
function plot_potential_and_trajectories(phi_mat, rgrid, zgrid, trajectories; params=nothing)
    # Create a contour plot of the potential.
    # Note: transpose phi_mat if needed so that the first dimension corresponds to r and the second to z.
    plt = contour(rgrid, zgrid, phi_mat', xlabel="r [m]", ylabel="z [m]",
                  title="Potential Field with Electron Trajectories", color=:viridis)
    # Overlay the trajectories.
    for (r_traj, z_traj) in trajectories
        plot!(plt, r_traj, z_traj, lw=2, color=:white)
    end

    if params !== nothing
        # Mark the electrode planes.
        plot!(plt, [0, params.rmax], [params.l3, params.l3], ls=:dash, lw=1, color=:black, label="Top Electrode")
        plot!(plt, [0, params.rmax], [params.l2, params.l2], ls=:dash, lw=1, color=:gray, label="Gate")
        plot!(plt, [0, params.rmax], [params.l1, params.l1], ls=:dash, lw=1, color=:red, label="Anode")
        plot!(plt, [0, params.rmax], [0, 0],           ls=:dash, lw=1, color=:blue, label="Ground")
        # Mark the hole boundary.
        vline!(plt, [params.d/2], ls=:dot, lw=1, color=:purple, label="Hole Boundary")
    end

    return plt
end

# ╔═╡ 8de129f0-a5c6-45ae-a14d-471e487d83c1
md"""
# Gala Analysis
"""

# ╔═╡ 6188d230-867a-4ea6-adaf-69a09818e2bb
begin
	d    = 5      # 1 mm hole diameter
	rmax = 10      # Let the dielectric region extend to 2 mm
	l1   = 5    # anode at 0.5 mm
	l2   = 10    # gate at 1.5 mm
	l3   = 11    # top conductor at 2.0 mm
	Va   = -100.0    # anode potential (V)
	Vg   = -10000.0  # gate potential (V)
	V0   = -10100.0   # top electrode potential (V)
	gparams = GALAParams(d, rmax, l1, l2, l3, Va, Vg, V0) 
end

# ╔═╡ e5d96e0e-bbf1-40d5-8a2d-a2adbfa9b172
(phi_mat, rgrid, zgrid) = solve_laplace_gala(gparams; Nr=51, Nz=101)

# ╔═╡ 0de0323a-ec60-4e3b-a221-dcf2b4d69af5
plot_phi_3D(phi_mat, rgrid, zgrid)

# ╔═╡ 38146cfa-3c40-4498-a402-373475f72995
plot_phi_contour(phi_mat, rgrid, zgrid)

# ╔═╡ 47c4926c-24a3-405e-919a-81d6b7007280
plot_E_field_quiver(phi_mat, rgrid, zgrid; skip_r=2, skip_z=2, scale=0.1)

# ╔═╡ 6d97a595-59e7-4235-b8ec-764d2e10b679

trajectories = compute_electron_trajectories(phi_mat, rgrid, zgrid, gparams;
                                             N_electrons=200, 
                                             ds=1e-3, max_steps=500000)


# ╔═╡ e153637f-7dad-46db-b157-8fd9b5f54a07
plot_trajectories(trajectories)

# ╔═╡ 23aac60b-1d40-49e1-900d-a3315e5bc9d2
plot_potential_and_trajectories(phi_mat, rgrid, zgrid, trajectories)

# ╔═╡ Cell order:
# ╠═2a6b84b4-e730-11ef-0930-255b8575d196
# ╠═e699b1e6-ef3b-430a-865e-40977e134323
# ╠═2011a9c4-13f8-41bf-b2aa-93657000fb88
# ╠═2b388872-f0a1-4ef5-bd3f-8b6a83dde773
# ╠═10d5f968-b409-4897-aba7-8bdcb2928cca
# ╠═7cb4c032-be03-4425-a2c5-87aaa0d90104
# ╠═cfc3c94a-275d-4457-a4b2-d926ecb08dbb
# ╠═0991d40d-0a3e-41c8-973e-677b8b026fdd
# ╠═7516a730-dd72-44a0-9508-3597ced3782c
# ╠═199b2d71-3352-451d-b647-7f9c7d742441
# ╠═e5b02c54-8c5a-46fb-b4fe-8c41a293d2cc
# ╠═81e7087f-c403-4e52-ae21-215a9d654ea0
# ╠═c91d7363-7081-4282-896c-39837ddac50e
# ╠═06051d9d-b188-4f33-9a94-ac94c27e35f8
# ╠═1d40d40d-8c45-4925-84d3-4a1b28f03f89
# ╠═04cd4641-8629-4386-806c-f379c71430e2
# ╠═9ebfa33b-392e-4ab7-acbf-08cb51149c75
# ╠═8de129f0-a5c6-45ae-a14d-471e487d83c1
# ╠═6188d230-867a-4ea6-adaf-69a09818e2bb
# ╠═e5d96e0e-bbf1-40d5-8a2d-a2adbfa9b172
# ╠═0de0323a-ec60-4e3b-a221-dcf2b4d69af5
# ╠═38146cfa-3c40-4498-a402-373475f72995
# ╠═47c4926c-24a3-405e-919a-81d6b7007280
# ╠═6d97a595-59e7-4235-b8ec-764d2e10b679
# ╠═e153637f-7dad-46db-b157-8fd9b5f54a07
# ╠═23aac60b-1d40-49e1-900d-a3315e5bc9d2
