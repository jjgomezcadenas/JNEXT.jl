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

"""function to find the grid index closest to a given z value"""
    function find_z_index(zval, zgrid)
        diffs = abs.(zgrid .- zval)
        return argmin(diffs)
    end

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
