
using SparseArrays
using LinearAlgebra
#using Plots
using Interpolations
using Statistics
using StatsBase
using StatsPlots
using Distributions
using Interpolations


    
    """ GALA parameters struct """
    struct GALAParams
        d::Float64      # hole diameter
        rmax::Float64   # maximum radial extent of the simulation domain (choose rmax ≥ d)
        l1::Float64     # z-position of anode (potential Va)
        l2::Float64     # z-position of gate (potential Vg)
        l3::Float64     # z-position of top conductor (potential V0)
        Va::Float64     # potential of anode
        Vg::Float64     # potential of gate
        V0::Float64     # potential of top electrode
    end
    
    
    """ 
    Function to find the grid index closest to a given z value.
    """
    @inline function find_z_index(zval, zgrid)
        diffs = abs.(zgrid .- zval)
        return argmin(diffs)
    end

   

    """ Mapping: index(i,j) = (j-1)*Nr + i, for i = 1:Nr, j = 1:Nz."""

    function index(i::Int, j::Int, Nr::Int) 
        (j - 1) * Nr + i
    end
    
    """
        solve_laplace_gala(params::GALAParams; Nr=51, Nz=101)
    
    
    This equation is discretized using central differences.
    Returns: (phi_mat, rgrid, zgrid) where phi_mat is an Nr×Nz matrix.
    """
    function solve_laplace_galax(params::GALAParams; Nr::Int=51, Nz::Int=101, returnMatrix=false)

        function assemble_matrix(params::GALAParams, Nr::Int, N::Int)

            # Define the domain in r and z.
            rmin = 0.0
            rmax = params.rmax
            zmin = 0.0
            zmax = params.l3

            dr = (rmax - rmin) / (Nr - 1)
            dz = (zmax - zmin) / (Nz - 1)
            rgrid = collect(range(rmin, rmax, length=Nr))
            zgrid = collect(range(zmin, zmax, length=Nz))

            Ntot = Nr * Nz
            A_mat = spzeros(Float64, Ntot, Ntot)
            b_vec = zeros(Float64, Ntot)
            
            # Find the grid indices for the electrode planes.
            i_z0  = find_z_index(0.0,   zgrid)
            i_zl1 = find_z_index(params.l1, zgrid)
            i_zl2 = find_z_index(params.l2, zgrid)
            i_zl3 = find_z_index(params.l3, zgrid)

            #println("indexes: i_z0=$(i_z0)  i_zl1 =$(i_zl1) i_zl2 =$(i_zl2) i_zl3 =$(i_zl3)")
            # Loop over grid points and assemble the discretized Laplace equation.
            for j in 1:Nz
                for i in 1:Nr
                    idx = index(i, j, Nr)
                    r_val = rgrid[i]

                    # Fixed potentials on z=0 and z=l3:
                    if j == i_z0
                        A_mat[idx, idx] = 1.0
                        b_vec[idx] = 0.0
                    elseif j == i_zl3
                        A_mat[idx, idx] = 1.0
                        b_vec[idx] = params.V0

                    # At z = l1 (anode), set φ = Va for r ≤ d/2.
                    elseif (j == i_zl1) && (r_val <= params.d / 2)
                        A_mat[idx, idx] = 1.0
                        b_vec[idx] = params.Va
                    
                    # At z = l2 (gate), set φ = Vg for r ≤ d/2.
                    elseif (j == i_zl2) && (r_val <= params.d / 2)
                        A_mat[idx, idx] = 1.0
                        b_vec[idx] = params.Vg
                    else
                        # Interior nodes (or nodes not constrained by electrode data)
                        # Discretize the z second derivative:
                        # Using central differences: phi(i,j+1) - 2 phi(i,j) + phi(i,j-1)
                        coeff_z = 1 / dz^2
                        A_mat[idx, index(i, j+1, Nr)] += coeff_z
                        A_mat[idx, index(i, j-1, Nr)] += coeff_z
                        A_mat[idx, idx] += -2 * coeff_z

                        # r-direction discretization
                        # At r = 0, enforce symmetry: phi(1,j) = phi(2,j)
                        if i == 1
                            A_mat[idx, index(1, j, Nr)] = 1.0
                            A_mat[idx, index(2, j, Nr)] = -1.0
                        
                        # At r = rmax, impose Neumann: phi(Nr-1,j) = phi(Nr,j)
                        elseif i == Nr
                            A_mat[idx, index(Nr, j, Nr)] = 1.0
                            A_mat[idx, index(Nr-1, j, Nr)] = -1.0
                        else

                            # For interior r points:
                            # Second derivative in r:
                            A_mat[idx, index(i+1, j, Nr)] += 1 / dr^2
                            A_mat[idx, index(i-1, j,Nr)] += 1 / dr^2
                            A_mat[idx, idx] += -2 / dr^2

                            # First derivative term (phi_r)/r:
                            # Approximate phi_r by central difference: (phi(i+1,j)-phi(i-1,j))/(2dr)
                            # Then add 1/(r)*phi_r contribution:
                            A_mat[idx, index(i+1, j, Nr)] += 1 / (2 * dr * r_val)
                            A_mat[idx, index(i-1, j, Nr)] += -1 / (2 * dr * r_val)
                        end
                    end
                end
            end
            return A_mat, b_vec, rgrid, zgrid
        end 

    
        A_mat, b_vec, rgrid, zgrid = assemble_matrix(params, Nr, Nz)
        
        # Solve the sparse linear system.
        phi_vec = A_mat \ b_vec
        phi_mat = reshape(phi_vec, Nr, Nz)
       
        if returnMatrix
            return (A_mat, b_vec, rgrid, zgrid), (phi_mat, rgrid, zgrid)
        else
            return (phi_mat, rgrid, zgrid)
        end
    end

    
    """
        solve_laplace_gala(params::GALAParams; Nr=51, Nz=101, returnMatrix=false)
    
    Assembles and solves the finite difference discretization of the axisymmetric Laplace equation 
    in the (r,z) domain for the GALA problem with one hole. Returns the potential matrix φ(r,z) 
    and the r- and z-grids. If returnMatrix is true, also returns the assembled matrix and right-hand side.
    """
    function solve_laplace_gala(params::GALAParams; Nr::Int=51, Nz::Int=101, returnMatrix::Bool=false)
        # Domain: r in [0, rmax], z in [0, l3]
        rmin = 0.0
        rmax = params.rmax
        zmin = 0.0
        zmax = params.l3
        dr = (rmax - rmin) / (Nr - 1)
        dz = (zmax - zmin) / (Nz - 1)
        rgrid = collect(range(rmin, rmax, length=Nr))
        zgrid = collect(range(zmin, zmax, length=Nz))
        Ntot = Nr * Nz
    
        # Preallocate lists for COO assembly.
        rows = Vector{Int}()
        cols = Vector{Int}()
        data = Vector{Float64}()
        b_vec = zeros(Float64, Ntot)
    
        # Local index function (using 1-indexing).
        @inline index(i::Int, j::Int) = (j - 1) * Nr + i
    
        # Precompute electrode indices.
        i_z0  = find_z_index(0.0, zgrid)
        i_zl1 = find_z_index(params.l1, zgrid)
        i_zl2 = find_z_index(params.l2, zgrid)
        i_zl3 = find_z_index(params.l3, zgrid)
    
        @inbounds for j in 1:Nz
            for i in 1:Nr
                idx = index(i,j)
                r_val = rgrid[i]
                if j == i_z0
                    # Bottom boundary: φ = 0.
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = 0.0
                elseif j == i_zl3
                    # Top boundary: φ = V0.
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = params.V0
                elseif (j == i_zl1) && (r_val <= params.d/2)
                    # Anode (z=l1): φ = Va for r ≤ d/2.
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = params.Va
                elseif (j == i_zl2) && (r_val <= params.d/2)
                    # Gate (z=l2): φ = Vg for r ≤ d/2.
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = params.Vg
                else
                    # Interior node: add contributions from z-direction.
                    local coeff_z = 1/dz^2
                    push!(rows, idx); push!(cols, index(i, j+1)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, index(i, j-1)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, idx); push!(data, -2*coeff_z)
                    
                    # r-direction:
                    if i == 1
                        # r = 0: symmetry, φ(1,j) = φ(2,j)
                        push!(rows, idx); push!(cols, index(1, j)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(2, j)); push!(data, -1.0)
                    elseif i == Nr
                        # r = rmax: Neumann condition, φ(Nr,j) = φ(Nr-1,j)
                        push!(rows, idx); push!(cols, index(Nr, j)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(Nr-1, j)); push!(data, -1.0)
                    else
                        # Interior r points:
                        push!(rows, idx); push!(cols, index(i+1, j)); push!(data, 1/dr^2)
                        push!(rows, idx); push!(cols, index(i-1, j)); push!(data, 1/dr^2)
                        push!(rows, idx); push!(cols, idx); push!(data, -2/dr^2)
                        # First derivative term divided by r:
                        push!(rows, idx); push!(cols, index(i+1, j)); push!(data, 1/(2*dr*r_val))
                        push!(rows, idx); push!(cols, index(i-1, j)); push!(data, -1/(2*dr*r_val))
                    end
                end
            end
        end
    
        # Assemble the sparse matrix in one step.
        A_mat = sparse(rows, cols, data, Ntot, Ntot)
        # Solve the linear system.
        phi_vec = A_mat \ b_vec
        phi_mat = reshape(phi_vec, Nr, Nz)
        #println(phi_mat)
        return returnMatrix ? ((A_mat, b_vec, rgrid, zgrid), (phi_mat, rgrid, zgrid)) : (phi_mat, rgrid, zgrid)
    end



"""
    compute_E_field(phi_mat, rgrid, zgrid)

Computes the electric field components (E_r and E_z) from the potential field phi_mat
using finite difference approximations.

# Arguments
- phi_mat::Array{Float64,2}: Potential values on an (Nr × Nz) grid.
- rgrid::Vector{Float64}: Radial grid (length Nr).
- zgrid::Vector{Float64}: Vertical grid (length Nz).

# Returns
A tuple (E_r, E_z) of matrices of the same size as phi_mat representing the electric field components.
"""
function compute_E_field(phi_mat::AbstractMatrix{Float64},
                         rgrid::AbstractVector{Float64},
                         zgrid::AbstractVector{Float64})
    Nr, Nz = size(phi_mat)
    dr = rgrid[2] - rgrid[1]
    dz = zgrid[2] - zgrid[1]
    
    E_r = zeros(Float64, Nr, Nz)
    E_z = zeros(Float64, Nr, Nz)
    
    @inbounds for j in 1:Nz
        for i in 1:Nr
            # Compute radial derivative ∂φ/∂r
            if i == 1
                # Forward difference at r = 0.
                E_r[i,j] = -(phi_mat[i+1,j] - phi_mat[i,j]) / dr
            elseif i == Nr
                # Backward difference at r = rmax.
                E_r[i,j] = -(phi_mat[i,j] - phi_mat[i-1,j]) / dr
            else
                # Central difference for interior points.
                E_r[i,j] = -(phi_mat[i+1,j] - phi_mat[i-1,j]) / (2 * dr)
            end
            
            # Compute vertical derivative ∂φ/∂z.
            if j == 1
                # Forward difference at z = 0.
                E_z[i,j] = -(phi_mat[i,j+1] - phi_mat[i,j]) / dz
            elseif j == Nz
                # Backward difference at z = l3.
                E_z[i,j] = -(phi_mat[i,j] - phi_mat[i,j-1]) / dz
            else
                # Central difference for interior points.
                E_z[i,j] = -(phi_mat[i,j+1] - phi_mat[i,j-1]) / (2 * dz)
            end
        end
    end
    
    return E_r, E_z
end

    
"""
    transport_electrons(phi_mat, rgrid, zgrid, l1; N_electrons=10, ds=1e-6)

Transport electrons starting at z = l₃ (top electrode) with uniformly spaced r‑positions.
Electrons follow the field lines (using Euler integration) until they reach the anode (z ≤ l₁)
or leave the computational domain.

Arguments:
  - phi_mat::AbstractMatrix{Float64}: Potential field (size Nr × Nz)
  - rgrid::AbstractVector{Float64}: Radial grid (length Nr)
  - zgrid::AbstractVector{Float64}: Vertical grid (length Nz)
  - l1::Float64: z-coordinate of the anode (stopping condition)
  - N_electrons::Int: Number of electrons to transport (uniform initial r positions)
  - ds::Float64: Euler integration step size

Returns:
  - trajectories: A vector of trajectories; each trajectory is a vector of [r, z] positions.
"""
function transport_electrons(phi_mat::AbstractMatrix{Float64},
                             rgrid::AbstractVector{Float64},
                             zgrid::AbstractVector{Float64},
                             la::Float64; N_electrons::Int=10, ds::Float64=1e-6)
    # Compute the electric field components from the potential.
    E_r, E_z = compute_E_field(phi_mat, rgrid, zgrid)
    
    # Build interpolants for E_r and E_z.
    itp_Er = interpolate((rgrid, zgrid), E_r, Gridded(Linear()))
    itp_Ez = interpolate((rgrid, zgrid), E_z, Gridded(Linear()))
    
    trajectories = Vector{Vector{Vector{Float64}}}()
    trbk = Vector{Vector{Vector{Float64}}}()
    
    # Generate uniformly spaced initial positions in r at the top electrode (z = l3).
    r_initial = collect(range(first(rgrid), stop=last(rgrid), length=N_electrons))
    
    for r0 in r_initial
        pos = [r0, zgrid[end]]  # starting position: (r0, l₃)
        traj = [copy(pos)]
        # Transport electron until it reaches the anode (z <= la)
        while pos[2] > la
            # Check if position is within the grid bounds otherwise we are done.
            if pos[1] < minimum(rgrid) || pos[1] > maximum(rgrid) ||
               pos[2] < minimum(zgrid) || pos[2] > maximum(zgrid)
                break
            end
            # Evaluate the electric field at the current position.
            Er_val = itp_Er(pos[1], pos[2])
            Ez_val = itp_Ez(pos[1], pos[2])
            normE = sqrt(Er_val^2 + Ez_val^2)
            if normE < 1e-12
                break  # Avoid division by zero.
            end
            # Electrons are negatively charged, so they move opposite to E.
            drds = -Er_val / normE
            dzds = -Ez_val / normE
            pos[1] += ds * drds
            pos[2] += ds * dzds
            push!(traj, copy(pos))
        end
        push!(trajectories, traj)
    end

    # if the anode is at positive voltage, la>0 by construction
        # in this case, trajectories will move toward ground and then turn around.
    if la > 0 
        for trj in trajectories
            # start with the end of the previous trajectory
            pos = copy(trj[end])
            #println("start back tracking: pos =$(pos)")
            trj2 = [copy(pos)]
            
            while pos[2] < la # Transport electron until it reaches BACK the anode (z >= la)
                # Check if position is within the grid bounds otherwise we are done.
                if pos[1] < minimum(rgrid) || pos[1] > maximum(rgrid) ||
                    pos[2] < minimum(zgrid) || pos[2] > maximum(zgrid)
                    break
                end
                # Evaluate the electric field at the current position.
                Er_val = itp_Er(pos[1], pos[2])
                Ez_val = itp_Ez(pos[1], pos[2])
                normE = sqrt(Er_val^2 + Ez_val^2)
                if normE < 1e-12
                    break  # Avoid division by zero.
                end
                # Electrons are negatively charged, so they move opposite to E.
                drds = -Er_val / normE
                dzds = -Ez_val / normE
                pos[1] += ds * drds
                pos[2] += ds * dzds
                #println("step: pos =$(pos)")
                push!(trj2, copy(pos))
            end
            #println("trj2:  =$(trj2)")
            push!(trbk, trj2)
        end
    end
    #println("trbk:  =$(trbk)")

    if la > 0 
        return trajectories, trbk
    else
        return trajectories
    end
end