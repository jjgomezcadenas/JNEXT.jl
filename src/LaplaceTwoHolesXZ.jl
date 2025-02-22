
using SparseArrays, LinearAlgebra, Statistics, Interpolations

#############################
# 1. Compute the Electric Potential
#############################

# Define the new parameters struct for the 2D (x,z) problem with two holes.
struct GALAParams2D
    x1::Float64    # x-position of Hole 1 center
    d1::Float64    # Diameter of Hole 1 (and Hole 2)
    p::Float64     # Pitch between holes (Hole 2 center = x1 + p)
    l1::Float64    # z-position of anode (potential Va)
    l2::Float64    # z-position of gate (potential Vg)
    l3::Float64    # z-position of top electrode (potential V0)
    Va::Float64    # anode potential
    Vg::Float64    # gate potential
    V0::Float64    # top electrode potential
end

# Domain in x: from x_min to x_max.
function x_domain(params::GALAParams2D)
    x_min = params.x1 - params.p/2
    x_max = (params.x1 + params.p) + params.p/2  # since x2 = x1 + p
    return x_min, x_max
end

# Local index mapping (for a grid of size Nx x Nz)
@inline function index(i::Int, j::Int, Nx::Int)
    (j - 1) * Nx + i
end

"""
    solve_laplace_gala_2d(params::GALAParams2D; Nx=51, Nz=101)

Returns: (phi_mat, xgrid, zgrid)
"""
function solve_laplace_gala_2d(params::GALAParams2D; Nx::Int=51, Nz::Int=101)
    # Define domain.
    x_min, x_max = x_domain(params)
    z_min = 0.0
    z_max = params.l3
    dx = (x_max - x_min) / (Nx - 1)
    dz = (z_max - z_min) / (Nz - 1)
    xgrid = collect(range(x_min, x_max, length=Nx))
    zgrid = collect(range(z_min, z_max, length=Nz))
    Ntot = Nx * Nz

    # Preallocate arrays for COO assembly.
    rows = Vector{Int}()
    cols = Vector{Int}()
    data = Vector{Float64}()
    b_vec = zeros(Float64, Ntot)

    # Determine indices for z-boundaries.
    i_z0  = argmin(abs.(zgrid .- 0.0))
    i_zl1 = argmin(abs.(zgrid .- params.l1))
    i_zl2 = argmin(abs.(zgrid .- params.l2))
    i_zl3 = argmin(abs.(zgrid .- params.l3))

    # Define hole regions in x.
    # Hole 1: x in [x1 - d1/2, x1 + d1/2]
    # Hole 2: x in [x2 - d1/2, x2 + d1/2] with x2 = x1 + p.
    x2 = params.x1 + params.p
    hole1 = (params.x1 - params.d1/2, params.x1 + params.d1/2)
    hole2 = (x2 - params.d1/2, x2 + params.d1/2)

    @inbounds for j in 1:Nz
        for i in 1:Nx
            idx = index(i, j, Nx)
            x_val = xgrid[i]
            # For clarity, define current z index.
            if j == i_z0
                # Bottom boundary: φ = 0.
                push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                b_vec[idx] = 0.0
            elseif j == i_zl3
                # Top boundary: φ = V0.
                push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                b_vec[idx] = params.V0
            elseif j == i_zl1
                # At z=l1 (anode) impose φ = Va inside either hole.
                if (x_val >= hole1[1] && x_val <= hole1[2]) || (x_val >= hole2[1] && x_val <= hole2[2])
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = params.Va
                else
                    # Otherwise, use interior discretization.
                    # z-part:
                    local coeff_z = 1/dz^2
                    push!(rows, idx); push!(cols, index(i, j+1, Nx)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, index(i, j-1, Nx)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, idx); push!(data, -2*coeff_z)
                    # x-part:
                    if i == 1
                        push!(rows, idx); push!(cols, index(1, j, Nx)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(2, j, Nx)); push!(data, -1.0)
                    elseif i == Nx
                        push!(rows, idx); push!(cols, index(Nx, j, Nx)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(Nx-1, j, Nx)); push!(data, -1.0)
                    else
                        push!(rows, idx); push!(cols, index(i+1, j, Nx)); push!(data, 1/dx^2)
                        push!(rows, idx); push!(cols, index(i-1, j, Nx)); push!(data, 1/dx^2)
                        push!(rows, idx); push!(cols, idx); push!(data, -2/dx^2)
                        #push!(rows, idx); push!(cols, idx); push!(data, 2/dx^2)
                    end
                end
            elseif j == i_zl2
                # At z=l2 (gate) impose φ = Vg inside either hole.
                if (x_val >= hole1[1] && x_val <= hole1[2]) || (x_val >= hole2[1] && x_val <= hole2[2])
                    push!(rows, idx); push!(cols, idx); push!(data, 1.0)
                    b_vec[idx] = params.Vg
                else
                    # Otherwise, use interior discretization.
                    local coeff_z = 1/dz^2
                    push!(rows, idx); push!(cols, index(i, j+1, Nx)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, index(i, j-1, Nx)); push!(data, coeff_z)
                    push!(rows, idx); push!(cols, idx); push!(data, -2*coeff_z)
                    if i == 1
                        push!(rows, idx); push!(cols, index(1, j, Nx)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(2, j, Nx)); push!(data, -1.0)
                    elseif i == Nx
                        push!(rows, idx); push!(cols, index(Nx, j, Nx)); push!(data, 1.0)
                        push!(rows, idx); push!(cols, index(Nx-1, j, Nx)); push!(data, -1.0)
                    else
                        push!(rows, idx); push!(cols, index(i+1, j, Nx)); push!(data, 1/dx^2)
                        push!(rows, idx); push!(cols, index(i-1, j, Nx)); push!(data, 1/dx^2)
                        push!(rows, idx); push!(cols, idx); push!(data, -2/dx^2)
                        #push!(rows, idx); push!(cols, idx); push!(data, 2/dx^2)
                    end
                end
            else
                # Interior nodes (or nodes not on electrode planes)
                local coeff_z = 1/dz^2
                push!(rows, idx); push!(cols, index(i, j+1, Nx)); push!(data, coeff_z)
                push!(rows, idx); push!(cols, index(i, j-1, Nx)); push!(data, coeff_z)
                push!(rows, idx); push!(cols, idx); push!(data, -2*coeff_z)
                if i == 1
                    push!(rows, idx); push!(cols, index(1, j, Nx)); push!(data, 1.0)
                    push!(rows, idx); push!(cols, index(2, j, Nx)); push!(data, -1.0)
                elseif i == Nx
                    push!(rows, idx); push!(cols, index(Nx, j, Nx)); push!(data, 1.0)
                    push!(rows, idx); push!(cols, index(Nx-1, j, Nx)); push!(data, -1.0)
                else
                    push!(rows, idx); push!(cols, index(i+1, j, Nx)); push!(data, 1/dx^2)
                    push!(rows, idx); push!(cols, index(i-1, j, Nx)); push!(data, 1/dx^2)
                    push!(rows, idx); push!(cols, idx); push!(data, -2/dx^2)
                    #push!(rows, idx); push!(cols, idx); push!(data, 2/dx^2)
                end
            end
        end
    end

    A_mat = sparse(rows, cols, data, Ntot, Ntot)
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nx, Nz)
    return phi_mat, xgrid, zgrid
end

#############################
# 2. Compute the Electric Field
#############################
"""
    compute_E_field_2d(phi_mat, xgrid, zgrid)

Computes the electric field components (E_x and E_z) from the potential phi_mat 
using finite differences.

Arguments:
  - phi_mat::AbstractMatrix{Float64}: Potential values on an (Nx×Nz) grid.
  - xgrid::AbstractVector{Float64}: Horizontal grid (x).
  - zgrid::AbstractVector{Float64}: Vertical grid (z).

Returns:
  - (E_x, E_z): Tuple of matrices representing the electric field components.
"""
function compute_E_field_2d(phi_mat::AbstractMatrix{Float64},
                            xgrid::AbstractVector{Float64},
                            zgrid::AbstractVector{Float64})
    Nx, Nz = size(phi_mat)
    dx = xgrid[2] - xgrid[1]
    dz = zgrid[2] - zgrid[1]
    E_x = zeros(Float64, Nx, Nz)
    E_z = zeros(Float64, Nx, Nz)
    
    @inbounds for j in 1:Nz
        for i in 1:Nx
            if i == 1
                E_x[i,j] = -(phi_mat[i+1,j] - phi_mat[i,j]) / dx
            elseif i == Nx
                E_x[i,j] = -(phi_mat[i,j] - phi_mat[i-1,j]) / dx
            else
                E_x[i,j] = -(phi_mat[i+1,j] - phi_mat[i-1,j]) / (2*dx)
            end
            if j == 1
                E_z[i,j] = -(phi_mat[i,j+1] - phi_mat[i,j]) / dz
            elseif j == Nz
                E_z[i,j] = -(phi_mat[i,j] - phi_mat[i,j-1]) / dz
            else
                E_z[i,j] = -(phi_mat[i,j+1] - phi_mat[i,j-1]) / (2*dz)
            end
        end
    end
    return E_x, E_z
end

#############################
# 3. Transport Electrons
#############################
"""
    transport_electrons_2d(phi_mat, xgrid, zgrid, l1; N_electrons=10, ds=1e-6)

Transports electrons from the top electrode (z = l3) to the anode (z = l1) along the field lines.
Electrons are initialized at uniformly spaced x positions (at z = l3) and then moved using Euler integration.
They move opposite to the local electric field (since electrons are negatively charged).

Arguments:
  - phi_mat::AbstractMatrix{Float64}: Potential matrix (Nx×Nz).
  - xgrid::AbstractVector{Float64}: Horizontal grid (x).
  - zgrid::AbstractVector{Float64}: Vertical grid (z).
  - l1::Float64: z-coordinate of the anode (stopping condition).
  - N_electrons::Int: Number of electrons (uniform initial positions in x).
  - ds::Float64: Integration step size.

Returns:
  - trajectories: A vector of trajectories, where each trajectory is a vector of [x, z] positions.
"""
function transport_electrons_2d(phi_mat::AbstractMatrix{Float64},
                                xgrid::AbstractVector{Float64},
                                zgrid::AbstractVector{Float64},
                                l1::Float64; N_electrons::Int=10, ds::Float64=1e-6)

    E_x, E_z = compute_E_field_2d(phi_mat, xgrid, zgrid)
    itp_E_x = interpolate((xgrid, zgrid), E_x, Gridded(Linear()))
    itp_E_z = interpolate((xgrid, zgrid), E_z, Gridded(Linear()))
    
    trajectories = Vector{Vector{Vector{Float64}}}()
    x_initial = collect(range(first(xgrid), stop=last(xgrid), length=N_electrons))
    print("x_initial = $(x_initial)")
    for x0 in x_initial
        pos = [x0, zgrid[end]]  # Starting at (x0, l3)
        traj = [copy(pos)]

        println("Initial pos = $(pos)\n")
        while pos[2] > l1
            if pos[1] < minimum(xgrid) || pos[1] > maximum(xgrid) ||
               pos[2] < minimum(zgrid) || pos[2] > maximum(zgrid)
                break
            end
            ex = itp_E_x(pos[1], pos[2])
            ez = itp_E_z(pos[1], pos[2])
            normE = sqrt(ex^2 + ez^2)
            if normE < 1e-12
                break
            end
            dxds = -ex / normE
            dzds = -ez / normE

            println("dxds =$(dxds)  dzds = $(dzds)")

            pos[1] += ds * dxds
            pos[2] += ds * dzds

            println("after displacement: pos =$(pos)")
            push!(traj, copy(pos))
        end
        push!(trajectories, traj)
    end
    return trajectories
end



