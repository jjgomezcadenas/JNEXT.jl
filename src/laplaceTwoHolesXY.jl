###############################################################################
# laplaceTwoHolesXY.jl
###############################################################################
using SparseArrays
using LinearAlgebra
using Plots
using Interpolations
using Statistics
using StatsBase
using StatsPlots
using Distributions





# =============================================================================
# Helper function: find_z_index
#
# Given a target z–value and a grid vector zgrid, return the index of the grid point
# closest to zval.
# =============================================================================
function find_z_index(zval, zgrid)
    diffs = abs.(zgrid .- zval)
    return argmin(diffs)
end


struct GalaParams
    x1::Float64     # x-position of Hole 1 center
    d1::Float64     # hole diameter [m]
    p::Float64      # pitch [m]
    l0::Float64     # z-position of ground (potential 0) [m]
    l1::Float64     # z-position of anode (potential Va) [m]
    l2::Float64     # z-position of gate (potential Vg) [m]
    l3::Float64     # z-position of top electrode (potential V0) [m]
    Va::Float64     # anode potential [V]
    Vg::Float64     # gate potential [V]
    V0::Float64     # top electrode potential [V]
end

"""
Solve laplace in two dimensiones
"""
function phi2d(params::GalaParams, Nx::Int, Nz::Int)
    # Unpack parameters
    x1_center = params.x1
    d1 = params.d1
    p = params.p
    l0, l1, l2, l3 = params.l0, params.l1, params.l2, params.l3
    Va, Vg, V0 = params.Va, params.Vg, params.V0

    # Define x-domain: x from (x1 - p/2) to (x1 + p + p/2)
    x_min = x1_center - p/2
    x_max = x1_center + p/2
    x = range(x_min, stop=x_max, length=Nx)
    z = range(l0, stop=l3, length=Nz)
    dx = (x_max - x_min) / (Nx - 1)
    dz = (l3 - l0) / (Nz - 1)

    # Compute grid indices for Dirichlet boundaries.
    # (argmin returns the index of the minimum value)
    j_l0 = argmin(abs.(z .- l0))
    j_l1 = argmin(abs.(z .- l1))
    j_l2 = argmin(abs.(z .- l2))
    j_l3 = argmin(abs.(z .- l3))

    N = Nx * Nz
    # Preallocate space for sparse matrix entries.
    max_entries = 7 * N   # upper-bound estimate for nonzero entries
    rows  = Vector{Int}(undef, N)
    cols  = Vector{Int}(undef, N)
    vals  = Vector{Float64}(undef, N)
    counter = 0
    b = zeros(Float64, N)

    # Helper to add a nonzero entry.
    function add_entry!(k, kk, coeff)
        nonlocal counter
        counter += 1
        rows[counter] = k
        cols[counter] = kk
        vals[counter] = coeff
    end

    # Return true if x_val is within the hole centered at hole_center.
    in_hole(x_val, hole_center) = (x_val > (hole_center - d1/2)) && (x_val < (hole_center + d1/2))

    # Check if grid point (j, i) is Dirichlet and return (flag, value).
    function is_dirichlet(j, i)
        # Bottom electrode: z = l0 → φ = 0
        if j == j_l0
            return true, 0.0
        end
        # Top electrode: z = l3 → φ = V0
        if j == j_l3
            return true, V0
        end
        # Anode: z = l1, if x lies in either hole → φ = Va
        if j == j_l1 && in_hole(x[i], x1_center) 
            return true, Va
        end
        # Gate: z = l2, if x lies in either hole → φ = Vg
        if j == j_l2 && in_hole(x[i], x1_center) 
            return true, Vg
        end
        return false, 0.0
    end

    # Build the finite-difference system.
    @inbounds for j in 1:Nz
        for i in 1:Nx
            k = (j - 1) * Nx + i  # Flatten 2D index to 1D.
            fixed, val = is_dirichlet(j, i)
            if fixed
                add_entry!(k, k, 1.0)
                b[k] = val
                continue
            end

            # x-direction finite differences.
            if i == 1
                # Left boundary: Neumann (one-sided derivative).
                add_entry!(k, k, -2.0/dx^2)
                k_right = k + 1  # (j, i+1)
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r
                    b[k] -= (2.0/dx^2) * val_r
                else
                    add_entry!(k, k_right, 2.0/dx^2)
                end
            elseif i == Nx
                # Right boundary: Neumann.
                add_entry!(k, k, -2.0/dx^2)
                k_left = k - 1  # (j, i-1)
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l
                    b[k] -= (2.0/dx^2) * val_l
                else
                    add_entry!(k, k_left, 2.0/dx^2)
                end
            else
                # Interior in x.
                add_entry!(k, k, -2.0/dx^2)
                k_right = k + 1
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r
                    b[k] -= (1.0/dx^2)*val_r
                else
                    add_entry!(k, k_right, 1.0/dx^2)
                end
                k_left = k - 1
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l
                    b[k] -= (1.0/dx^2)*val_l
                else
                    add_entry!(k, k_left, 1.0/dx^2)
                end
            end

            # z-direction finite differences (skip boundaries which are Dirichlet).
            if j != 1 && j != Nz
                add_entry!(k, k, -2.0/dz^2)
                k_up = k + Nx   # (j+1, i)
                fixed_up, val_up = is_dirichlet(j+1, i)
                if fixed_up
                    b[k] -= (1.0/dz^2)*val_up
                else
                    add_entry!(k, k_up, 1.0/dz^2)
                end
                k_down = k - Nx  # (j-1, i)
                fixed_down, val_down = is_dirichlet(j-1, i)
                if fixed_down
                    b[k] -= (1.0/dz^2)*val_down
                else
                    add_entry!(k, k_down, 1.0/dz^2)
                end
            end
        end
    end

    # Trim the arrays to the actual number of nonzero entries.
    rows = rows[1:counter]
    cols = cols[1:counter]
    vals = vals[1:counter]

    # Build the sparse matrix.
    A = sparse(rows, cols, vals, N, N)

    # Solve the system.
    phi_flat = A \ b
    # Reshape the solution to a 2D array (Nz rows, Nx columns). Note that we transpose
    # because our linear index was computed in row-major order.
    phi = reshape(phi_flat, (Nx, Nz))'
    return collect(x), collect(z), phi
end

"""
    solve_laplace_gala_xz(params::GALAParams; Nx=101, Nz=101)

Solve Laplace’s equation in the (x,z) plane for a two–hole configuration.
The x–domain now extends from –(p+d) to (p+d) and the holes are defined as:
  - Hole 1 center: x = -(p+d)/2
  - Hole 2 center: x =  (p+d)/2
Each hole has diameter d.
Dirichlet conditions (φ = Va at the anode and φ = Vg at the gate) are applied 
only within the hole regions.

Parameters:
  - params: a GALAParams instance (with fields d, p, l1, l2, l3, Va, Vg, V0)
  - Nx, Nz: number of grid points in x and z

Returns a tuple (phi_mat, xgrid, zgrid).
"""
function solve_laplace_gala_xz(params::GALAParams; Nx=101, Nz=101)
    # Define the horizontal domain:
    # x ∈ [-(p+d), p+d]
    x_min = -params.rmax 
    x_max =  params.rmax
    z_min = 0.0
    z_max = params.l3

    dx = (x_max - x_min) / (Nx - 1)
    dz = (z_max - z_min) / (Nz - 1)
    xgrid = collect(range(x_min, x_max, length=Nx))
    zgrid = collect(range(z_min, z_max, length=Nz))
    
    Ntot = Nx * Nz
    A_mat = spzeros(Float64, Ntot, Ntot)
    b_vec = zeros(Float64, Ntot)
    
    # Mapping from grid indices to linear index.
    index(i, j) = (j - 1) * Nx + i
    
    # Determine indices for key z-planes.
    i_z0  = find_z_index(0.0, zgrid)
    i_zl1 = find_z_index(params.l1, zgrid)
    i_zl2 = find_z_index(params.l2, zgrid)
    i_zl3 = find_z_index(params.l3, zgrid)
    
    # Define hole positions.
    # Hole 1 center is at - (p+d)/2 and hole 2 center is at + (p+d)/2.
    center1 = -(params.p) / 2
    center2 =  (params.p) / 2
    # Each hole extends from center - d/2 to center + d/2.
    hole1_left  = center1 - params.d/2
    hole1_right = center1 + params.d/2
    hole2_left  = center2 - params.d/2
    hole2_right = center2 + params.d/2

    # Loop over grid points.
    for j in 1:Nz
        for i in 1:Nx
            idx = index(i, j)
            x_val = xgrid[i]
            z_val = zgrid[j]
            
            # Apply Dirichlet conditions on the horizontal boundaries in z.
            if j == i_z0
                A_mat[idx, idx] = 1.0
                b_vec[idx] = 0.0
            elseif j == i_zl3
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.V0
            # At the anode (z = l1): apply φ = Va only in the hole regions.
            elseif j == i_zl1
                if (x_val >= hole1_left && x_val <= hole1_right) || 
                   (x_val >= hole2_left && x_val <= hole2_right)
                    A_mat[idx, idx] = 1.0
                    b_vec[idx] = params.Va
                else
                    # Outside the holes, use Laplace discretization.
                    A_mat[idx, index(i, j+1)] += 1/dz^2
                    A_mat[idx, index(i, j-1)] += 1/dz^2
                    A_mat[idx, idx]           += -2/dz^2
                    if i == 1
                        A_mat[idx, index(1, j)] = 1.0
                        A_mat[idx, index(2, j)] = -1.0
                    elseif i == Nx
                        A_mat[idx, index(Nx, j)]   = 1.0
                        A_mat[idx, index(Nx-1, j)] = -1.0
                    else
                        A_mat[idx, index(i+1, j)] += 1/dx^2
                        A_mat[idx, index(i-1, j)] += 1/dx^2
                        A_mat[idx, idx]           += -2/dx^2
                    end
                end
            # At the gate (z = l2): apply φ = Vg only in the hole regions.
            elseif j == i_zl2
                if (x_val >= hole1_left && x_val <= hole1_right) || 
                   (x_val >= hole2_left && x_val <= hole2_right)
                    A_mat[idx, idx] = 1.0
                    b_vec[idx] = params.Vg
                else
                    # Outside the holes, use Laplace discretization.
                    A_mat[idx, index(i, j+1)] += 1/dz^2
                    A_mat[idx, index(i, j-1)] += 1/dz^2
                    A_mat[idx, idx]           += -2/dz^2
                    if i == 1
                        A_mat[idx, index(1, j)] = 1.0
                        A_mat[idx, index(2, j)] = -1.0
                    elseif i == Nx
                        A_mat[idx, index(Nx, j)]   = 1.0
                        A_mat[idx, index(Nx-1, j)] = -1.0
                    else
                        A_mat[idx, index(i+1, j)] += 1/dx^2
                        A_mat[idx, index(i-1, j)] += 1/dx^2
                        A_mat[idx, idx]           += -2/dx^2
                    end
                end
            else
                # For all other points, discretize Laplace's equation.
                A_mat[idx, index(i, j+1)] += 1/dz^2
                A_mat[idx, index(i, j-1)] += 1/dz^2
                A_mat[idx, idx]           += -2/dz^2
                if i == 1
                    A_mat[idx, index(1, j)] = 1.0
                    A_mat[idx, index(2, j)] = -1.0
                elseif i == Nx
                    A_mat[idx, index(Nx, j)]   = 1.0
                    A_mat[idx, index(Nx-1, j)] = -1.0
                else
                    A_mat[idx, index(i+1, j)] += 1/dx^2
                    A_mat[idx, index(i-1, j)] += 1/dx^2
                    A_mat[idx, idx]           += -2/dx^2
                end
            end
        end
    end

    # Solve the system.
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nx, Nz)
    return (phi_mat, xgrid, zgrid)
end


# =============================================================================
# solve_laplace_gala_xz
#
# Solve Laplace’s equation in the (x,z) plane for the two–hole configuration.
#
# Domain:
#   x ∈ [–rmax, rmax]
#   z ∈ [0, l3]
#
# Boundary conditions:
#   - At z = 0: φ = 0 (ground) for all x.
#   - At z = l3: φ = V0 for all x.
#   - At z = l1 (anode): impose φ = Va only for x in the union of the two hole regions.
#       (Hole 1: x ∈ [–d, 0] and Hole 2: x ∈ [0, d].)
#   - At z = l2 (gate): impose φ = Vg only for x in the hole regions.
#   - At x = –rmax and x = rmax: impose Neumann (∂φ/∂x = 0).
#
# The Laplace equation is discretized with central differences.
# =============================================================================
function solve_laplace_gala_xz2(params::GALAParams; Nx=101, Nz=101)
    # Domain definitions.
    x_min = -params.rmax
    x_max =  params.rmax
    z_min = 0.0
    z_max = params.l3

    dx = (x_max - x_min) / (Nx - 1)
    dz = (z_max - z_min) / (Nz - 1)
    xgrid = collect(range(x_min, x_max, length=Nx))
    zgrid = collect(range(z_min, z_max, length=Nz))

    Ntot = Nx * Nz
    A_mat = spzeros(Float64, Ntot, Ntot)
    b_vec = zeros(Float64, Ntot)

    # Mapping: index(i,j) = (j-1)*Nx + i.
    index(i, j) = (j - 1) * Nx + i

    # Determine grid indices corresponding to the key z–planes.
    i_z0  = find_z_index(0.0, zgrid)
    i_zl1 = find_z_index(params.l1, zgrid)
    i_zl2 = find_z_index(params.l2, zgrid)
    i_zl3 = find_z_index(params.l3, zgrid)

    # Define the hole regions.
    # Hole 1: centered at –d/2, covering x ∈ [–d, 0].
    # Hole 2: centered at d/2, covering x ∈ [0, d].
    hole_left  = -params.d    # left edge of hole 1
    hole_right =  params.d    # right edge of hole 2

    # Loop over all grid points.
    for j in 1:Nz
        for i in 1:Nx
            idx = index(i, j)
            x_val = xgrid[i]
            z_val = zgrid[j]

            # Apply Dirichlet conditions on the full-plane boundaries in z.
            if j == i_z0
                # Ground: φ = 0 at z = 0.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = 0.0
            elseif j == i_zl3
                # Top electrode: φ = V0 at z = l3.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.V0
            # Electrode conditions applied only in the hole regions.
            elseif (j == i_zl1) && (x_val >= hole_left && x_val <= hole_right)
                # Anode at z = l1: φ = Va.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Va
            elseif (j == i_zl2) && (x_val >= hole_left && x_val <= hole_right)
                # Gate at z = l2: φ = Vg.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Vg
            else
                # Interior points: discretize Laplace’s equation, φₓₓ + φ_zz = 0.
                # z–direction (central differences).
                A_mat[idx, index(i, j+1)] += 1/dz^2
                A_mat[idx, index(i, j-1)] += 1/dz^2
                A_mat[idx, idx]           += -2/dz^2

                # x–direction.
                if i == 1
                    # Left boundary (x = x_min): impose Neumann, ∂φ/∂x = 0.
                    A_mat[idx, index(1, j)] = 1.0
                    A_mat[idx, index(2, j)] = -1.0
                elseif i == Nx
                    # Right boundary (x = x_max): impose Neumann.
                    A_mat[idx, index(Nx, j)]   = 1.0
                    A_mat[idx, index(Nx-1, j)] = -1.0
                else
                    # Interior x nodes: central differences.
                    A_mat[idx, index(i+1, j)] += 1/dx^2
                    A_mat[idx, index(i-1, j)] += 1/dx^2
                    A_mat[idx, idx]           += -2/dx^2
                end
            end
        end
    end

    # Solve the linear system.
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nx, Nz)
    return (phi_mat, xgrid, zgrid)
end

# =============================================================================
# compute_E_field: compute the electric field components from the potential φ.
#
# The field is defined as:
#   E_x = -∂φ/∂x   and   E_z = -∂φ/∂z.
# Central differences are used in the interior and one–sided differences at the boundaries.
# =============================================================================
function compute_E_field(phi_mat::AbstractMatrix, xgrid::AbstractVector, zgrid::AbstractVector)
    Nx, Nz = size(phi_mat)
    dx = xgrid[2] - xgrid[1]
    dz = zgrid[2] - zgrid[1]
    E_x = zeros(Nx, Nz)
    E_z = zeros(Nx, Nz)
    for j in 1:Nz
        for i in 1:Nx
            # x derivative.
            if i == 1
                E_x[i,j] = - (phi_mat[i+1,j] - phi_mat[i,j]) / dx
            elseif i == Nx
                E_x[i,j] = - (phi_mat[i,j] - phi_mat[i-1,j]) / dx
            else
                E_x[i,j] = - (phi_mat[i+1,j] - phi_mat[i-1,j]) / (2*dx)
            end
            # z derivative.
            if j == 1
                E_z[i,j] = - (phi_mat[i,j+1] - phi_mat[i,j]) / dz
            elseif j == Nz
                E_z[i,j] = - (phi_mat[i,j] - phi_mat[i,j-1]) / dz
            else
                E_z[i,j] = - (phi_mat[i,j+1] - phi_mat[i,j-1]) / (2*dz)
            end
        end
    end
    return E_x, E_z
end

# =============================================================================
# electron_trajectory: integrate a single electron trajectory in the (x,z) plane.
#
# The electron is assumed to move opposite to the electric field (since it is negatively
# charged). Integration is performed with a simple Euler scheme.
#
# The integration stops if:
#   - the electron leaves the computational domain,
#   - the electron reaches (or passes) the anode plane (z ≤ params.l1), or
#   - the maximum number of steps is exceeded.
# =============================================================================
function electron_trajectory(x0, z0, interp_Ex, interp_Ez, xgrid, zgrid; ds=1e-6, params, xmax, max_steps=1000000)
    x = x0
    z = z0
    x_traj = [x]
    z_traj = [z]
    for step in 1:max_steps
        # Check bounds.
        if x < first(xgrid) || x > last(xgrid) || z < first(zgrid) || z > last(zgrid)
            break
        end
        # Evaluate the interpolated electric field.
        Ex = interp_Ex(x, z)
        Ez = interp_Ez(x, z)
        normE = sqrt(Ex^2 + Ez^2)
        if normE < 1e-12
            break
        end
        # Electrons move opposite to the electric field.
        dxds = -Ex / normE
        dzds = -Ez / normE
        x_new = x + ds * dxds
        z_new = z + ds * dzds
        if x_new < first(xgrid) || x_new > last(xgrid) || z_new < first(zgrid) || z_new > last(zgrid)
            break
        end
        push!(x_traj, x_new)
        push!(z_traj, z_new)
        x = x_new
        z = z_new
        # Stop if the anode is reached.
        if z <= params.l1
            break
        end
    end
    return x_traj, z_traj
end

# =============================================================================
# compute_electron_trajectories_xz: compute trajectories for many electrons in (x,z).
#
# Electrons are injected uniformly along x ∈ [–rmax, rmax] at z = l3 (the top electrode)
# and then integrated along the field lines.
# =============================================================================
function compute_electron_trajectories_xz(phi_mat, xgrid, zgrid, params; N_electrons=10, ds=1e-6, max_steps=1000000)
    E_x, E_z = compute_E_field(phi_mat, xgrid, zgrid)
    interp_Ex = extrapolate(interpolate((xgrid, zgrid), E_x, Gridded(Linear())), Flat())
    interp_Ez = extrapolate(interpolate((xgrid, zgrid), E_z, Gridded(Linear())), Flat())
    trajectories = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    x_min = first(xgrid)
    x_max = last(xgrid)
    step = 1/N_electrons
    #println("xmin = $(x_min), xmax = $(x_max), step =$(step)")

    for n in 1:N_electrons
        x0 = (n-1) * step * (x_max - x_min) + x_min
        z0 = last(zgrid)  # injection at the top (z = l3)

        #println("electron number $(n), x0 = $(x0), zo = $(z0)")
        traj = electron_trajectory(x0, z0, interp_Ex, interp_Ez, xgrid, zgrid;
                                   ds=ds, params=params, xmax=x_max, max_steps=max_steps)

        println("Trajectory starts at $(traj[1][1]), ends at $(traj[end][end])")
        push!(trajectories, traj)
    end
    return trajectories
end

# =============================================================================
# Plotting functions.
# =============================================================================
function plot_phi_contour(phi_mat, xgrid, zgrid)
    contour(xgrid, zgrid, phi_mat', xlabel="x [m]", ylabel="z [m]",
            title="Contour of φ(x,z)", fill=true, c=:viridis)
end

function plot_E_field_quiver(phi_mat, xgrid, zgrid; skip_x=2, skip_z=2, scale=0.1)
    E_x, E_z = compute_E_field(phi_mat, xgrid, zgrid)
    x_points = Float64[]
    z_points = Float64[]
    U = Float64[]
    V = Float64[]
    for j in 1:skip_z:length(zgrid)
        for i in 1:skip_x:length(xgrid)
            push!(x_points, xgrid[i])
            push!(z_points, zgrid[j])
            push!(U, scale * E_x[i,j])
            push!(V, scale * E_z[i,j])
        end
    end
    quiver(x_points, z_points, quiver=(U, V),
           xlabel="x [m]", ylabel="z [m]", title="Electric Field (Quiver)",
           aspect_ratio=:equal, legend=false)
end

function plot_trajectories_xz(trajectories; params=nothing)
    plt = plot(legend=false, xlabel="x [m]", ylabel="z [m]",
               title="Electron Trajectories (XZ)")
    for (x_traj, z_traj) in trajectories
        plot!(plt, x_traj, z_traj, lw=2)
    end
    if params !== nothing
        # Mark the electrode planes.
        plot!(plt, [-params.rmax, params.rmax], [params.l3, params.l3],
              ls=:dash, lw=1, color=:black, label="Top Electrode")
        plot!(plt, [-params.rmax, params.rmax], [params.l2, params.l2],
              ls=:dash, lw=1, color=:gray, label="Gate")
        plot!(plt, [-params.rmax, params.rmax], [params.l1, params.l1],
              ls=:dash, lw=1, color=:red, label="Anode")
        plot!(plt, [-params.rmax, params.rmax], [0, 0],
              ls=:dash, lw=1, color=:blue, label="Ground")
        # Mark the hole boundaries: vertical lines at x = –d and x = d.
        vline!(plt, [-params.d, params.d], ls=:dot, lw=1, color=:purple, label="Hole Boundaries")
    end
    return plt
end


# A helper function to compute circle coordinates.
function circle_coords(center_x, center_y, radius; num_points=100)
    θ = LinRange(0, 2π, num_points)
    x_circle = center_x .+ radius .* cos.(θ)
    y_circle = center_y .+ radius .* sin.(θ)
    return x_circle, y_circle
end


"""
    plot_trajectories_xz(trajectories; params=nothing)


"""
function plot_trajectories_xz2(trajectories; params=nothing)
    plt = plot(legend=false, xlabel="x [m]", ylabel="z [m]",
               title="Electron Trajectories (XZ)")
    # Plot each trajectory.
    for (x_traj, z_traj) in trajectories
        plot!(plt, x_traj, z_traj, lw=2)
    end

    if params !== nothing
        # Mark the electrode planes as horizontal dashed lines.
        plot!(plt, [-params.rmax, params.rmax], [params.l3, params.l3],
              ls=:dash, lw=1, color=:black, label="Top Electrode")
        plot!(plt, [-params.rmax, params.rmax], [params.l2, params.l2],
              ls=:dash, lw=1, color=:gray, label="Gate")
        plot!(plt, [-params.rmax, params.rmax], [params.l1, params.l1],
              ls=:dash, lw=1, color=:red, label="Anode")
        plot!(plt, [-params.rmax, params.rmax], [0, 0],
              ls=:dash, lw=1, color=:blue, label="Ground")
        # Optionally, mark the vertical boundaries for the holes.
        vline!(plt, [-params.d, params.d], ls=:dot, lw=1, color=:purple, label="Hole Boundaries")
        
        # Add two circles representing the holes.
        # In this configuration the holes are centered at -d/2 and d/2, with radius d/2.
        # Here, we plot them at the anode level (z = params.l1).
        x_circle1, y_circle1 = circle_coords(-params.d/2, params.l1, params.d/2)
        x_circle2, y_circle2 = circle_coords( params.d/2, params.l1, params.d/2)
        plot!(plt, Shape(x_circle1, y_circle1), fillcolor=:none, linecolor=:purple, lw=2)
        plot!(plt, Shape(x_circle2, y_circle2), fillcolor=:none, linecolor=:purple, lw=2)
    end

    return plt
end