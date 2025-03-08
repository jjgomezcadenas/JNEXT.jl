using SparseArrays
using LinearAlgebra
using Plots
using Interpolations
using Statistics
using StatsBase
using StatsPlots
using Distributions


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
    rows  = Vector{Int}(undef, max_entries)
    cols  = Vector{Int}(undef, max_entries)
    vals  = Vector{Float64}(undef, max_entries)
    counter = 0
    b = zeros(Float64, N)

    # Helper to add a nonzero entry.
    function add_entry!(k, kk, coeff)
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
        # Anode: z = l1, if x no in hole
        if j == j_l1 && !in_hole(x[i], x1_center) 
            return true, Va
        end
        # Gate: z = l2, if x lies in either hole → φ = Vg
        if j == j_l2 && !in_hole(x[i], x1_center) 
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


function exz(x, z, phi, Nx, Nz)
    # Compute the electric field E = -∇φ using finite differences.
    # Use central differences in the interior and one-sided differences at the boundaries.
    E_x = zeros(size(phi))
    E_z = zeros(size(phi))
    
    dx = x[2] - x[1]
    dz = z[2] - z[1]
    
    # x-direction derivative
    @inbounds for j in 1:Nz
        @inbounds for i in 1:Nx
            if i == 1
                E_x[j, i] = - (phi[j, i+1] - phi[j, i]) / dx
            elseif i == Nx
                E_x[j, i] = - (phi[j, i] - phi[j, i-1]) / dx
            else
                E_x[j, i] = - (phi[j, i+1] - phi[j, i-1]) / (2 * dx)
            end
        end
    end

    # z-direction derivative
    for j in 1:Nz
        for i in 1:Nx
            if j == 1
                E_z[j, i] = - (phi[j+1, i] - phi[j, i]) / dz
            elseif j == Nz
                E_z[j, i] = - (phi[j, i] - phi[j-1, i]) / dz
            else
                E_z[j, i] = - (phi[j+1, i] - phi[j-1, i]) / (2 * dz)
            end
        end
    end
    
    return E_x, E_z
end



function simulate_electron_transport(x, z, E_x, E_z, electron_x0, params; dt=0.05, max_steps=2000)
    # Create interpolants for the electric field components.
    # The grid for interpolation is (z, x) since E_x and E_z are arrays of size (length(z), length(x))
    itp_E_x = interpolate((z, x), E_x, Gridded(Linear()))
    itp_E_z = interpolate((z, x), E_z, Gridded(Linear()))
    
    # Extract parameters from params.
    x1_center = params.x1      # center of hole 1 (mm)
    d1        = params.d1
    p         = params.p       # pitch between holes (mm)
    
    
    start_z = params.l3        # starting z-position (top electrode)
    stop_z  = params.l1        # stopping z-position (anode)
    stop_z0 = params.l0        # ground position
    tolz    = dt              # tolerance for stopping in z
    
    println("start_z = $(start_z), stop_anode_z = $(stop_z), stop_gnd_z = $(stop_z0)")
    
    # Helper: returns true if x_val lies inside the hole centered at hole_center.
    in_hole(x_val, hole_center) = (x_val > (hole_center - d1/2)) && (x_val < (hole_center + d1/2))
    
    # Initialize vectors to store the trajectories.
    trajectories  = Vector{Matrix{Float64}}()  # first-stage trajectories
    btrajectories = Vector{Matrix{Float64}}()  # secondary trajectories
    
    #############################
    # Stage 1: Initial trajectory
    #############################
    for x0 in electron_x0
        traj_points = Vector{Vector{Float64}}()  # each point is a 2-element [x, z] vector.
        pos = [x0, start_z]    # initial position: [x, z]
        push!(traj_points, copy(pos))
        
        for step in 1:max_steps
            # For interpolation we need (z, x) order.
            pt = [pos[2], pos[1]]
            # Check that pt lies inside the grid; if not, stop this trajectory.
            if pt[1] < minimum(z) || pt[1] > maximum(z) || pt[2] < minimum(x) || pt[2] > maximum(x)
                break
            end
            # Evaluate the local electric field.
            Ex_local = itp_E_x(pt[1], pt[2])
            Ez_local = itp_E_z(pt[1], pt[2])
            # Electrons move opposite to the electric field.
            drift = -[Ex_local, Ez_local]
            nrm = norm(drift)
            if nrm != 0
                drift ./= nrm
            end
            # Update the position using Euler integration.
            pos .+= dt * drift
            push!(traj_points, copy(pos))
            # Stop if the electron has reached or passed the stopping z position.
            if pos[2] <= stop_z
                break
            end
        end
        # Convert the collected points into a matrix where each row is a point.
        # For example, if traj_points contained 
        #[ [x1, z1], [x2, z2], ..., [xn, zn] ], then hcat(traj_points...) 
        # creates a matrix with 2 rows and n columns:
        #[ x1   x2   ... xn;
        #  z1   z2   ... zn ]
       #The apostrophe (') transposes the matrix created by hcat, turning it into an n x 2 matrix.
       #[ x1  z1;
       # x2  z2;
       # ...
       #xn  zn ]

        traj_mat = hcat(traj_points...)'
        push!(trajectories, traj_mat)
    end
    
    ##############################################
    # Stage 2: Further trajectory for electrons in the hole
    ##############################################
    for traj in trajectories
        # Start from the last point of the first stage.
        pos = copy(traj[end, :])
        # Adjust the z coordinate.
        pos[2] -= 2*dt
        # Only continue if the final x-position is inside the hole (centered at x1_center).
        if !in_hole(pos[1], x1_center)
            continue
        end
        btraj_points = Vector{Vector{Float64}}()
        push!(btraj_points, copy(pos))
        
        for ii in 1:max_steps
            pt = [pos[2], pos[1]]
            if pt[1] < minimum(z) || pt[1] > maximum(z) || pt[2] < minimum(x) || pt[2] > maximum(x)
                break
            end
            Ex_local = itp_E_x(pt[1], pt[2])
            Ez_local = itp_E_z(pt[1], pt[2])
            # In this stage electrons drift along the electric field.
            drift = -[Ex_local, Ez_local]
            nrm = norm(drift)
            if nrm != 0
                drift ./= nrm
            end
            pos .+= dt * drift
            push!(btraj_points, copy(pos))
            # Stop if the electron goes back above the anode threshold or reaches ground.
            if pos[2] > stop_z - tolz || pos[2] <= stop_z0
                break
            end
        end
        if !isempty(btraj_points)
            btraj_mat = hcat(btraj_points...)'
            push!(btrajectories, btraj_mat)
        end
    end
    
    return trajectories, btrajectories
end

# Helper function: returns true if point (x,y) is inside the hole.
in_hole(x, y, par) = sqrt((x - par.x0)^2 + (y - par.y0)^2) < (par.d1/2)

function simulate_electron_transport3Dx(x::AbstractVector, y::AbstractVector, z::AbstractVector,
    E_x, E_y, E_z, electron_xy0::Array{<:AbstractFloat,2}, par;
    dt=0.05, max_steps=2000)

    # Number of electrons from the rows of electron_xy0.
    N_electrons = size(electron_xy0, 1)
    
    # Define start and stop z positions.
    # In our case: start at collector and stop at anode.
    start_z = par.zc
    stop_z  = par.za    # stage 1 stops when electron reaches the anode.
    stop_z0 = par.z0    # stage 2 stops if electron reaches ground.
    tol   = dt        # tolerance for stopping in z
    
    println("Stage 1: Propagating electrons from z = $(start_z) down to z = $(stop_z)")
    
    # Create 3D interpolants for the electric field components.
    # Note: the grid order is (x, y, z)
    itp_E_x = interpolate((x, y, z), E_x, Gridded(Linear()))
    itp_E_y = interpolate((x, y, z), E_y, Gridded(Linear()))
    itp_E_z = interpolate((x, y, z), E_z, Gridded(Linear()))
    
    # Containers for trajectories.
    trajectories  = Vector{Matrix{Float64}}()  # stage 1 trajectories
    btrajectories = Vector{Matrix{Float64}}()  # stage 2 (extended) trajectories

    ##############################
    # Stage 1: Propagate electrons from collector to anode.
    ##############################
    for n in 1:N_electrons
        traj_points = Vector{Vector{Float64}}()  # each point is a 3-element [x,y,z] vector.
        # Initial position: use provided (x,y) and start_z.
        pos = [electron_xy0[n,1], electron_xy0[n,2], start_z]
        push!(traj_points, copy(pos))
        
        for step in 1:max_steps
            # Check that pos is inside the grid.
            if pos[1] < minimum(x) || pos[1] > maximum(x) ||
               pos[2] < minimum(y) || pos[2] > maximum(y) ||
               pos[3] < minimum(z) || pos[3] > maximum(z)
                break
            end
            # Evaluate the local electric field at pos.
            # In stage 1, electrons move opposite to E (since they are negative).
            Ex_local = itp_E_x(pos[1], pos[2], pos[3])
            Ey_local = itp_E_y(pos[1], pos[2], pos[3])
            Ez_local = itp_E_z(pos[1], pos[2], pos[3])
            field = [Ex_local, Ey_local, Ez_local]
            # Drift: opposite to the field.
            drift = -field
            nrm = norm(drift)
            if nrm != 0
                drift ./= nrm
            end
            pos .+= dt * drift
            push!(traj_points, copy(pos))
            # Stop if electron reaches (or crosses) the anode plane.
            if pos[3] <= stop_z
                break
            end
        end
        traj_mat = hcat(traj_points...)'   # Each row is a point [x,y,z]
        push!(trajectories, traj_mat)
    end

    ##############################
    # Stage 2: Further propagation for electrons that ended in the hole.
    ##############################
    println("Stage 2: Extending trajectories for electrons in the hole...")
    for traj in trajectories
        pos = copy(traj[end, :])   # starting from last point of stage 1
        # Adjust z by a small amount to ensure we are in the hole region.
        pos[3] -= 2*dt
        # Only propagate further if the (x,y) position is inside the hole.
        #if !in_hole(pos[1], pos[2], par)
        #    continue
        #end
        btraj_points = Vector{Vector{Float64}}()
        push!(btraj_points, copy(pos))
        for ii in 1:max_steps
            if pos[1] < minimum(x) || pos[1] > maximum(x) ||
               pos[2] < minimum(y) || pos[2] > maximum(y) ||
               pos[3] < minimum(z) || pos[3] > maximum(z)
                break
            end
            # In stage 2, electrons drift along the field (drift = +E)
            Ex_local = itp_E_x(pos[1], pos[2], pos[3])
            Ey_local = itp_E_y(pos[1], pos[2], pos[3])
            Ez_local = itp_E_z(pos[1], pos[2], pos[3])
            field = [Ex_local, Ey_local, Ez_local]
            drift = -field
            nrm = norm(drift)
            if nrm != 0
                drift ./= nrm
            end
            pos .+= dt * drift
            push!(btraj_points, copy(pos))
            # Stop if the electron goes back above the anode or reaches ground.
            if pos[3] > stop_z - tol || pos[3] <= stop_z0
                break
            end
        end
        if !isempty(btraj_points)
            btraj_mat = hcat(btraj_points...)'
            push!(btrajectories, btraj_mat)
        end
    end

    return trajectories, btrajectories
end