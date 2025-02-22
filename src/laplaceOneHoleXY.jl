begin
	using SparseArrays
	using LinearAlgebra
	using Plots
	using Statistics
	using StatsBase
	using StatsPlots
	using Distributions
	using Interpolations
end

""" 
    GALAParams

Structure to hold the parameters of the GALA problem.
  - d: hole diameter.
  - rmax: half–extent of the horizontal domain (the x–domain runs from –rmax to rmax).
  - l1: z–position of the anode (potential Va).
  - l2: z–position of the gate (potential Vg).
  - l3: z–position of the top electrode (potential V0).
  - Va: potential at the anode.
  - Vg: potential at the gate.
  - V0: potential at the top electrode.
"""
struct GALAParams
    d::Float64      
    rmax::Float64   
    l1::Float64     
    l2::Float64     
    l3::Float64     
    Va::Float64     
    Vg::Float64     
    V0::Float64     
end

"""
    find_z_index(zval, zgrid)

Given a target z–value and a grid vector zgrid, return the index of the grid point closest to zval.
"""
function find_z_index(zval, zgrid)
    diffs = abs.(zgrid .- zval)
    return argmin(diffs)
end

"""
    solve_laplace_gala_xz(params::GALAParams; Nx=101, Nz=101)

Solve Laplace’s equation in the (X,Z) plane with:
  - x ∈ [ -rmax, rmax ]
  - z ∈ [ 0, l3 ]

Boundary conditions:
  - At z = 0 (the ground): φ = 0 for all x.
  - At z = l3 (the top electrode): φ = V0 for all x.
  - At z = l1 (the anode): impose φ = Va only for x ∈ [ -d/2, d/2 ].
  - At z = l2 (the gate): impose φ = Vg only for x ∈ [ -d/2, d/2 ].
  - At x = –rmax and x = rmax: impose Neumann conditions (∂φ/∂x = 0).

The Laplace equation in Cartesian coordinates,
  
    φₓₓ + φzz = 0,

is discretized using central differences in the interior.

Returns a tuple (phi_mat, xgrid, zgrid) where phi_mat is an (Nx×Nz) matrix.
"""
function solve_laplace_gala_xz(params::GALAParams; Nx=101, Nz=101)
    # Define the domain:
    # x runs from -rmax to rmax.
    x_min = -params.rmax
    x_max =  params.rmax
    # z runs from 0 to l3.
    z_min = 0.0
    z_max = params.l3

    dx = (x_max - x_min) / (Nx - 1)
    dz = (z_max - z_min) / (Nz - 1)
    xgrid = collect(range(x_min, x_max, length=Nx))
    zgrid = collect(range(z_min, z_max, length=Nz))

    Ntot = Nx * Nz
    A_mat = spzeros(Float64, Ntot, Ntot)
    b_vec = zeros(Float64, Ntot)

    # Mapping: index(i,j) = (j-1)*Nx + i, for i = 1:Nx and j = 1:Nz.
    index(i, j) = (j - 1) * Nx + i

    # Determine grid indices corresponding to the electrode planes in z.
    i_z0  = find_z_index(0.0,   zgrid)
    i_zl1 = find_z_index(params.l1, zgrid)
    i_zl2 = find_z_index(params.l2, zgrid)
    i_zl3 = find_z_index(params.l3, zgrid)

    # Loop over grid points.
    for j in 1:Nz
        for i in 1:Nx
            idx = index(i, j)
            x_val = xgrid[i]
            z_val = zgrid[j]

            # --- Dirichlet conditions in z: ---
            if j == i_z0
                # Ground at z = 0.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = 0.0
            elseif j == i_zl3
                # Top electrode at z = l3.
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.V0
            # At z = l1 (anode): impose φ = Va only for x inside the hole.
            elseif (j == i_zl1) && (x_val >= -params.d/2 && x_val <= params.d/2)
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Va
            # At z = l2 (gate): impose φ = Vg only for x inside the hole.
            elseif (j == i_zl2) && (x_val >= -params.d/2 && x_val <= params.d/2)
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Vg
            else
                # --- Interior or Neumann boundaries ---
                # z–direction: use standard central differences.
                A_mat[idx, index(i, j+1)] += 1/dz^2
                A_mat[idx, index(i, j-1)] += 1/dz^2
                A_mat[idx, idx]          += -2/dz^2

                # x–direction:
                if i == 1
                    # Left boundary (x = x_min): impose Neumann (∂φ/∂x = 0) via a one-sided difference.
                    A_mat[idx, index(1, j)] = 1.0
                    A_mat[idx, index(2, j)] = -1.0
                elseif i == Nx
                    # Right boundary (x = x_max): impose Neumann.
                    A_mat[idx, index(Nx, j)]   = 1.0
                    A_mat[idx, index(Nx-1, j)] = -1.0
                else
                    # Interior x nodes: central difference.
                    A_mat[idx, index(i+1, j)] += 1/dx^2
                    A_mat[idx, index(i-1, j)] += 1/dx^2
                    A_mat[idx, idx]           += -2/dx^2
                end
            end
        end
    end

    # Solve the sparse linear system.
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nx, Nz)
    return (phi_mat, xgrid, zgrid)
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


"""
    compute_electron_trajectories_xz(phi_mat, xgrid, zgrid, params; N_electrons=10, ds=1e-6, max_steps=1000000)

Compute trajectories for multiple electrons injected at the top electrode (z = l3) in the (X,Z) domain.

Electrons are injected uniformly in x ∈ [x_min, x_max] (with x_min = –rmax and x_max = rmax)
at z = l3 and then integrated along the electric field lines using `electron_trajectory`.

# Arguments
- `phi_mat`: Matrix of the potential (computed from solve_laplace_gala_xz).
- `xgrid, zgrid`: Vectors defining the spatial grid in x and z.
- `params`: A GALAParams structure containing electrode positions and potentials.
- `N_electrons`: Number of electrons to simulate (default is 10).
- `ds`: Integration step size (default is 1e-6).
- `max_steps`: Maximum number of integration steps for each electron (default is 1e6).

# Returns
A vector of trajectories, where each trajectory is a tuple `(x_traj, z_traj)`.
"""
function compute_electron_trajectories_xz(phi_mat, xgrid, zgrid, params; N_electrons=10, ds=1e-6, max_steps=1000000)
    # Compute the electric field on the (x,z) grid.
    # (Here, compute_E_field works generically; the first coordinate is now x.)
    E_x, E_z = compute_E_field(phi_mat, xgrid, zgrid)
    
    # Create interpolation objects for the field components and wrap them with extrapolation.
    interp_Ex = extrapolate(interpolate((xgrid, zgrid), E_x, Gridded(Linear())), Flat())
    interp_Ez = extrapolate(interpolate((xgrid, zgrid), E_z, Gridded(Linear())), Flat())
    
    trajectories = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    x_min = first(xgrid)
    x_max = last(xgrid)  # This is rmax in the original parameters, but here the domain is x ∈ [x_min, x_max]
    
    # Electrons are injected at z = l3 with x uniformly in [x_min, x_max].
    for n in 1:N_electrons
        x0 = rand() * (x_max - x_min) + x_min
        z0 = last(zgrid)  # should equal params.l3
        traj = electron_trajectory(x0, z0, interp_Ex, interp_Ez, xgrid, zgrid;
                                   ds=ds, params=params, rmax=x_max, max_steps=max_steps)
        push!(trajectories, traj)
    end
    
    return trajectories
end

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


"""
    plot_trajectories_xz(trajectories; params=nothing)

Plot the electron trajectories in the (X,Z) plane.

# Arguments
- `trajectories::Vector{Tuple{AbstractVector,AbstractVector}}`: A vector where each element is a tuple `(x_traj, z_traj)` containing the electron's (x,z) positions along its path.
- `params` (optional): A `GALAParams` instance. If provided, horizontal lines are added to mark the electrode positions (ground at z = 0, anode at z = l1, gate at z = l2, and top electrode at z = l3) and vertical lines at x = –d/2 and x = d/2 to mark the hole boundaries.

# Returns
A Plots.jl plot object.
"""
function plot_trajectories_xz(trajectories; params=nothing)
    plt = plot(legend=false, xlabel="x [m]", ylabel="z [m]", title="Electron Trajectories (XZ)")
    # Plot each trajectory.
    for (x_traj, z_traj) in trajectories
        plot!(plt, x_traj, z_traj, lw=2)
    end

    if params !== nothing
        # Mark electrode positions as horizontal dashed lines over the full x-domain.
        plot!(plt, [-params.rmax, params.rmax], [params.l3, params.l3],
              ls=:dash, lw=1, color=:black, label="Top Electrode")
        plot!(plt, [-params.rmax, params.rmax], [params.l2, params.l2],
              ls=:dash, lw=1, color=:gray, label="Gate")
        plot!(plt, [-params.rmax, params.rmax], [params.l1, params.l1],
              ls=:dash, lw=1, color=:red, label="Anode")
        plot!(plt, [-params.rmax, params.rmax], [0, 0],
              ls=:dash, lw=1, color=:blue, label="Ground")
        # Mark the hole boundaries at x = -d/2 and x = d/2 as vertical dotted lines.
        vline!(plt, [-params.d/2, params.d/2],
               ls=:dot, lw=1, color=:purple, label="Hole Boundary")
    end

    return plt
end

"""
    plot_phi_3D(phi_mat, rgrid, zgrid)

Plot a 3D surface of φ(r,z).
"""
function plot_phi_3D(phi_mat, rgrid, zgrid)
    surface(rgrid, zgrid, phi_mat',
        xlabel="r", ylabel="z", zlabel="φ(r,z)",
        title="3D Surface of φ(r,z)")
end


"""
    plot_phi_contour(phi_mat, rgrid, zgrid)

Plot a contour (filled) of φ(r,z) in the (r,z) plane.
"""
function plot_phi_contour(phi_mat, rgrid, zgrid)
    contour(rgrid, zgrid, phi_mat',
        xlabel="r", ylabel="z", title="Contour of φ(r,z)", fill=true, c=:viridis)
end

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


plot_phi_contour(phi_mat, rgrid, zgrid)

plot_E_field_quiver(phi_mat, rgrid, zgrid; skip_r=2, skip_z=2, scale=0.1)

trajectories = compute_electron_trajectories(phi_mat, rgrid, zgrid, gparams;
                                             N_electrons=200, 
                                             ds=1e-3, max_steps=500000)

plot_trajectories(trajectories)
