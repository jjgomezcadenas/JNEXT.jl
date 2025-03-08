using SparseArrays
using Interpolations, LinearAlgebra

struct ParH3D
    x0::Float64  # center in x
    y0::Float64  # center in y
    d1::Float64  # hole diameter (in x-y plane)
    p::Float64   # pitch (used here to define domain size in x and y)
    z0::Float64  # ground position
    za::Float64  # anode position
    zg::Float64  # gate position
    zc::Float64  # collector position
    V0::Float64  # ground potential
    Va::Float64  # anode potential
    Vg::Float64  # gate potential
    Vc::Float64  # collector potential
end

function phi3D(par::ParH3D, Nx::Int, Ny::Int, Nz::Int)
    # Helper function: return fixed potential at a Dirichlet node.
    function get_dirichlet_value(i, j, k)
        if k == i_z0
            return V0
        elseif k == i_zc
            return Vc
        elseif k == i_za
            return Va
        elseif k == i_zg
            return Vg
        else
            error("Node ($i,$j,$k) is not a Dirichlet boundary")
        end
    end

    # Define the computational domain in x and y from x0-p/2 to x0+p/2, etc.
    x_min = par.x0 - par.p/2 
    x_max = par.x0 + par.p/2 
    y_min = par.y0 - par.p/2 
    y_max = par.y0 + par.p/2 
   
    # Compute grid spacings
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    dz = (par.zc - par.z0) / (Nz - 1)

    # Precompute the inverse squared spacings
    invdx2 = 1.0/dx^2
    invdy2 = 1.0/dy^2
    invdz2 = 1.0/dz^2
    
    # Coordinate arrays
    xg = collect(range(x_min, x_max, length=Nx))
    yg = collect(range(y_min, y_max, length=Ny))
    zg = collect(range(par.z0, par.zc, length=Nz))

    # Dynamically find the grid indices corresponding to the electrode planes.
    # (Assume par.z0 < par.za < par.zg < par.zc)
    i_z0 = findmin(abs.(zg .- par.z0))[2]
    i_za = findmin(abs.(zg .- par.za))[2]
    i_zg = findmin(abs.(zg .- par.zg))[2]
    i_zc = findmin(abs.(zg .- par.zc))[2]
    @assert zg[i_z0] ≤ zg[i_za] ≤ zg[i_zg] ≤ zg[i_zc] "Plane indices out of order"

    # Extract electrode potentials for convenience
    V0, Va, Vg, Vc = par.V0, par.Va, par.Vg, par.Vc
    hole_radius = par.d1/2  # hole radius in the x-y plane

    # Prepare a mapping for unknown nodes.
    # unknown_index[i,j,k] = -1 for known (Dirichlet) nodes, or a unique positive index for unknown nodes.
    unknown_index = fill(-1, Nx, Ny, Nz)
    unknown_count = 0

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if k == i_z0
            # Ground plane: full Dirichlet (V0)
            unknown_index[i,j,k] = -1
        elseif k == i_zc
            # Collector plane: full Dirichlet (Vc)
            unknown_index[i,j,k] = -1
        elseif k == i_za
            # Anode plane: outside the hole, Dirichlet (Va); inside the hole, unknown.
            # We approximate the circular hole by using the condition (x^2+y^2 < hole_radius^2)
            if ((xg[i] - par.x0)^2 + (yg[j] - par.y0)^2) < hole_radius^2
                unknown_count += 1
                unknown_index[i,j,k] = unknown_count
            else
                unknown_index[i,j,k] = -1
            end
        elseif k == i_zg
            # Gate plane: outside the hole, Dirichlet (Vg); inside the hole, unknown.
            if ((xg[i] - par.x0)^2 + (yg[j] - par.y0)^2) < hole_radius^2
                unknown_count += 1
                unknown_index[i,j,k] = unknown_count
            else
                unknown_index[i,j,k] = -1
            end
        else
            # Interior points: unknown
            unknown_count += 1
            unknown_index[i,j,k] = unknown_count
        end
    end


    # Assemble the sparse matrix using a 7-point stencil and construct RHS vector b.
    I = Int[]
    J = Int[]
    V_arr = Float64[]
    b = zeros(Float64, unknown_count)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        let idx = unknown_index[i,j,k]
            if idx == -1
                continue  # Skip Dirichlet (known) nodes.
            end
            neighbor_coeff = 0.0  # Sum of coefficients for diagonal

            # --- X-direction ---
            if i == 1
                # Left boundary: use mirror with neighbor at i+1 (Neumann)
                local_i = i+1
                if unknown_index[local_i,j,k] != -1
                    push!(I, idx); push!(J, unknown_index[local_i,j,k]); push!(V_arr, 2.0*invdx2)
                else
                    b[idx] -= 2.0*invdx2 * get_dirichlet_value(local_i,j,k)
                end
                neighbor_coeff += 2.0*invdx2
            else
                local_i = i-1
                if unknown_index[local_i,j,k] != -1
                    push!(I, idx); push!(J, unknown_index[local_i,j,k]); push!(V_arr, invdx2)
                else
                    b[idx] -= invdx2 * get_dirichlet_value(local_i,j,k)
                end
                neighbor_coeff += invdx2
            end

            if i == Nx
                # Right boundary: mirror using neighbor at i-1
                local_i = i-1
                if unknown_index[local_i,j,k] != -1
                    push!(I, idx); push!(J, unknown_index[local_i,j,k]); push!(V_arr, 2.0*invdx2)
                else
                    b[idx] -= 2.0*invdx2 * get_dirichlet_value(local_i,j,k)
                end
                neighbor_coeff += 2.0*invdx2
            else
                local_i = i+1
                if unknown_index[local_i,j,k] != -1
                    push!(I, idx); push!(J, unknown_index[local_i,j,k]); push!(V_arr, invdx2)
                else
                    b[idx] -= invdx2 * get_dirichlet_value(local_i,j,k)
                end
                neighbor_coeff += invdx2
            end

            # --- Y-direction ---
            if j == 1
                # Bottom boundary (ymin): mirror using neighbor at j+1
                local_j = j+1
                if unknown_index[i,local_j,k] != -1
                    push!(I, idx); push!(J, unknown_index[i,local_j,k]); push!(V_arr, 2.0*invdy2)
                else
                    b[idx] -= 2.0*invdy2 * get_dirichlet_value(i,local_j,k)
                end
                neighbor_coeff += 2.0*invdy2
            else
                local_j = j-1
                if unknown_index[i,local_j,k] != -1
                    push!(I, idx); push!(J, unknown_index[i,local_j,k]); push!(V_arr, invdy2)
                else
                    b[idx] -= invdy2 * get_dirichlet_value(i,local_j,k)
                end
                neighbor_coeff += invdy2
            end

            if j == Ny
                # Top boundary (ymax): mirror using neighbor at j-1
                local_j = j-1
                if unknown_index[i,local_j,k] != -1
                    push!(I, idx); push!(J, unknown_index[i,local_j,k]); push!(V_arr, 2.0*invdy2)
                else
                    b[idx] -= 2.0*invdy2 * get_dirichlet_value(i,local_j,k)
                end
                neighbor_coeff += 2.0*invdy2
            else
                local_j = j+1
                if unknown_index[i,local_j,k] != -1
                    push!(I, idx); push!(J, unknown_index[i,local_j,k]); push!(V_arr, invdy2)
                else
                    b[idx] -= invdy2 * get_dirichlet_value(i,local_j,k)
                end
                neighbor_coeff += invdy2
            end

            # --- Z-direction ---
            if k > 1
                local_k = k-1
                if unknown_index[i,j,local_k] != -1
                    push!(I, idx); push!(J, unknown_index[i,j,local_k]); push!(V_arr, invdz2)
                else
                    b[idx] -= invdz2 * get_dirichlet_value(i,j,local_k)
                end
                neighbor_coeff += invdz2
            end
            if k < Nz
                local_k = k+1
                if unknown_index[i,j,local_k] != -1
                    push!(I, idx); push!(J, unknown_index[i,j,local_k]); push!(V_arr, invdz2)
                else
                    b[idx] -= invdz2 * get_dirichlet_value(i,j,local_k)
                end
                neighbor_coeff += invdz2
            end

            # Diagonal entry: negative sum of neighbor coefficients
            push!(I, idx); push!(J, idx); push!(V_arr, -neighbor_coeff)
        end
    end

    # 6. Build sparse matrix and solve the linear system
    A = sparse(I, J, V_arr, unknown_count, unknown_count)
    φ_vec = A \ b

    # 7. Reconstruct full 3D potential array by inserting Dirichlet values and solved unknowns
    phi_full = Array{Float64}(undef, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if unknown_index[i,j,k] == -1
            # Dirichlet node: assign fixed potential based on plane
            if k == i_z0
                phi_full[i,j,k] = V0
            elseif k == i_zc
                phi_full[i,j,k] = Vc
            elseif k == i_za
                phi_full[i,j,k] = Va
            elseif k == i_zg
                phi_full[i,j,k] = Vg
            else
                error("Unexpected Dirichlet node at ($i,$j,$k)")
            end
        else
            phi_full[i,j,k] = φ_vec[ unknown_index[i,j,k] ]
        end
    end

    return phi_full, xg, yg, zg
end


function phi3D_sor(par::ParH3D, Nx::Int, Ny::Int, Nz::Int; ω=1.5, tol=1e-6, max_iter=10000)
    # Define the computational domain (same as before)
    x_min = par.x0 - par.p/2 
    x_max = par.x0 + par.p/2 
    y_min = par.y0 - par.p/2 
    y_max = par.y0 + par.p/2 
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    dz = (par.zc - par.z0) / (Nz - 1)
    xg = collect(range(x_min, x_max, length=Nx))
    yg = collect(range(y_min, y_max, length=Ny))
    zg = collect(range(par.z0, par.zc, length=Nz))
    
    # Determine electrode plane indices
    i_z0 = findmin(abs.(zg .- par.z0))[2]
    i_za = findmin(abs.(zg .- par.za))[2]
    i_zg = findmin(abs.(zg .- par.zg))[2]
    i_zc = findmin(abs.(zg .- par.zc))[2]
    @assert zg[i_z0] ≤ zg[i_za] ≤ zg[i_zg] ≤ zg[i_zc] "Plane indices out of order"
    
    V0, Va, Vg, Vc = par.V0, par.Va, par.Vg, par.Vc
    hole_radius = par.d1/2

    # Create a 3D array for potential and a Boolean mask for fixed nodes.
    phi = zeros(Float64, Nx, Ny, Nz)
    fixed = falses(Nx, Ny, Nz)
    
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if k == i_z0
            phi[i,j,k] = V0
            fixed[i,j,k] = true
        elseif k == i_zc
            phi[i,j,k] = Vc
            fixed[i,j,k] = true
        elseif k == i_za
            # Anode plane: outside hole fixed at Va; inside hole free
            if ((xg[i]-par.x0)^2 + (yg[j]-par.y0)^2) < hole_radius^2
                fixed[i,j,k] = false
            else
                phi[i,j,k] = Va
                fixed[i,j,k] = true
            end
        elseif k == i_zg
            # Gate plane: outside hole fixed at Vg; inside hole free
            if ((xg[i]-par.x0)^2 + (yg[j]-par.y0)^2) < hole_radius^2
                fixed[i,j,k] = false
            else
                phi[i,j,k] = Vg
                fixed[i,j,k] = true
            end
        else
            fixed[i,j,k] = false
        end
    end

    # Precompute constants for the update
    invdx2 = 1.0/dx^2
    invdy2 = 1.0/dy^2
    invdz2 = 1.0/dz^2
    denom = 2*invdx2 + 2*invdy2 + 2*invdz2

    # SOR iteration
    resid = Inf
    iter = 0
    while resid > tol && iter < max_iter
        resid = 0.0
        iter += 1
        @inbounds for k in 2:(Nz-1), j in 2:(Ny-1), i in 2:(Nx-1)
            if fixed[i,j,k]
                continue
            end
            # For neighbors in x, if at boundary use mirror (Neumann)
            left  = (i == 1)   ? phi[i+1,j,k] : phi[i-1,j,k]
            right = (i == Nx)  ? phi[i-1,j,k] : phi[i+1,j,k]
            down  = (j == 1)   ? phi[i,j+1,k] : phi[i,j-1,k]
            up    = (j == Ny)  ? phi[i,j-1,k] : phi[i,j+1,k]
            back  = phi[i,j,k-1]  # z-direction are Dirichlet at boundaries
            front = phi[i,j,k+1]
            new_val = (invdx2*(left + right) + invdy2*(down + up) + invdz2*(back + front)) / denom
            # Over-relaxation update:
            diff = new_val - phi[i,j,k]
            phi[i,j,k] += ω * diff
            resid = max(resid, abs(diff))
        end
        # (Optionally, print or log iteration and residual)
        # @show iter, resid
        if iter%500 == 0
            println("iter = $(iter), residual =$(resid)")
        end
    end

    return phi, xg, yg, zg
end


function exyz(x::AbstractVector, y::AbstractVector, z::AbstractVector, phi, Nx::Int, Ny::Int, Nz::Int)
    # Preallocate electric field arrays with same size as phi.
    E_x = zeros(Float64, Nx, Ny, Nz)
    E_y = zeros(Float64, Nx, Ny, Nz)
    E_z = zeros(Float64, Nx, Ny, Nz)
    
    # Compute grid spacings (assumed uniform)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]
    
    @inbounds for k in 1:Nz
        for j in 1:Ny
            for i in 1:Nx
                # X-component of E = -∂phi/∂x
                if i == 1
                    E_x[i,j,k] = - (phi[i+1,j,k] - phi[i,j,k]) / dx
                elseif i == Nx
                    E_x[i,j,k] = - (phi[i,j,k] - phi[i-1,j,k]) / dx
                else
                    E_x[i,j,k] = - (phi[i+1,j,k] - phi[i-1,j,k]) / (2*dx)
                end

                # Y-component of E = -∂phi/∂y
                if j == 1
                    E_y[i,j,k] = - (phi[i,j+1,k] - phi[i,j,k]) / dy
                elseif j == Ny
                    E_y[i,j,k] = - (phi[i,j,k] - phi[i,j-1,k]) / dy
                else
                    E_y[i,j,k] = - (phi[i,j+1,k] - phi[i,j-1,k]) / (2*dy)
                end

                # Z-component of E = -∂phi/∂z
                if k == 1
                    E_z[i,j,k] = - (phi[i,j,k+1] - phi[i,j,k]) / dz
                elseif k == Nz
                    E_z[i,j,k] = - (phi[i,j,k] - phi[i,j,k-1]) / dz
                else
                    E_z[i,j,k] = - (phi[i,j,k+1] - phi[i,j,k-1]) / (2*dz)
                end
            end
        end
    end

    return E_x, E_y, E_z
end



function simulate_electron_transport3D(x::AbstractVector, y::AbstractVector, z::AbstractVector,
    E_x, E_y, E_z, electron_xy0::Array{<:AbstractFloat,2}, par;
    dt=0.05, max_steps=2000)
    # electron_xy0 is an array of size (N_electrons, 2)
    N_electrons = size(electron_xy0, 1)
    trajectories = Vector{Matrix{Float64}}(undef, N_electrons)

    # Set start and stop z positions (for example, electrons injected at collector, travel downward)
    start_z = par.zc
    stop_z  = par.za
    

    # Create interpolants for the electric field components.
    # The grid order is (x, y, z) matching our arrays.
    itp_E_x = interpolate((x, y, z), E_x, Gridded(Linear()))
    itp_E_y = interpolate((x, y, z), E_y, Gridded(Linear()))
    itp_E_z = interpolate((x, y, z), E_z, Gridded(Linear()))

    for n in 1:N_electrons
        traj = Vector{Vector{Float64}}()
        # Initialize electron position: from electron_xy0 and start_z.
        pos = [electron_xy0[n,1], electron_xy0[n,2], start_z]
        push!(traj, copy(pos))
        Ex_local = 0.0
        Ey_local = 0.0
        Ez_local = 0.0
        for step in 1:max_steps
            # Interpolate the local electric field at the current position.
            # Note: the interpolants expect the order (x, y, z).
            try
                Ex_local = itp_E_x(pos[1], pos[2], pos[3])
                Ey_local = itp_E_y(pos[1], pos[2], pos[3])
                Ez_local = itp_E_z(pos[1], pos[2], pos[3])
            catch e
                # If out-of-bound, break this trajectory.
                break
            end

            # Electrons move opposite to the electric field (assuming negative charge).
            drift = -[Ex_local, Ey_local, Ez_local]
            norm_drift = norm(drift)
            if norm_drift != 0
                drift .= drift ./ norm_drift
            end

            # Update the position using Euler's method.
            pos .+= dt * drift
            push!(traj, copy(pos))

            # Stop if electron reaches or passes the stop_z (e.g., anode)
            if pos[3] <= stop_z
                break
            end
        end
        # Convert trajectory to a matrix with each row as a point.
        trajectories[n] = hcat(traj...)'
    end

    return trajectories
end