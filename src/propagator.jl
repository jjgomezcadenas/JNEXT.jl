using Random

"""
Solve the intersection with the cylinder wall 
"""
function solve_t_barrel(x, y, vx, vy, R; eps=1e-10)
    a = vx^2 + vy^2
    b = 2 * (x * vx + y * vy)
    c = x^2 + y^2 - R^2

    if abs(a) < eps
        return nothing
    end

    disc = b^2 - 4 * a * c
    if disc < 0
        return nothing
    end

    sqrt_disc = sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    ts_candidates = Float64[]
    if t1 > eps
        push!(ts_candidates, t1)
    end
    if t2 > eps
        push!(ts_candidates, t2)
    end

    if !isempty(ts_candidates)
        return minimum(ts_candidates)
    else
        return nothing
    end
end


"""
 Solve for time to reach the top (z = ztop).
"""
function solve_t_top(z, vz, ztop; eps=1e-10)
    if vz > eps
        dt = (ztop - z) / vz
        if dt > eps
            return dt
        end
    end
    return nothing
end


"""
 Solve for time to reach the bottom (z = zb).
"""
function solve_t_bottom(z, vz, zb; eps=1e-10)
    if vz < -eps
        dtb = zb-z / vz
        if dtb > eps
            return dtb
        end
    end
    return nothing
end


"""
Generate an isotropic 3D direction.
"""
function generate_direction()
    cost = 2.0 * rand() - 1.0  # cosine(theta) uniformly in [-1, 1]
    theta = acos(cost)         # theta in [0, π]
    phi = 2.0 * pi * rand()      # phi in [0, 2π]
    sinth = sqrt(1 - cost^2)
    vx = sinth * cos(phi)
    vy = sinth * sin(phi)
    vz = cost
    return (vx, vy, vz)
end


function simulate_photons_cylinder_axis(dSIPM, dEL, lEL; pTPB=0.65, pPTFE=0.9, N=100_000, seed=123, eps=1e-10)
    

    # Cylinder and acceptance parameters:
    R = dEL / 2.0           # Cylinder radius = diameter of EL hole / 2.
    Z = lEL                 # Cylinder length = length of EL region.
    p1 = pTPB             # re-emission probability on first barrel collision (VUV to blue: TPB).
    p2 = pPTFE            # re-emission probability on subsequent barrel collisions (blue reflection).
    d_t = dSIPM / sqrt(2.0) # Acceptance radius at the top (z=Z).

    

    # --- Main simulation loop ---
    for _ in 1:N
        # 1. Set initial position.
        x, y, z = generate_position()

        # 2. Generate initial isotropic direction.
        vx, vy, vz = generate_direction()

        n_collisions = 0
        alive = true

        while alive
            t_barrel = solve_t_barrel(x, y, vx, vy)
            t_top    = solve_t_top(z, vz)
            t_bottom = solve_t_bottom(z, vz)

            # Gather valid intersection times and corresponding surface labels.
            possible_times = Float64[]
            labels = String[]

            if t_barrel !== nothing
                push!(possible_times, t_barrel)
                push!(labels, "barrel")
            end
            if t_top !== nothing
                push!(possible_times, t_top)
                push!(labels, "top")
            end
            if t_bottom !== nothing
                push!(possible_times, t_bottom)
                push!(labels, "bottom")
            end

            # If no intersection is found, the photon is lost.
            if isempty(possible_times)
                alive = false
                break
            end

            # Choose the intersection that happens first.
            idx_min = argmin(possible_times)
            t_min = possible_times[idx_min]
            surf = labels[idx_min]

            # Move the photon.
            x_new = x + t_min * vx
            y_new = y + t_min * vy
            z_new = z + t_min * vz

            if surf == "top"
                # At the top, check if the photon falls within the SiPM acceptance.
                rr = sqrt(x_new^2 + y_new^2)
                if rr < d_t
                    count_top += 1
                end
                alive = false

            elseif surf == "bottom"
                # Photon is lost when reaching the bottom.
                alive = false

            else  # "barrel"
                n_collisions += 1
                # Determine re-emission probability.
                p = (n_collisions == 1) ? p1 : p2
                if rand() < p
                    # Photon is re-emitted from the collision point with a new isotropic direction.
                    x, y, z = x_new, y_new, z_new
                    vx, vy, vz = generate_direction()
                else
                    # Photon is lost if not re-emitted.
                    alive = false
                end
            end
        end
    end

    fraction_top = count_top / N
    return count_top, fraction_top
end