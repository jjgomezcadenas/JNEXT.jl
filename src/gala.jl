using LinearAlgebra
using Plots

using .SimpleLogger 

struct ELCCGeometry
	X::Float64         # ELCC total x dimension (mm)
	Y::Float64         # ELCC total y dimension (mm)
	Zc::Float64        # Z posiiton of collector (mm)
	Zg::Float64        # Z posiiton of gate (mm)
	Za::Float64        # Z posiiton of anode (mm)
	Zs::Float64        # Z posiiton of SiPM plane (mm)
	Vg::Float64        # potential at gate (in kV)
	Va::Float64        # potential at anode (in kV)
	d_hole::Float64    # Hole diameter in each dice (mm)
	pitch::Float64     # pitch (mm)	    
end


struct SiPMGeometry
	sipmSize::Float64  # Side length of SiPM active area (mm); assumed square.
	pitch::Float64     # Pitch (center-to-center distance) between SiPMs (mm)
	X::Float64         # Overall SiPM panel x dimension (mm)
	Y::Float64         # Overall SiPM panel y dimension (mm)
end


begin
	ndicex(elcc::ELCCGeometry) = Int(floor(elcc.X/elcc.pitch))
	ndicey(elcc::ELCCGeometry) = Int(floor(elcc.Y/elcc.pitch))
	nsipmx(sipm::SiPMGeometry) = Int(floor(sipm.X/sipm.pitch))
	nsipmy(sipm::SiPMGeometry) = Int(floor(sipm.Y/sipm.pitch))
end

set_log_level(DEBUG)

"""
Computes the yield per mm. 
- zg: position of the gate (mm)
- za: position of the anode (mm)
- vg: voltage at the gate (in kV)
- va: voltage at the anode (in kV)
"""
function yield_mm(zg, za, vg, va; p=10) # p in bar  
	vv = vg-va # vg and va in kV
	zz = (zg-za)/10 # zg and za in mm, so convert to cm
	eovp = abs(vv/zz)
	ycm = 140 * eovp - 116 * p
	ycm/10  # returns the yield/mm
end

"""
Generate a vector of fixed electron positions
"""
function generate_electron_positions(N::Int, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
	# Choose numbers of electrons in x and y directions so that n_x*n_y >= N.
	n_x = floor(Int, sqrt(N))
	n_y = ceil(Int, N / n_x)

	# Create uniform grids in x and y
	xs = collect(range(x_min, x_max, length=n_x))
	ys = collect(range(y_min, y_max, length=n_y))

	# Create the full grid of (x,y) pairs.
	pos = [(x,y) for y in ys, x in xs]  # Note: yields an n_y×n_x array of tuples.
	pos = vec(pos)  # Flatten to a 1D vector

	# Take the first N positions
	pos = pos[1:N]

	# Convert to a matrix with N rows and 2 columns.
	electron_xy0 = reduce(vcat, [reshape([p[1], p[2]], 1, 2) for p in pos])

	return electron_xy0
end

"""
Generate randmly the position of the initial electron.
"""
function generate_electron_positions_random(N::Int, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
	# Generate N random x positions uniformly distributed between x_min and x_max
	xs = x_min .+ (x_max - x_min) .* rand(N)

	# Generate N random y positions uniformly distributed between y_min and y_max
	ys = y_min .+ (y_max - y_min) .* rand(N)

	# Combine the x and y positions into an N×2 matrix
	electron_xy0 = hcat(xs, ys)

	return electron_xy0
end



"""
Given an electron absolute (x,y) position on the ELCC surface,
find the dice to which it is assigned and compute local (x_e,y_e).
"""
function find_dice(xpos::Vector{Float64}, elcc::ELCCGeometry)
    # Compute dice indices (starting at 1)
	
    i = clamp(floor(Int, (xpos[1]) / elcc.pitch) + 1, 1, ndicex(elcc))
    j = clamp(floor(Int, (xpos[2]) / elcc.pitch) + 1, 1, ndicey(elcc))
    # Dice lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*elcc.pitch, (j-1)*elcc.pitch )
    # Local coordinates inside dice
    xe = xpos[1] - dice_origin[1]
    ye = xpos[2] - dice_origin[2]
    return (i, j), dice_origin, (xe, ye)
end


"""
Given an electron absolute (x,y) position on the ELCC surface,
find the sipm to which it is assigned 
"""
function find_sipm(xpos::Vector{Float64}, sipm::SiPMGeometry)
    # Compute dice indices (starting at 1)
    i = clamp(floor(Int, (xpos[1]) / sipm.pitch) + 1, 1, nsipmx(sipm))
    j = clamp(floor(Int, (xpos[2]) / sipm.pitch) + 1, 1, nsipmy(sipm))
    # Dice lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*sipm.pitch, (j-1)*sipm.pitch )
   
	return (i, j), dice_origin
end


"""
Given an electron absolute (x,y) position on the ELCC surface,
find the sipm to which it is assigned 
"""
function find_abspos(xr::Tuple{Float64, Float64}, sipmIJ::Tuple{Int64, Int64}, sipm::SiPMGeometry)
    # Compute dice indices (starting at 1)
    i = sipmIJ[1]
    j = sipmIJ[2]
    # SiPM lower left corner (absolute coordinates)
    dice_origin = ( (i-1)*sipm.pitch, (j-1)*sipm.pitch )
   	xabs = xr[1] + dice_origin[1]
    yabs = xr[2] + dice_origin[2]
	return  xabs, yabs
end


"""
Simulate photons along the trajectory.

At each step along the trajectory, generate a number of photons.
Propagate them along.
Count photons that hit a SiPM (if the impact falls within a sensor active area).

- electron_post: the absolute position of the electron
- trj: A matrix holding a large number of trajectories, starting in the collector plane (where the electron_pos is defined) and ending in the anode.
- The ELCC structure
- The SiPM structure
- ymm: yield/mm in the EL region
- p1: probability of the photon to be absorbed in the first interaction
- p2: probability to be absorbed in further interactions.
- samplet: sampling of trayectory (to make it shorter than the frozen ones)
- maxgam: allows to generate less gammas than stipulated by the yield.
- saveg: allows to save gammas for plotting.
- savet: allows to save trajectory for plotting.
- ncmax: max number of bounces allowd for a given photon
- eps: tolerance factor
"""
function simulate_photons_along_trajectory(electron_pos::Vector{Float64},                                                       trj::AbstractMatrix, 
	                                       elcc::ELCCGeometry,
	                                       sipm::SiPMGeometry; 
										   ymm=10, # yield per mm
										   p1=1.00, # Prob abs first interaction
										   p2=0.5,  # Prob abs >1 interaction
										   samplet=1e+6, # sample trajectory
										   maxgam=1e6, # max number of gammas
										   saveg = false, # save gammas for plotting
										   savet = false, # save trj for plotting
                                           ncmax=20, # max number of bounces
										   eps=1e-14) # tolerance
	
	sipm_indices, sipm_origin =find_sipm(electron_pos, sipm)
	dice_indices, dice_origin, xlocal = find_dice(electron_pos, elcc)
	xl = collect(xlocal)
	
	dd = elcc.d_hole
	R = dd/2
	pp = (elcc.pitch - dd)/2
	x0 = y0 = pp + R
	ztop = elcc.Zg
	zbot = elcc.Zs
	
	debug2("--simulate_photons_along_trajectory--")
	debug2("r hole = $(R) position of hole x0=$(x0) y0=$(y0)")
	debug2("->absolute electron position $(electron_pos)")
	debug2("->electron position relative to dice $(xl)")
	debug2("->dice indices $(dice_indices), origin =$(dice_origin)")
	debug2("->sipm indices $(sipm_indices), origin =$(sipm_origin)")

	YL, XC, YC, ZC = get_coord_and_yield(trj, ztop, ymm)

	gammas = Vector{Vector{Vector{Float64}}}()
	jsteps =Vector{Vector{Float64}}()
	NTOP = Vector{Int}()
	NBOT = Vector{Int}()
	SIJ = Vector{Vector{Tuple{Int, Int}}}()
	
	# Trajectory steps
	ltrj = min(length(YL), samplet)
	for ri in range(1, stop=length(YL), length=ltrj)
	
		i =Int(floor(ri))
		xe = XC[i]
		ye = YC[i]
		ze = ZC[i]
		yy = YL[i]

		if savet
			push!(jsteps, [xe, ye, ze])
		end

		yield = min(yy, maxgam)
		count_top = 0
		count_bot = 0
		debug("--------000------------\n")
		debug("-->for step =$(i), xe = $(xe), y=$(ye), z=$(ze), yield=$(yield)")
		debug("--------000-----------\n")

		GIJ = Vector{Tuple{Int, Int}}()
		
		for ng in range(1, yield) # number of photons per step
			vx, vy, vz= generate_direction() # generate random direction
			
			debug("------xxxx-----------\n")
			debug("--->for ng =$(ng), vx =$(vx), vy =$(vy), vz = $(vz)")
			debug("------xxxx-----------\n")

			n_collisions = 0
			
        	alive = true

			if saveg
				steps = Vector{Vector{Float64}}()
				push!(steps, [xe, ye, ze])
			end

			x = xe
			y = ye
			z = ze

	        while alive
				if n_collisions > ncmax
					warn("exceeds number of collisions =", n_collisions)
					break
				end
				if z < elcc.Zg && z > elcc.Za
	            	t_barrel = solve_t_barrel(x, y, x0, y0, vx, vy, R; eps= eps)
				elseif z < elcc.Za && vz >0
					debug("gamma emitted outside hole, moving towards top: vz =$(vz)")
					count_top += 1
					break

					# ctx =atan(vx,-vz) 
					# cty =atan(vy,-vz) 
					# ctl = atan(R/abs(z - elcc.Za))
					# debug("ctx =$(ctx), cty =$(cty), ctl = $(ctl)")

					# if ctx < ctl || cty < ctl
					# 	#t_barrel = solve_t_barrel(x, y, x0, y0, vx, vy, R; eps= eps)
					# 	t_barrel = nothing
					# else
					# 	t_barrel = nothing
					# end
				elseif z < elcc.Za && vz <0
					t_bottom = solve_t_bottom(z, vz, elcc.Zs; eps=eps)
					debug("--->out of hole going towards bottom: t_bottom =$(t_bottom)")

					if t_bottom != nothing 
						x_new = x + t_bottom * vx
						y_new = y + t_bottom * vy
						z_new = z + t_bottom * vz
						push!(steps, [x_new, y_new, z_new])
						if saveg
							push!(gammas, steps)
						end
						count_bot += 1
						# keep position of sipm
						xabs, yabs = find_abspos((x_new, y_new), sipm_indices, sipm)
						debug("--->xabs =$(xabs), yabs = $(yabs)")
						sipmij, _ = find_sipm(collect((xabs, yabs)), sipm)
						debug("--->sipm(i,j) =($(sipmij[1]), $(sipmij[2]))")
						push!(GIJ, (sipmij[1], sipmij[2]))
						break
					end
				else
					t_barrel = nothing
				end

	            t_top    = solve_t_top(z, vz, elcc.Zc; eps=eps)
	            t_bottom = solve_t_bottom(z, vz,elcc.Zs; eps=eps)
				
				debug("--->t_barrel =$(t_barrel), t_top =$(t_top), t_bottom =$(t_bottom)")
	
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

				debug("--->t_min =$(t_min), surf =$(surf)")
	
	            # Move the photon.
	            x_new = x + t_min * vx
	            y_new = y + t_min * vy
	            z_new = z + t_min * vz

				debug("--->x_new =$(x_new), y_new =$(y_new), z_new =$(z_new)")
				push!(steps, [x_new, y_new, z_new])
	
	            if surf == "bottom" # anode 
	                alive = false
	                count_bot += 1
					if saveg
						push!(gammas, steps)
					end
					
					debug("--->photon hits bottom at x =$(x_new), y = $(y_new), count =$(count_bot)")
					
					# keep position of sipm
					xabs, yabs = find_abspos((x_new, y_new), sipm_indices, sipm)
					debug("--->xabs =$(xabs), yabs = $(yabs)")
					
					sipmij, _ = find_sipm(collect((xabs, yabs)), sipm)
					debug("--->sipm(i,j) =($(sipmij[1]), $(sipmij[2]))")

					push!(GIJ, (sipmij[1], sipmij[2]))
						
	            elseif surf == "top"
	                alive = false
					count_top += 1
					if saveg
						push!(gammas, steps)
					end
					
					debug("--->photon hits top at x =$(x_new), y = $(y_new), count =$(count_top)")

	            else  # "barrel only if we are still in hole"

					if z > elcc.Zg # bounced higher than gate. Count and kill
						t_top    = solve_t_top(z_new, vz, elcc.Zc; eps=eps)
						debug("--->bounce out on top: t_top =$(t_top)")
						if t_top != nothing 
							x_new = x + t_top * vx
	            			y_new = y + t_top * vy
	            			z_new = z + t_top * vz
							push!(steps, [x_new, y_new, z_new])
							if saveg
								push!(gammas, steps)
							end
							count_top += 1
							
							break
						end
					elseif z < elcc.Za # bounced lower than anode. Extrapolate, count and kill
						t_bottom = solve_t_bottom(z_new, vz, elcc.Zs; eps=eps)
						debug("--->bounce out on bottom: t_bottom =$(t_bottom)")

						if t_bottom != nothing 
							x_new = x + t_bottom * vx
	            			y_new = y + t_bottom * vy
	            			z_new = z + t_bottom * vz
							push!(steps, [x_new, y_new, z_new])
							if saveg
								push!(gammas, steps)
							end
							count_bot += 1
							# keep position of sipm
							xabs, yabs = find_abspos((x_new, y_new), sipm_indices, sipm)
							debug("--->xabs =$(xabs), yabs = $(yabs)")
							sipmij, _ = find_sipm(collect((xabs, yabs)), sipm)
							debug("--->sipm(i,j) =($(sipmij[1]), $(sipmij[2]))")
							push!(GIJ, (sipmij[1], sipmij[2]))
							break
						end
					end

					# now we are truly in the barrel 
	                n_collisions += 1
					
					debug("--->photon hits barrel at x =$(x_new), y = $(y_new), n_collisions =$(n_collisions)")
					
	                # Determine re-emission probability.
	                p = (n_collisions == 1) ? p1 : p2
	                
	                # Photon is re-emitted 
	                x, y, z = x_new, y_new, z_new
					vx0, vy0, vz0 = vx, vy, vz
	                    
					cteta = 1.0
					while cteta >= 0 #photon goes against the wall
						#p = (n_collisions == 1) ? p1 : p2
						if rand() < p # Photon is re-emitted 
							vx, vy, vz = generate_direction()
							cteta = dot([vx,vy,vz],[vx0,vy0,vx0])
							debug("---->photon reemited in barrel")
							debug("vx = $(vx), vy=$(vy), vz=$(vz)")
							debug("cos(theta) =$(cteta)")

							if cteta >=0
								debug("---->hit wall, n_collisions=$(n_collisions)")
							else
								debug("---->photon goes away from wall")
							end
							
						else   # Photon is absorbed
							debug("---->photon absorbed in barrel")
							alive = false	
							if saveg
								push!(gammas, steps)
							end
							
							break
						end
						 n_collisions += 1
					end   
					debug("x = $(x), y=$(y), z=$(z)")
					debug("vx = $(vx), vy=$(vy), vz=$(vz)")
					
            	end
			end
		end
		push!(NTOP, count_top)
		push!(NBOT, count_bot)
		push!(SIJ, GIJ)
	end
	NTOP, NBOT, SIJ, gammas, jsteps
end


"""
Get coordinates and yield
"""
function get_coord_and_yield(tr::AbstractMatrix, zg::Float64, ymm::Float64)
	iz0 = 0
	z0 = zg
	YL = Vector{Int64}()
	XC = Vector{Float64}()
	YC = Vector{Float64}()
	ZC = Vector{Float64}()
	for i in range(1, size(tr)[1])
		z = tr[i, 3]
		if z >zg
			continue
		elseif iz0 == 0
			iz0 = i
			z0 = tr[i, 3]
		end

		dz = abs(z0 - z)
		yl = Int(floor(dz * ymm))
		push!(YL, yl)
		push!(XC, tr[i, 1])
		push!(YC, tr[i, 2])
		push!(ZC, tr[i, 3])
		z0 = z
		
	end
	YL, XC, YC, ZC
end



"""
Generate random direction
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


"""
	Solve the intersection with the cylinder wall 
"""
function solve_t_barrel(x, y, x0, y0, vx, vy, R; eps=1e-16)
	a = vx^2 + vy^2
	b = 2 * ((x-x0) * vx + (y-y0) * vy)
	c = (x - x0)^2 + (y - y0)^2 - R^2

	debug("####solve_t_barrel: x-x0 = $(x-x0), y-y0=$(y-y0), vx = $(vx), vy=$(vy),R = $(R)")
	#debug("####a = $(a), b = $(b), c = $(c)")
	if abs(a) < eps
		return nothing
	end

	disc = b^2 - 4 * a * c
	#debug("####disc = $(disc)")
	if disc < 0
		return nothing
	end

	sqrt_disc = sqrt(disc)
	t1 = (-b + sqrt_disc) / (2 * a)
	t2 = (-b - sqrt_disc) / (2 * a)

	debug("####t1 = $(t1), t2 = $(t2)")

	ts_candidates = Float64[]
	
	if t1 > 0 && t2 > 0
		push!(ts_candidates, t1)
		push!(ts_candidates, t2)
	elseif t1 > 0 && t2 <= 0
		push!(ts_candidates, t1)
	elseif t2 >0 && t1 <= 0
		push!(ts_candidates, t2)
	elseif t1 >0 && abs(t2) <= eps
		push!(ts_candidates, t1)
	elseif t2 >0 && abs(t1) <= eps
		push!(ts_candidates, t2)
	elseif abs(t1) < eps || abs(t2) < eps
		push!(ts_candidates, 0.0)
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
function solve_t_top(z, vz, ztop; eps=1e-16)
	debug("####solve_t_top: z = $(z), vz=$(vz), ztop=$(ztop)")
    if vz > eps
        dt = (ztop - z) / vz
		debug("####solve_t_top: dt = $(dt)")
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
	debug("####solve_t_bottom: z = $(z), vz=$(vz), zb=$(zb)")
    if vz < -eps
        dtb = (abs(zb) -abs(z)) / abs(vz)
		debug("####solve_t_bottom: dt = $(dtb)")
        if dtb > eps
            return dtb
        end
    end
    return nothing
end


### PLOTS and Graphics


"""
Graphical representation of GALA structure.
"""
function plot_gala_geom(elcc::ELCCGeometry, sipm::SiPMGeometry)
	p1, p2 = pstructure(elcc, sipm)
    plot(p1, p2, layout = (1,2))
end


"""
Plots impact (xy) of the electron in the GALA
"""
function plot_impact_xy(elcc::ELCCGeometry, sipm::SiPMGeometry, electron_impact_xy::Vector{Float64})
	p1, p2 = pstructure(elcc, sipm)
	# Plot the electron impact point in red.
	p1 = scatter!(p1, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
	mc=:red, label=false)
	p2 = scatter!(p2, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
	mc=:red, label=false)
	plot(p1, p2, layout = (1,2))
end


"""
Helper function.
"""
function pstructure(elcc::ELCCGeometry, sipm::SiPMGeometry)
    # ELCC: Represented as a rectangle in the xy-plane.
    # Coordinates of ELCC (we assume its top face is at z = elcc.Z)
    elcc_rect = Shape([0, elcc.X, elcc.X, 0], [0, 0, elcc.Y, elcc.Y])
    
    # Plot ELCC outline.
    p1 = plot(elcc_rect, fillcolor=:lightblue, alpha=0.3, label="ELCC", 
		      aspect_ratio=1,
              xlabel="x (mm)", ylabel="y (mm)", title="ELCC and SiPM Layout")
    
    # Draw dice boundaries and holes.
    for i in 0:ndicex(elcc)
        plot!([i*elcc.pitch, i*elcc.pitch], [0, elcc.Y], 
			   lc=:gray, lw=0.5, label=false)
    end
    for j in 0:ndicey(elcc)
        plot!([0, elcc.X], [j*elcc.pitch, j*elcc.pitch], 
			  lc=:gray, lw=0.5, label=false)
    end
	
    # Draw holes: each hole is a circle at the center of its dice.
    for i in 0:(ndicex(elcc)-1)
        for j in 0:(ndicey(elcc)-1)
            cx = i*elcc.pitch + elcc.pitch/2
            cy = j*elcc.pitch + elcc.pitch/2
            circle = Shape(cx .+ (elcc.d_hole/2)*cos.(LinRange(0,2π,50)),
                           cy .+ (elcc.d_hole/2)*sin.(LinRange(0,2π,50)))
            plot!(circle, fillcolor=:white, linecolor=:black, lw=0.5, label=false)
        end
    end

    # Plot the SiPM plane: assume it is located below the ELCC, at a distance elcc.Zsipm from the bottom.
    # For visualization, we show the SiPM panel as a rectangle.
    sipm_rect = Shape([0, sipm.X, sipm.X, 0], [0, 0, sipm.Y, sipm.Y])
    p2 = plot(sipm_rect, fillcolor=:lightgreen, alpha=0.3, label="SiPM Panel", 
		      aspect_ratio=1,
              xlabel="x (mm)", ylabel="y (mm)", title="SiPM Panel Layout")
    # Draw SiPM boundaries (sensors are squares of side sipm.sipmSize centered in cells of size sipm.pitch).
    n_sipm_x = round(Int, sipm.X / sipm.pitch)
    n_sipm_y = round(Int, sipm.Y / sipm.pitch)
    for i in 0:n_sipm_x-1
        for j in 0:n_sipm_y-1
            sx = i*sipm.pitch + (sipm.pitch - sipm.sipmSize)/2
            sy = j*sipm.pitch + (sipm.pitch - sipm.sipmSize)/2
            sensor = Shape([sx, sx+sipm.sipmSize, sx+sipm.sipmSize, sx],
                           [sy, sy, sy+sipm.sipmSize, sy+sipm.sipmSize])
            plot!(sensor, fillcolor=:white, linecolor=:black, lw=1.0, label=false)
        end
    end

    # Layout plots side by side.
   p1, p2
end


"""
Plot the trajectory in the GALA structure
"""
function p_trajectory(epos::Vector{Float64}, elcc::ELCCGeometry, sipm::SiPMGeometry, tr::AbstractMatrix)

	function plott!(i, p, traj)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		plot!(p, traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false)


	end
	dice_indices, dice_origin, _ = find_dice(epos, elcc)
	posd = [dice_origin[1], dice_origin[2], elcc.Zc]
	xtrj = [tr[i,:] + [dice_origin[1], dice_origin[2], 0.0] for i in range(1, size(tr)[1]) ] 
	# Concatenate columns then transpose to get an n×3 matrix
	trj = hcat(xtrj...)'   
	p1, p2 = pt_lgeom(epos, elcc, sipm, elcc.Zs -0.5, elcc.Zc)
	p1 = plott!(1, p1, trj)
	p2 = plott!(2, p2, trj)

	plot(p1,p2)
end


function p_trajectory(epos::Vector{Float64}, elcc::ELCCGeometry, sipm::SiPMGeometry, 
	                  jsteps::Vector{Vector{Float64}}, gammas::Vector{Vector{Vector{Float64}}})

	function plott!(i, p, traj)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		plot!(p, traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false)
	end

	dice_i, dice_o, xlocal = find_dice(epos, elcc)
	sipm_i, sipm_o =find_sipm(epos, sipm)
	
	p1, p2 = pt_lgeom(epos, elcc, sipm, elcc.Zs -0.5, elcc.Zg + 0.5)

	# Extract x, y, z coordinates for each step in the jsteps.
	xs = [step[1] + dice_o[1] for step in jsteps]
	ys = [step[2] + dice_o[2] for step in jsteps]
	zs = [step[3] + elcc.Zg   for step in jsteps]

	p1 = plot!(p1, xs, zs, lw=1.0, lc=:cyan, label=false)
	p2 = plot!(p2, ys, zs, lw=1.0, lc=:cyan, label=false)

	# Extract x, y, z coordinates for each step in the gamma.
	for gamma in gammas
		debug("gamma = $(gamma)")
	 	xgs = [step[1] + dice_o[1] for step in gamma]
		ygs = [step[2] + dice_o[2] for step in gamma]
	 	zgs = [step[3] + elcc.Zg for step in gamma]

		debug("xgs = $(xgs)")
		debug("ygs = $(ygs)")
		debug("zgs = $(zgs)")

		p1 = scatter!(p1, xgs, zgs, ms=1, lc=:green, label=false)
		p2 = scatter!(p2, ygs, zgs, ms=1, lc=:green, label=false)
		p1 = plot!(p1, xgs, zgs, lw=0.5, lc=:green, linestyle=:dashdot, label=false)
		p2 = plot!(p2, ygs, zgs, lw=0.5, lc=:green, linestyle=:dashdot, label=false)
	end

	debug("xs = $(xs)")
	debug("ys = $(ys)")
	debug("zs = $(zs)")

	#p1 = plott!(1, p1, trj)
	#p2 = plott!(2, p2, trj)

	plot(p1,p2)
end


function pt_lgeom(epos::Vector{Float64}, elcc::ELCCGeometry, sipm::SiPMGeometry, zmin::Float64, zmax::Float64)
	
	dice_i, dice_o, xlocal = find_dice(epos, elcc)
	sipm_i, sipm_o =find_sipm(epos, sipm)

	xsipml = sipm_o[1]
	ysipmd = sipm_o[2]
	xdicel = dice_o[1]
	ydiced = dice_o[2]
	xl = minimum([xsipml, xdicel])
	yd = minimum([ysipmd, ydiced])
	

	dd = elcc.d_hole
	R = dd/2
	pp = (elcc.pitch - dd)/2
	x0 = y0 = pp + R
	
	println("xsipml= $(xsipml), ysipmd = $(ysipmd), xdicel =$(xdicel), ydiced =$(ydiced)")
	println("xl= $(xl), yd = $(yd)")

	println("epos= $(epos)")

	six0l = xsipml
	six0r = six0l + sipm.sipmSize
	six1l = six0l - sipm.pitch
	six1r = six1l + sipm.sipmSize
	six2r = six0r + sipm.pitch
	six2l = six2r - sipm.sipmSize

	siy0l = ysipmd
	siy0r = siy0l + sipm.sipmSize
	siy1l = siy0l - sipm.pitch
	siy1r = siy1l + sipm.sipmSize
	siy2r = siy0r + sipm.pitch
	siy2l = siy2r - sipm.sipmSize

	#dcx0l = xdicel
	#dcx0r = dcx0l + elcc.d_hole
	#dcx1l = dcx0l - elcc.pitch
	#dcx1r = dcx1l + elcc.d_hole
	#dcx2r = dcx0r + elcc.pitch
	#dcx2l = dcx2r - elcc.d_hole

	dcx0l = xdicel + x0 - R
	dcx0r = xdicel+ x0 + R
	

	# dcy0l = ydiced
	# dcy0r = dcy0l + elcc.d_hole
	# dcy1l = dcy0l - elcc.pitch
	# dcy1r = dcy1l + elcc.d_hole
	# dcy2r = dcy0r + elcc.pitch
	# dcy2l = dcy2r - elcc.d_hole

	dcy0l = ydiced + y0 - R
	dcy0r = ydiced+ y0 + R

	# xmin = minimum([six1l, dcx1l])
	# xmax = maximum([six2r, dcx2r])
	# ymin = minimum([siy1l, dcy1l])
	# ymax = maximum([siy2r, dcy2r])

	xmin = minimum([six1l, dcx0l])
	xmax = maximum([six2r, dcx0r])
	ymin = minimum([siy1l, dcy0l])
	ymax = maximum([siy2r, dcy0r])
	

	# println("six0l= $(six0l), six0r = $(six0r), six1l =$(six1l), six1r =$(six1r), six2l =$(six2l), six2r =$(six2r)")

	# println("dcy0l= $(dcy0l), dcy0r = $(dcy0r), dcy1l =$(dcy1l), dcy1r =$(dcy1r), dcy2l =$(dcy2l), dcy2r =$(dcy2r)")

	# println("xmin= $(xmin), xmax = $(xmax), ymin =$(ymin), ymax =$(ymax)")

	debug2("six0l= $(six0l), six0r = $(six0r), six1l =$(six1l), six1r =$(six1r), six2l =$(six2l), six2r =$(six2r)")

	debug2("dcx0l= $(dcx0l), dcx0r = $(dcx0r) dcy0l= $(dcy0l), dcy0r = $(dcy0r)")

	debug2("xmin= $(xmin), xmax = $(xmax), ymin =$(ymin), ymax =$(ymax)")
	
	p1 = plot(xlims=(xmin, xmax), ylims = (zmin, zmax),
		      xlabel="x (mm)", ylabel="z (mm)", title="XZ")
	p1 = scatter!(p1, [epos[1]], [zmax - 0.5], ms=2,
		          mc=:red, label=false)

	p1 = plot!(p1, [six0l, six0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p1 = plot!(p1, [six1l, six1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p1 = plot!(p1, [six2l, six2r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)

	p1 = plot!(p1, [dcx0l, dcx0l], [elcc.Zg, elcc.Za], 	
			       lw=2, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx0r, dcx0r], [elcc.Zg, elcc.Za], 	
			       lw=2, lc=:red, linestyle=:solid, label=false)
	#p1 = plot!(p1, [dcx1l, dcx1l], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p1 = plot!(p1, [dcx1r, dcx1r], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p1 = plot!(p1, [dcx2l, dcx2l], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p1 = plot!(p1, [dcx2r, dcx2r], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	
	p2 = plot(xlims=(ymin, ymax), ylims = (zmin, zmax),
		      xlabel="y (mm)", ylabel="z (mm)", title="YZ")
	p2 = scatter!(p2, [epos[2]], [zmax - 0.5], ms=2,
		          mc=:red, label=false)

	p2 = plot!(p2, [siy0l, siy0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p2 = plot!(p2, [siy1l, siy1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)
	p2 = plot!(p2, [siy2l, siy2r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:solid, label=false)

	p2 = plot!(p2, [dcy0l, dcy0l], [elcc.Zg, elcc.Za], 	
			       lw=2, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy0r, dcy0r], [elcc.Zg, elcc.Za], 	
			       lw=2, lc=:red, linestyle=:solid, label=false)
	#p2 = plot!(p2, [dcy1l, dcy1l], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p2 = plot!(p2, [dcy1r, dcy1r], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p2 = plot!(p2, [dcy2l, dcy2l], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	#p2 = plot!(p2, [dcy2r, dcy2r], [elcc.Zg, elcc.Za], 	
	#		       lw=1, lc=:red, linestyle=:solid, label=false)
	
	return p1, p2
end


"""
Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
point (shown in blue) on the SiPM plane.
- `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
- `photon_impact` is a tuple (x, z) on the SiPM plane.
"""
# function plot_gammas(gammas::Vector{Vector{Vector{Float64}}}, 
# 	                 traj::AbstractMatrix, 
# 	                 elcc::ELCCGeometry, sipm::SiPMGeometry;
#                      num_to_plot::Int=1)
	
# 	p1,p2 = plt_trajectory(traj, elcc, sipm)

	
# 	# Create a 3D plot with axis labels and a title.
# 	#plt = plot(title="Gamma Trajectories", xlabel="x", ylabel="y", zlabel="z", legend=:outertopright)

# 	# Use a built-in palette for distinct colors.
# 	colors = palette(:tab10)

# 	for i in 1:num_to_plot
#     	gamma = gammas[i]
#    	 # Extract x, y, z coordinates from each step in the gamma.
#     	xs = [step[1] for step in gamma]
#     	ys = [step[2] for step in gamma]
#     	zs = [step[3] for step in gamma]
    
#     	# Select color for this gamma.
#     	col = colors[(i - 1) % length(colors) + 1]
    
#     	# Plot points for each step.
#     	p1 = scatter!(p1, xs, ys, zs, label=false, marker=:circle, markersize=1, 
# 			     color=col)
#     	# Connect the points with a dashed line.
#     	p1 =plot!(p1, xs, ys, zs, label="", linestyle=:dash, linewidth=1, color=col)
# 	end

#     plot(p1,p2)
# end


# """

# Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
# point (shown in blue) on the SiPM plane.
# - `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
# - `photon_impact` is a tuple (x, z) on the SiPM plane.
# """
# function plt_trajectory(traj::AbstractMatrix, 
# 	                     xyl ::ELCCGeometry, sipm::SiPMGeometry)
	
# 	function plott(i, title)
# 		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
# 		p = plot(traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false,
# 			  #xlims=(xymin, xymax), ylims = (zmin, zmax),
# 			  xlims=(sixy1l, sixy2r), ylims = (zmin, zmax),
# 		      xlabel="x (mm)", ylabel="z (mm)", title=title)

# 		p = plot!(p, [xyl, xyr], [elcc.Zg, elcc.Zg], 
# 			       lw=1, lc=:red, linestyle=:dash, label=false)
# 		p = plot!(p, [xyl, xyr], [elcc.Za, elcc.Za], 
# 			       lw=1, lc=:red, linestyle=:dash, label=false)

# 		p = plot!(p, [sixy1l, sixy1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
# 			       lw=2, lc=:blue, linestyle=:dash, label=false)
		
# 		p = plot!(p, [sixy2l, sixy2r], [elcc.Zs + 0.2, elcc.Zs + 0.2], 
# 			       lw=2, lc=:blue, linestyle=:dash, label=false)

# 		p = plot!(p, [sixy0l, sixy0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
# 			       lw=2, lc=:blue, linestyle=:dash, label=false)
	
# 		p = plot!(p, [xyl, xyl], [elcc.Zg, elcc.Za], 	
# 			       lw=1, lc=:red, linestyle=:dash, label=false)
		
# 		p = plot!(p, [xyr, xyr], [elcc.Zg, elcc.Za], 	
# 			       lw=1, lc=:red, linestyle=:dash, label=false)
# 	end
		
# 	xymin = 0
# 	xyl = (elcc.pitch - elcc.d_hole)/2
# 	xy0 = xyl +  elcc.d_hole/2
# 	xyr = xyl + elcc.d_hole
# 	xymax = elcc.pitch
# 	zmax = elcc.Zc
# 	zmin = elcc.Zs 

# 	sixy0l = 0
# 	sixy0r = sixy0l + sipm.sipmSize
# 	sixy1l = sixy0l - sipm.pitch
# 	sixy1r = sixy1l + sipm.sipmSize
# 	sixy2r = sixy0r + sipm.pitch
# 	sixy2l = sixy2r - sipm.sipmSize
	
	
# 	println("xymin= $(xymin), xymax = $(xymax), zmin =$(zmin), zmax =$(zmax)")
# 	println("xyl= $(xyl), xyr = $(xyr), xy0 =$(xy0)")
# 	println("sixy0l= $(sixy0l), sixy0r = $(sixy0r), sixy1l= $(sixy1l), sixy1r = $(sixy1r), sixy2l =$(sixy2l), sixy2r =$(sixy2r)")
	
# 	p1 = plott(1, "Trajectory (x,z)")
# 	p2 = plott(2, "Trajectory (y,z)")
	
#     p1,p2
# end


