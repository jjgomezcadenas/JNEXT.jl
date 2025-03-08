### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7884ab66-f9e2-11ef-03ea-f10faf671dba
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")


# ╔═╡ 108bd997-6d2d-416d-b2df-ec034273d62e
begin
	using PlutoUI
	using CSV
	using DataFrames
	using Plots
	using Printf
	using InteractiveUtils
	using Statistics
	using LinearAlgebra
	using JLD2
end

# ╔═╡ 2617ae05-e1db-473a-9b0f-befeea6d0e12
md"""
# Simulation
"""

# ╔═╡ a891cff0-6910-4f78-8fc5-ff4e90163a7e
begin
	kV = 1e+3
	mm = 1.0
end

# ╔═╡ 3b1d7427-73ca-4dca-99f9-93b2cb6df9a8
struct ELCCGeometry
	X::Float64         # ELCC total x dimension (mm)
	Y::Float64         # ELCC total y dimension (mm)
	Zc::Float64        # Z posiiton of collector (mm)
	Zg::Float64        # Z posiiton of gate (mm)
	Za::Float64        # Z posiiton of anode (mm)
	Zs::Float64        # Z posiiton of SiPM plane (mm)
	Vg::Float64        # potential at gate 
	Va::Float64        # potential at anode 
	d_hole::Float64    # Hole diameter in each dice (mm)
	pitch::Float64     # pitch (mm)	    
end

# ╔═╡ af54e98e-15fb-4ad0-a990-66e183265867
struct SiPMGeometry
	sipmSize::Float64  # Side length of SiPM active area (mm); assumed square.
	pitch::Float64     # Pitch (center-to-center distance) between SiPMs (mm)
	X::Float64         # Overall SiPM panel x dimension (mm)
	Y::Float64         # Overall SiPM panel y dimension (mm)
end

# ╔═╡ 6097b338-4107-4d8e-9ee3-3f806f73c45b
begin
	ndicex(elcc::ELCCGeometry) = Int(floor(elcc.X/elcc.pitch))
	ndicey(elcc::ELCCGeometry) = Int(floor(elcc.Y/elcc.pitch))
	nsipmx(sipm::SiPMGeometry) = Int(floor(sipm.X/sipm.pitch))
	nsipmy(sipm::SiPMGeometry) = Int(floor(sipm.Y/sipm.pitch))
end

# ╔═╡ ac79ab2e-af61-499a-94e7-964a8f04b111
begin

	# ELCSS geometry. The collector is located at 10 mm, the gate at 0 mm and the anode at -5 mm (the SiPMs at -10 mm)
	X = 120.0mm      # ELCC total x dimension (mm)
	Y = 120.0mm      # ELCC total y dimension (mm)
	Zc = 10.0mm     # Z posiiton of collector (mm)
	Zg = 0.0mm      # Z posiiton of gate (mm)
	Za = -5.0mm     # Z posiiton of anode (mm)
	Zs = -10.0mm    # Z posiiton of SiPM plane (mm)
	Vg = -10.0kV  # potential at gate 
	Va = 1kV      # potential at anode 
	d_hole = 3.0mm  # Hole diameter in each dice (mm)
	pitch = 6mm     # pitch
	
end

# ╔═╡ d0bc3255-be2b-4fab-96ed-07c6c96bd0e2
140 * 18 - 116 * 10

# ╔═╡ 321fb432-4464-47b8-94ac-30d466670224
md"""
## ELCC geometry
"""

# ╔═╡ 154133b1-81bd-4bfe-86dc-cb3ccfec48f0
elcc = ELCCGeometry(X, Y, Zc, Zg, Za, Zs, Vg, Va, d_hole, pitch)

# ╔═╡ a0dfd610-50ca-4a75-9ab8-8c3937f31c33
sipm = SiPMGeometry(6.0, 10.0, 120.0, 120.0)

# ╔═╡ 16e4221a-4dd8-4571-8ce2-ef259400562a
nsipmx(sipm)

# ╔═╡ 0aaa3ebf-92f4-4169-bb94-3ba19a70d074
ndicex(elcc)

# ╔═╡ a340f566-c9c0-4293-988e-11b7e69e8e4a
md"""
## Trajectories
"""

# ╔═╡ c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
@load "trajectories_data.jld2" ftrj btrj

# ╔═╡ 7c38062b-0671-451c-911e-f88272f97937
begin
	zftrj = [ftrj[i][1,1:2] for i in range(1, length(ftrj))]
	xmax = maximum([z[1] for z in zftrj])
	xmin = minimum([z[1] for z in zftrj])
	ymax = maximum([z[2] for z in zftrj])
	ymin = minimum([z[2] for z in zftrj])
	md"""
	Trajectories:
	- in X, between $(xmin) and $(xmax)
	- in Y, between $(ymin) and $(ymax)
	
	"""
end

# ╔═╡ 0e31a7b1-e95c-477a-9212-a5a1726370e5
# Example usage:
# Assume that your extended trajectory from simulate_electron_transport3D is stored in `traj_ext`
# and its last row is the impact point on the SiPM plane. Then:
# photon_impact = (traj_ext[end,1], traj_ext[end,3])
# p2 = plot_trajectory_xz(traj_ext, photon_impact)
# display(p2)

# ╔═╡ 63839355-db75-4fd5-b12e-ba348a23d3fc
function propagate(tr, elcc)
	k = 0
	for i in range(1, size(tr)[1])
		xyz = tr[i, :]
		if xyz[3] > elcc.Zg
			continue
		end
		k+=1
		if k >5
			break
		else
		println("i = $(i), k = $(k)  xyz = $(xyz)")
		end
	end
end

# ╔═╡ b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
md"""
# Functions
"""

# ╔═╡ af2b2805-9e6c-4078-9427-02f787212f19
function generate_electron_positions(N::Int, 
	                                 x_min::Float64, x_max::Float64,
	                                 y_min::Float64, y_max::Float64)
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

# ╔═╡ 15552c1d-9bcf-40bc-b14a-c1bb55f63673
electron_xy0 = generate_electron_positions(2000, 0.0, elcc.X, 0.0, elcc.Y)

# ╔═╡ 2273c136-4709-43a6-bf68-1184493fbb70
begin
	i = 120 # electron number to run 
	electron_pos = electron_xy0[i, :]  # an example electron arriving at (x,y) in mm.
	#result = run_simulation(elcc, sipm, electron_absolute)
end

# ╔═╡ d1aec6ee-f530-4a15-9bf9-3081c7f55f4a
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

# ╔═╡ 232fab2c-fd22-449c-be78-f4e55c7021e8
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

# ╔═╡ ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
begin 
	dice_indices, dice_origin, xlocal = find_dice(electron_pos, elcc)
	sipm_indices, sipm_origin =find_sipm(electron_pos, sipm)
end

# ╔═╡ 34ad3d6b-8452-4426-96d6-f004c48b4bae
begin
	xl = collect(xlocal)
	izfr = argmin([norm(xl -zftrj[i]) for i in 1:length(zftrj)])
	println("Electron at $(electron_pos) assigned to dice $(dice_indices) with dice origin $(dice_origin) and local coords $(xlocal)")
	println("sipm indices = $(sipm_indices) with origin $(sipm_origin)")
	println("Closer trajectory number $(izfr), with coordinates $(ftrj[izfr][1,:])")

	end

# ╔═╡ 951ea570-d232-47a3-bbe8-b216de1469a8
begin 
	trjf = ftrj[izfr]
	trjb = btrj[izfr]
	tr = vcat(trjf, trjb)
	md"""
	- Lenght of trajectory $(size(tr))
	"""
end

# ╔═╡ 1bc5d640-7a09-49a6-a1a7-c1ea0c8d1664
size(tr)[1]

# ╔═╡ edde87b6-47d2-4fee-9ec3-62d4d2701c6d
propagate(tr, elcc)

# ╔═╡ 82882beb-98b0-4a53-9f0d-9d16bcbc6c09
"""
Graphical representation of the full structure.
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

# ╔═╡ 79a51e0b-60ec-4ce7-b15b-1f4d88c6aa28
"""
Graphical representation of the full structure.
"""
function plot_structure(elcc::ELCCGeometry, sipm::SiPMGeometry)
	p1, p2 = pstructure(elcc, sipm)
    plot(p1, p2, layout = (1,2))
end

# ╔═╡ dfd7cbf8-adaa-454f-957e-ecc6eee905d3
plot_structure(elcc, sipm)

# ╔═╡ 2b780346-122b-463f-ac4b-498e45dfa84f
function plot_impact_xy(elcc::ELCCGeometry, 
	                sipm::SiPMGeometry, 
	                electron_impact_xy::Vector{Float64})
	p1, p2 = pstructure(elcc, sipm)
	# Plot the electron impact point in red.
    p1 = scatter!(p1, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
		          mc=:red, label=false)
	p2 = scatter!(p2, [electron_impact_xy[1]], [electron_impact_xy[2]], ms=2,
		          mc=:red, label=false)
	plot(p1, p2, layout = (1,2))
end

# ╔═╡ 9eb86c8c-4347-46c4-9111-793f921fac56
plot_impact_xy(elcc, sipm, electron_pos)

# ╔═╡ 60269ab6-d610-409e-90de-48022143ef1e
function pt_lgeom(epos::Vector{Float64}, elcc::ELCCGeometry, sipm::SiPMGeometry)
	
	dice_i, dice_o, xlocal = find_dice(electron_pos, elcc)
	sipm_i, sipm_o =find_sipm(electron_pos, sipm)

	xsipml = sipm_o[1]
	ysipmd = sipm_o[2]
	xdicel = dice_o[1]
	ydiced = dice_o[2]
	xl = minimum([xsipml, xdicel])
	yd = minimum([ysipmd, ydiced])
	zmax = elcc.Zc
	zmin = elcc.Zs 
	
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

	dcx0l = xdicel
	dcx0r = dcx0l + elcc.d_hole
	dcx1l = dcx0l - elcc.pitch
	dcx1r = dcx1l + elcc.d_hole
	dcx2r = dcx0r + elcc.pitch
	dcx2l = dcx2r - elcc.d_hole

	dcy0l = ydiced
	dcy0r = dcy0l + elcc.d_hole
	dcy1l = dcy0l - elcc.pitch
	dcy1r = dcy1l + elcc.d_hole
	dcy2r = dcy0r + elcc.pitch
	dcy2l = dcy2r - elcc.d_hole

	xmin = minimum([six1l, dcx1l])
	xmax = maximum([six2r, dcx2r])
	ymin = minimum([siy1l, dcy1l])
	ymax = maximum([siy2r, dcy2r])
	

	println("six0l= $(six0l), six0r = $(six0r), six1l =$(six1l), six1r =$(six1r), six2l =$(six2l), six2r =$(six2r)")

	println("dcy0l= $(dcy0l), dcy0r = $(dcy0r), dcy1l =$(dcy1l), dcy1r =$(dcy1r), dcy2l =$(dcy2l), dcy2r =$(dcy2r)")

	println("xmin= $(xmin), xmax = $(xmax), ymin =$(ymin), ymax =$(ymax)")
	
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
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx0r, dcx0r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx1l, dcx1l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx1r, dcx1r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx2l, dcx2l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p1 = plot!(p1, [dcx2r, dcx2r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	
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
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy0r, dcy0r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy1l, dcy1l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy1r, dcy1r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy2l, dcy2l], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	p2 = plot!(p2, [dcy2r, dcy2r], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:solid, label=false)
	
	return p1, p2
end
	                     
	

# ╔═╡ 90d67f13-a612-42ec-9a21-fd40d822c17c
function p_trajectory(epos::Vector{Float64}, elcc::ELCCGeometry, 
	                  sipm::SiPMGeometry, tr::AbstractMatrix)
	
	function plott!(i, p, traj)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		plot!(p, traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false)

	
	end
	dice_indices, dice_origin, _ = find_dice(epos, elcc)
	posd = [dice_origin[1], dice_origin[2], elcc.Zc]
	xtrj = [tr[i,:] + [dice_origin[1], dice_origin[2], 0.0] for i in range(1, size(tr)[1]) ] 
	# Concatenate columns then transpose to get an n×3 matrix
	trj = hcat(xtrj...)'   
	p1, p2 = pt_lgeom(epos, elcc, sipm)
	p1 = plott!(1, p1, trj)
	p2 = plott!(2, p2, trj)
	
    plot(p1,p2)
end

# ╔═╡ 2144e49f-1505-4c19-a047-7733c7cfc0c1
p_trajectory(electron_pos, elcc, sipm, tr) 

# ╔═╡ 36c8c1d9-689b-446e-a8d0-83c7b5115944
"""

Plots the electron trajectory in the (x,z) plane and overlays the photon impact 
point (shown in blue) on the SiPM plane.
- `traj` is a matrix whose columns are [x y z] coordinates, and we plot the x and z.
- `photon_impact` is a tuple (x, z) on the SiPM plane.
"""
function plot_trajectory(traj::AbstractMatrix, 
	                     xyl ::ELCCGeometry, sipm::SiPMGeometry)
	
	function plott(i, title)
		println("i = $(i), traj[1,$(i)]=$(traj[1,i])")
		p = plot(traj[:,i], traj[:,3], lw=0.5, lc=:green, label=false,
			  #xlims=(xymin, xymax), ylims = (zmin, zmax),
			  xlims=(sixy1l, sixy2r), ylims = (zmin, zmax),
		      xlabel="x (mm)", ylabel="z (mm)", title=title)

		p = plot!(p, [xyl, xyr], [elcc.Zg, elcc.Zg], 
			       lw=1, lc=:red, linestyle=:dash, label=false)
		p = plot!(p, [xyl, xyr], [elcc.Za, elcc.Za], 
			       lw=1, lc=:red, linestyle=:dash, label=false)

		p = plot!(p, [sixy1l, sixy1r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)
		
		p = plot!(p, [sixy2l, sixy2r], [elcc.Zs + 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)

		p = plot!(p, [sixy0l, sixy0r], [elcc.Zs+ 0.2, elcc.Zs + 0.2], 
			       lw=2, lc=:blue, linestyle=:dash, label=false)
	
		p = plot!(p, [xyl, xyl], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:dash, label=false)
		
		p = plot!(p, [xyr, xyr], [elcc.Zg, elcc.Za], 	
			       lw=1, lc=:red, linestyle=:dash, label=false)
	end
		
	xymin = 0
	xyl = (elcc.pitch - elcc.d_hole)/2
	xy0 = xyl +  elcc.d_hole/2
	xyr = xyl + elcc.d_hole
	xymax = elcc.pitch
	zmax = elcc.Zc
	zmin = elcc.Zs 

	sixy0l = 0
	sixy0r = sixy0l + sipm.sipmSize
	sixy1l = sixy0l - sipm.pitch
	sixy1r = sixy1l + sipm.sipmSize
	sixy2r = sixy0r + sipm.pitch
	sixy2l = sixy2r - sipm.sipmSize
	
	
	println("xymin= $(xymin), xymax = $(xymax), zmin =$(zmin), zmax =$(zmax)")
	println("xyl= $(xyl), xyr = $(xyr), xy0 =$(xy0)")
	println("sixy0l= $(sixy0l), sixy0r = $(sixy0r), sixy1l= $(sixy1l), sixy1r = $(sixy1r), sixy2l =$(sixy2l), sixy2r =$(sixy2r)")
	
	p1 = plott(1, "Trajectory (x,z)")
	p2 = plott(2, "Trajectory (y,z)")
	
    plot(p1,p2)
end

# ╔═╡ 23d039f4-b1db-4778-be0b-2fa01075a1a2
plot_trajectory(tr, elcc, sipm)

# ╔═╡ f1be2273-d0a9-47f1-a92c-d151d9f9bd5f
"""
Once we know the electron's local coordinates in a dice, 
return a pre-generated trajectory that will take the electron through the hole.
For simplicity, assume a straight-line trajectory along z through the dice center.
The trajectory goes from the ELCC surface (say, z = ELCC top = elcc.Z) to the SiPM plane.
"""
function get_trajectory_in_hole(dice_origin, elcc::ELCCGeometry)
    diceWidth = elcc.X / elcc.nDiceX
    diceHeight = elcc.Y / elcc.nDiceY
    # Hole is assumed to be centered in the dice.
    hole_center_local = (diceWidth/2, diceHeight/2)
    # Define a trajectory as a set of points.
    # For simplicity, assume 5 steps from ELCC top to the SiPM plane.
    z_start = elcc.Z   # top of ELCC
    z_end = elcc.Zsipm  # SiPM plane (assume it's at z = Zsipm, below the ELCC)
    n_steps = 5
    traj = zeros(Float64, n_steps, 3)
    for k in 1:n_steps
        t = (k-1)/(n_steps-1)
        # The x and y coordinates follow a straight line from the electron's initial local position
        # to the center of the hole. For simplicity we assume all electrons inside the hole are directed to the center.
        x_local = hole_center_local[1]
        y_local = hole_center_local[2]
        z_val = (1-t)*z_start + t*z_end
        # Convert local dice coordinate to absolute coordinate.
        traj[k, :] = [dice_origin[1] + x_local, dice_origin[2] + y_local, z_val]
    end
    return traj
end

# ╔═╡ 9d86737d-1009-4847-84ee-b5073408acde
"""
Simulate photons along the trajectory.

At each step along the trajectory, generate a constant number of photons.
Propagate them along the same straight line (they follow the electron path).
Count photons that hit a SiPM (if the impact falls within a sensor active area).
"""
function simulate_photons_along_trajectory(traj::AbstractMatrix, sipm::SiPMGeometry; N_photons_per_step=10)
    # Assume photons are generated at each step along the trajectory.
    # For simplicity, we assume that all photons follow the electron path.
    # The final impact position on the SiPM plane is the last point of the trajectory.
    impact = traj[end, :]
    # Determine which SiPM (if any) is hit.
    # We assume the SiPM plane is aligned with the ELCC, covering the same area (0 to sipm.X, 0 to sipm.Y).
    # Sensors are squares of side sipm.sipmSize centered in cells of size sipm.pitch.
    # Compute sensor indices from the impact (x,y) coordinates.
    x = impact[1]
    y = impact[2]
    # Map x and y into sensor cell indices:
    i_sensor = floor(Int, x / sipm.pitch) + 1
    j_sensor = floor(Int, y / sipm.pitch) + 1
    # Compute sensor cell center:
    cx = (i_sensor - 1)*sipm.pitch + sipm.pitch/2
    cy = (j_sensor - 1)*sipm.pitch + sipm.pitch/2
    # Check if the impact lies within the active area (sensor size):
    if abs(x - cx) <= sipm.sipmSize/2 && abs(y - cy) <= sipm.sipmSize/2
        # All photons hit this sensor.
        return (i_sensor, j_sensor, N_photons_per_step * size(traj, 1))
    else
        # Impact falls between sensors.
        return nothing
    end
end

# ╔═╡ a91f3f93-ffec-47fc-819e-e4f43bee7f95
"""
Simulation 
    
electron_pos: absolute (x,y) position where the electron arrives on the ELCC.

Determine dice assignment and local coordinates.
"""
function run_simulation(elcc::ELCCGeometry, sipm::SiPMGeometry, electron_pos::Tuple{Float64,Float64})

    dice_indices, dice_origin, xlocal = find_dice(electron_pos[1], electron_pos[2], elcc)
    println("Electron at $(electron_pos) assigned to dice $(dice_indices) with dice origin $(dice_origin) and local coords $(xlocal)")
    
    # Get a trajectory for the electron (assuming it follows a straight line through the hole).
    traj = get_trajectory_in_hole(dice_origin, elcc)
    
    # Simulate photon generation along the trajectory.
    photon_info = simulate_photons_along_trajectory(traj, sipm; N_photons_per_step=10)
    if photon_info !== nothing
        println("Photon hit on SiPM at index $(photon_info[1]), $(photon_info[2]) with total photons = $(photon_info[3])")
    else
        println("Photon impact was lost (did not hit an active SiPM).")
    end
    
    return (traj, dice_indices, dice_origin, xlocal, traj, photon_info)
end

# ╔═╡ 569026e1-b82f-4230-b8f5-4fe60afd2cb7


# ╔═╡ Cell order:
# ╠═7884ab66-f9e2-11ef-03ea-f10faf671dba
# ╠═108bd997-6d2d-416d-b2df-ec034273d62e
# ╠═2617ae05-e1db-473a-9b0f-befeea6d0e12
# ╠═a891cff0-6910-4f78-8fc5-ff4e90163a7e
# ╠═3b1d7427-73ca-4dca-99f9-93b2cb6df9a8
# ╠═6097b338-4107-4d8e-9ee3-3f806f73c45b
# ╠═af54e98e-15fb-4ad0-a990-66e183265867
# ╠═ac79ab2e-af61-499a-94e7-964a8f04b111
# ╠═d0bc3255-be2b-4fab-96ed-07c6c96bd0e2
# ╠═321fb432-4464-47b8-94ac-30d466670224
# ╠═154133b1-81bd-4bfe-86dc-cb3ccfec48f0
# ╠═a0dfd610-50ca-4a75-9ab8-8c3937f31c33
# ╠═dfd7cbf8-adaa-454f-957e-ecc6eee905d3
# ╠═16e4221a-4dd8-4571-8ce2-ef259400562a
# ╠═0aaa3ebf-92f4-4169-bb94-3ba19a70d074
# ╠═a340f566-c9c0-4293-988e-11b7e69e8e4a
# ╠═c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
# ╠═7c38062b-0671-451c-911e-f88272f97937
# ╠═15552c1d-9bcf-40bc-b14a-c1bb55f63673
# ╠═2273c136-4709-43a6-bf68-1184493fbb70
# ╠═9eb86c8c-4347-46c4-9111-793f921fac56
# ╠═ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
# ╠═34ad3d6b-8452-4426-96d6-f004c48b4bae
# ╠═951ea570-d232-47a3-bbe8-b216de1469a8
# ╠═2144e49f-1505-4c19-a047-7733c7cfc0c1
# ╠═23d039f4-b1db-4778-be0b-2fa01075a1a2
# ╠═0e31a7b1-e95c-477a-9212-a5a1726370e5
# ╠═1bc5d640-7a09-49a6-a1a7-c1ea0c8d1664
# ╠═63839355-db75-4fd5-b12e-ba348a23d3fc
# ╠═edde87b6-47d2-4fee-9ec3-62d4d2701c6d
# ╠═b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
# ╠═af2b2805-9e6c-4078-9427-02f787212f19
# ╠═d1aec6ee-f530-4a15-9bf9-3081c7f55f4a
# ╠═232fab2c-fd22-449c-be78-f4e55c7021e8
# ╠═82882beb-98b0-4a53-9f0d-9d16bcbc6c09
# ╠═79a51e0b-60ec-4ce7-b15b-1f4d88c6aa28
# ╠═2b780346-122b-463f-ac4b-498e45dfa84f
# ╠═60269ab6-d610-409e-90de-48022143ef1e
# ╠═90d67f13-a612-42ec-9a21-fd40d822c17c
# ╠═36c8c1d9-689b-446e-a8d0-83c7b5115944
# ╠═f1be2273-d0a9-47f1-a92c-d151d9f9bd5f
# ╠═9d86737d-1009-4847-84ee-b5073408acde
# ╠═a91f3f93-ffec-47fc-819e-e4f43bee7f95
# ╠═569026e1-b82f-4230-b8f5-4fe60afd2cb7
