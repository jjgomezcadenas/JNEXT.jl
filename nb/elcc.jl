### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 56471816-de2f-11ef-0099-9b1a882f96c8
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

# ╔═╡ 5769b663-7790-45eb-b5b1-dd513a284458
begin
	using PlutoUI
	using CSV
	using DataFrames
	using Images
	using Plots
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
	using CategoricalArrays
end

# ╔═╡ 2dd7f02d-73f6-45c2-a904-482efdd56b28
import Unitful:
    nm, μm, mm, cm, m, km,
    mg, g, kg,
    fs, ps, ns, μs, ms, s, minute, hr, d, yr, Hz, kHz, MHz, GHz,
    eV, keV, MeV,
    μJ, mJ, J,
	μW, mW, W,
    A, N, mol, mmol, V, L, M

# ╔═╡ d57ed4ef-8b54-4b55-9c56-47345aea1a26
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(m,
        Expr(:toplevel,
                :(eval(x) = $(Expr(:core, :eval))($name, x)),
                :(include(x) = $(Expr(:top, :include))($name, x)),
                :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
                :(include($path))))
    m
end

# ╔═╡ 775d3ebb-4e86-4e91-9954-628cbec2529b
jn = ingredients("../src/JNEXT.jl")

# ╔═╡ 0f23f280-f816-4aca-9c20-5425e4641b8b
load("elcc.jpeg")

# ╔═╡ a32930ae-120f-4f6e-8e6f-300580ec9ade
begin
	Wi    = 21.9 # in eV
	pTPC  = 13.0   # in bar
	dh    = 6.0    # in mm
	ph    = 10.0   # in mm
	tpc = jn.JNEXT.Cylinder(490*mm, 1200*mm)
	rtpc = tpc.R/mm # radius in mm
	ltpc = tpc.L/mm # length in mm

	md"""
	## ELCC study
	- Dimensions of TPC: R = $(rtpc) mm, L = $(ltpc), pressure = $(pTPC) bar
	- ELCC: dh = $(dh) mm, SiPM = $(dh) mm, pitch = $(ph) mm
	- Wi = $(Wi) eV
	
	"""
end

# ╔═╡ aabb2413-4512-4c39-8070-5e853c9c0127
md"""
edf.x is a vector with the position in x of the electron 
"""


# ╔═╡ 55be668d-4af3-4d32-b048-495ab020e113
md"""
xn_ev is a vector of vectors. For each position in edf.x, one takes the energy deposition in this position and generates the corresponding number of ionisation electrons, all in the same position. Then each electron is transported to the ELCC, and its position, after diffusion recorded. Thus, for each xi position in edf.x, we have a vector [xeli] with the positions of the ni electrons generated and transported for this point.

"""


# ╔═╡ 2cb0bcef-1892-4b3b-9a4b-89c290eb57b1
md"""
xmean\_ev takes the mean of each element of xn\_ev. Thus, it has the average of the position of the electrons in the ELCC, which is distributed around the initial position with diffusion Dl and Dt. 

"""


# ╔═╡ 5f1de077-147b-4fe1-84b3-0327d795214e
md"""
### Add the two points
"""

# ╔═╡ 24d3a2bc-3406-4705-9eaa-a3244c9b9244
md"""
Consider the following geometrical structure. a) A slab of transverse dimensions X,Y and thickness Z that we call "ELCC". b) A plane of dimensions X,Y, located at Z+Zsipm from ELCC. The ELCC is divided in "dices" of transverse dimension (x,y). In the center of each of these regions there is a hole of diameter d. For example, (x,y) = (4.5, 4.5) (mm) with a hole of diameter 3 mm in the center such that the pitch between hole and hole (pd) is 6 mm. In this example, X, Y = 90, 90, so that the ELCC is divided in 20 x 20 dices (and thus 20 x 20 holes). The thickness of the ELCC is 5 mm. The SiPM pane is located at Zsipm = 5mm from the bottom of the ELCC. The plane is composed of SiPMs of 6x6 mm2 at a pitch (from center of one SiPM to the next one in each dimension) of 9 mm. I need you to write julia code to achieve the following tasks. a) represent graphically the full structure; b) provide the ability to store information. For example. 1. An electron arrives to the ELCC. 2. We need to find the coordinates of the dice (in the absolute reference system) to which the electron is assigned, and the coordinates xe, ye, in the system of reference of the dice (that goes from 0 to l in both x, y, where l is the length of the dice. 3. One we know the position xe, ye inside the dice, we pick a pre-generated trayectory that will make the electron go through the hole. 4. While the electron follows the trayectory we will generate photons that will propagate inside the hole until they reach the SiPM plane. 5. We need then to count the number of photons that hit each SiPM (those who pass in the space between SiPMs are lost). For the example, assume that all trajectories simply take the electron through the center of the hole, following a straigh line along the z (while in the hole). Generate a constant number of photons in a number of steps (say 5 steps) inside the hole, and propagate the photons such that photons that move backward or hit the hole walls are lost.  
"""

# ╔═╡ 86149e81-ed71-4c57-aeb8-fe525be4ad64
md"""
## Event analysis
"""

# ╔═╡ 8d1e1d84-a7d0-44e0-8ddf-2dc3b2f43add
md"""
- Average number of electrons for each SiPM
"""

# ╔═╡ 921e6046-ff59-43d2-a834-a74320331ef9
md"""
- Maximum number of electrons for each SiPM
"""

# ╔═╡ 2ceb0f10-33c1-45fc-a5a8-e456f481f3ac
md"""
- Sum of  all electrons for each SiPM
"""

# ╔═╡ 8fd4e626-f8f4-4541-b520-19fe8beeef1c
begin
	gg = 1e+3
	eps = 0.3
	PDE = 0.65 * 0.5 * 0.25  # TPB QE x geometry x PDE for S13360-25 Hamamatsu
	ncl = 57600 # number of cells for S13360-25 Hamamatsu 
	gt = gg * eps * PDE
md"""
### Electrons to photoelectrons
- In the ELCC channel, each electron produces ``n_\gamma = g \times n_e`` where g is a gain
- The ELCC channel has an optical efficiency ``\epsilon``
- The SiPM has an effective PDE which depends on the SiPM type. 
- Taking g =$(gg), epsilon = $(eps) and PDE = $(PDE), the total gain is $(gt)
"""
end

# ╔═╡ dde3b9ff-3f28-49c8-b3cb-efe9f9827e85
md"""
## ELCC Algorithm
"""

# ╔═╡ df6e502e-7892-4158-bfb7-cc38214618a1
md"""
## Description of detector
"""

# ╔═╡ f7b2fbef-f189-4c4e-b7ce-ee46688d530d
"""
Range of an adc with nbits
"""
adcrange(nbits) = 2^nbits -1

# ╔═╡ 4c58b596-9da3-4e31-ae3e-5acf18e206a4
"""
maximum number of pes for an ADC of nbits
"""
maxpes(nbits, adctopes) = adcrange(nbits)/adctopes

# ╔═╡ ae4ae06f-a756-4cf6-94d0-d08dac29ed48
begin
	nbits = 13
	adctopes = 11
	mxp = round(maxpes(nbits, adctopes))
md"""
### Saturation
- Using a SiPM of 25 μm pitch, SiPM does not saturate and there is enough light for good resolution. 
- Instead, ADC saturation may become a problem.
- Assuming an ADC of $(nbits) with a LSE (adc to pes) of $(adctopes) we obtain a maximum number of pes equal to $(mxp) 
"""
end

# ╔═╡ a442b7d0-9035-48a9-bbce-98c66c09e05f
"""
Computes the fraction of SiPMs which have at least one saturated sample.
- ts is the vector containing the time series with the number of pes per sipm
- mxp is the maximum number of pes for a given adc
- gt is the total gain 
"""
function fraction_saturated_sipm(ts,mxp,gt)
	ns = 0
	for spm in ts
		npes = spm * gt 
		if any(x -> x > mxp, npes) 
			ns+=1
		end
	end	
	return ns/length(ts)
	
end

# ╔═╡ c446556a-239d-4e95-a3e7-3f7f3679e038
md"""
## Functions
"""

# ╔═╡ a01e9701-dce9-496e-85f9-cc511fb274ed
function sumts(ts)
	lx = length(ts[1])
	vpes = zeros(lx)
	
	for (i, v) in enumerate(ts)
		vpes = vpes .+ v
	end
	vpes
end

# ╔═╡ b9c13dfc-614d-4f95-9a71-8c3152a0c5fd
"""
Returns zxpes (a matrix of sipmx x tbins) and zypes (sipmy x tbins)
"""
function mat_hxz(hlx, ts)
	lx = length(hlx[1,:])
	px = Int(sqrt(lx))
	xx = zeros(px)
	yy = zeros(px)
	tl = length(ts[1])
	zxpes = zeros(px, tl)
	zypes = zeros(px, tl)
	
	
	for (i,k) in enumerate(1:px:lx)
		#println("i =$(i), k = $(k)")
		
		xx[i] = hlx[1,k]
		
		#println(" xs = $(xx[i])")
		
		vsum = zeros(tl)
		is = px*(i-1) + 1
		ie = is + px -1
		
		#println(" is = $(is), ie = $(ie)")
		for m in is:ie
			#println(" m = $(m)")
			#println(" max ts[$(m)] = $(maximum(ts[m]))")
			vsum = vsum .+ ts[m]
		end
		zxpes[i,:] = vsum
		#println(" vpes = $(vpes[i])")
	end

	for i in 1:px
		#println("i =$(i)")
		
		yy[i] = hlx[2,i]
		
		#println(" ys = $(yy[i])")
		
		vsum = zeros(tl)
		is = i
		ie = lx-px+(i-1)
		
		#println(" is = $(is), ie = $(ie)")
		for m in is:px:ie
			#println(" m = $(m)")
			#println(" max ts[$(m)] = $(maximum(ts[m]))")
			vsum = vsum .+ ts[m]
		end
		zypes[i,:] = vsum
		#println(" vpes = $(vpes[i])")
	end

	xx, yy, zxpes, zypes
end

# ╔═╡ 9ed19e6c-2f9b-4d16-940f-6269427ac8ee
"""
Compute the mean, max and total over the time series for each SiPM
"""
function statsts(ts)
	emax  = zeros(length(ts))
	emean = zeros(length(ts))
	etot  = zeros(length(ts))
	for (i, sipm) in enumerate(ts)
		emax[i] = maximum(sipm)
		emean[i] = mean(sipm)
		etot[i] =sum(sipm)
	end
	emax, emean, etot
end

# ╔═╡ 8b4ddbbf-a22b-4249-8980-630bc2b9bf2b
"""
For each SiPM add the time series for all the points in the event
"""
function add_time_series(txsipm)
	npoints = length(txsipm)
	nsi = length(txsipm[1])
	nst = length(txsipm[1][1])
	
	# Initialize the result vector with zeros for each SiPM's time series
	sadd = [zeros(nst) for _ in 1:nsi]

	# Loop over all events and accumulate the time series for each SiPM
	for point in txsipm
		for (sipm_idx, times) in enumerate(point) 
        	sadd[sipm_idx] .+= times
		end
    end
	sadd
end

# ╔═╡ 6ba99a14-a13e-44ea-a873-4597c9b0fd13
"""
Transports drift electrons in (x,y) to the nearer hole in the ELCC.
"""
function transport_xy_to_sipm(xx, yy, zz, holes, tbins)
	
	# Build k-d Tree for Nearest Neighbor Search
	kdtree = KDTree(holes)
	k = 2

	@assert length(xx) == length(yy)
	# find nearest 
	# Electron ends up in the hole nearest to it. 
	#sipm  = zeros(1, size(holes)[2])
	sipm  = zeros(size(holes)[2])
	tsipm = Vector{Vector{Float64}}(undef, size(holes)[2])

	#for i in 1:size(holes)[2]
	#	sipm[1,i] = holes[1, i] # xsipm = xhole
	#	sipm[2,i] = holes[2, i] # ysipm = yhole
	#end
		
	for i in 1:length(xx)
		pc = zeros(2)
		pc[1] = xx[i]
		pc[2] = yy[i]
	
		#println("xc =$(pc[1]),yc =$(pc[2])")

		idxs, dists = knn(kdtree, pc, k, true)
		
		#sipm[1,idxs[1]] = sipm[1,idxs[1]] + 1
		sipm[idxs[1]] = sipm[idxs[1]] + 1
	end

	for i in 1:length(sipm)
		#tsipm[i] = sipm[3,i] .* tbins
		tsipm[i] = sipm[i] .* tbins
	end
	
	sipm, tsipm
	
end

# ╔═╡ b38a1a09-00ea-4ad4-ac9d-c74a5a1f7700
function transport_to_el(hitdf, boxtpc, w=21.9, p=13)

	# express dl in cm
	dl = ustrip.(uconvert.(u"cm", tpc.L .- hitdf.z * mm))
	
	
	# number of drift electrons per energy cluster
	nel = round.(Int, hitdf.energy*1e+6/w) .+ 1  # energy in MeV, w in eV 

	# pressure p in bar, DL and DT in mm
	# DL = 1mm sqrt(b) / sqrt(cm)
	DL = sqrt(p) .* sqrt.(dl)
	DT = 0.3*sqrt(p) .* sqrt.(dl)

	# DL and DT are the coefficients of a Gaussian distribution
	sigmax = DT /sqrt(2) # in sqrt(mm)

	xn = Vector{Vector{Float64}}(undef, length(hitdf.x))
	yn = Vector{Vector{Float64}}(undef, length(hitdf.y))
	zn = Vector{Vector{Float64}}(undef, length(hitdf.z))
	for (i,x) in enumerate(hitdf.x)
		dx = Normal(0, sigmax[i])
		dz = Normal(0, DL[i])
		xx = x .+ rand(dx, nel[i]) # in mm
		
		#println("x = $(x), nel = $(nel[i]), <xx> = $(mean(xx)) ")

		yy = hitdf.y[i] .+ rand(dx, nel[i]) # in mm
		#zz = hitdf.z[i] .+ rand(dz, nel[i]) # in mm
		zz =  tpc.L/mm .+ rand(dz, nel[i]) # Take EL as z = L
		xn[i] = xx
		yn[i] = yy
		zn[i] = zz
	end

	xn, yn, zn
end

# ╔═╡ 166dd216-79be-4974-bab9-b2a64bd2d817
function get_xyz_point(xevt, yevt, zevt, i)
	xx = xevt[i]
	yy = yevt[i]
	zxmin = minimum(zevt[i])
	zz = zevt[i] .- zxmin
	xx,yy,zz
end
	


# ╔═╡ 79b86295-13e1-4277-8dc0-55106ba2e887
function get_dfs(filename::String, group::String)

	fid     = h5open(filename, "r")
	dset =fid["MC"][group]
	dda = read(dset)
	ddic = Dict()
    for (i, key) in enumerate(keys(dda[1]))
        ddic[key] = [ddi[i] for ddi in dda]
    end
	close(fid)
    DataFrame(ddic)
end


# ╔═╡ 4e0634f1-3b46-47fe-b742-a815ea1aef4e
function get_particles(fpath::String)
	pardf = get_dfs(fpath, "particles")
	pardf = pardf[pardf.creator_proc .== "none", :]
	pardf = select(pardf, :event_id, :final_x, :final_y, :final_z, :initial_momentum_x, :initial_momentum_y, :initial_momentum_z, :initial_x, :initial_y, :initial_z, :kin_energy, :length, :particle_id)
	rename!(pardf, Dict(:final_x => :xf, :final_y => :yf, :final_z=> :zf, :initial_momentum_x=> :px, :initial_momentum_y=> :py, :initial_momentum_z=> :pz,:initial_x => :x0, :initial_y => :y0, :initial_z=> :z0, :kin_energy=> :ekin,  :particle_id => :pid))
end

# ╔═╡ cc1ac654-e7b0-4441-ae9e-6faddb9cfbb0
begin
	pathtodata = "/Users/jjgomezcadenas/Data/nexus13bar/"
	filname = "nexus_9_0nubb.h5"
	pardf =get_particles(joinpath(pathtodata, filname))
end

# ╔═╡ 790948af-bed5-4f68-a52a-de7d6af932fb
function get_hits(fpath::String)
	pardf = get_dfs(fpath, "hits")
	#pardf = pardf[pardf.creator_proc .== "none", :]
	pardf = select(pardf, :energy, :event_id, :hit_id, :particle_id, :x, :y, :z)
	#rename!(pardf, Dict(:final_x => :xf, :final_y => :yf, :final_z=> :zf, :initial_momentum_x=> :px, :initial_momentum_y=> :py, :initial_momentum_z=> :pz,:initial_x => :x0, :initial_y => :y0, :initial_z=> :z0, :kin_energy=> :ekin,  :particle_id => :pid))
end

# ╔═╡ ce05f403-d0bf-41e9-bcc7-f6ed7768bf71
hitdf = get_hits(joinpath(pathtodata, filname))

# ╔═╡ 2a0ffd4a-0a8a-45a3-8f98-3893f8628ce4
function select_event(df, event)
	df[df.event_id .== event, :]
end


# ╔═╡ 5ebd4bf9-b53f-4610-8260-0c42b302a6ea
function get_dimensions(pardf)
	xminf = minimum(pardf.xf)
	xmaxf = maximum(pardf.xf)
	yminf = minimum(pardf.yf)
	ymaxf = maximum(pardf.yf)
	zminf = minimum(pardf.zf)
	zmaxf = maximum(pardf.zf)
	xmin0 = minimum(pardf.x0)
	xmax0 = maximum(pardf.x0)
	ymin0 = minimum(pardf.y0)
	ymax0 = maximum(pardf.y0)
	zmin0 = minimum(pardf.z0)
	zmax0 = maximum(pardf.z0)
	return minimum((xminf,xmin0)), maximum((xmaxf, xmax0)),
	minimum((yminf,ymin0)), maximum((ymaxf, ymax0)),
	minimum((zminf,zmin0)), maximum((zmaxf, zmax0))
end

# ╔═╡ 91d39551-02db-42da-9eca-f9623506e761
begin
	xmin, xmax, ymin, ymax, zmin, zmax = get_dimensions(pardf)
	boxtpc = jn.JNEXT.Box(xmin*mm, xmax*mm, ymin*mm, ymax*mm, zmin*mm, zmax*mm)

md"""

Data:

xmin = $( @sprintf("%.2f", xmin) ) mm, xmax = $( @sprintf("%.2f", xmax) ) mm

ymin = $( @sprintf("%.2f", ymin) ) mm, ymax = $( @sprintf("%.2f", ymax) ) mm

zmin = $( @sprintf("%.2f", zmin) ) mm, zmax = $( @sprintf("%.2f", zmax) ) mm

"""
end

# ╔═╡ 69b72ac6-c947-44bf-a10f-6aa7726a7a10
function mean_diff(xn)
	std0 = [std(xc) for xc in xn] 
	xstd = [isnan(x) ? 0.0 : x for x in std0]
	
	[mean(xc) for xc in xn], xstd
end

# ╔═╡ 08935cb7-3f21-4e8e-92a1-a8d5e60254f6
begin
	event_number         = 4488
	edf                  =select_event(hitdf, event_number) 
	nel_ev               = edf.energy * 1E+6/Wi
	xn_ev, yn_ev, zn_ev  = transport_to_el(edf, boxtpc)
	xmean_ev, xstd_ev    = mean_diff(xn_ev)
	ymean_ev, ystd_ev    = mean_diff(yn_ev)
	zmean_ev, zstd_ev    = mean_diff(zn_ev)
	md"""

	### Event number = $(event_number)
	"""
	
end

# ╔═╡ c255e0b0-bfe1-49e7-9259-ba87ce4ec55f
edf

# ╔═╡ 2e3cb152-50cb-42e7-a02d-7533e781686d
edf.x

# ╔═╡ 94f49857-c6b4-4991-8846-d57b93fd0901
xn_ev

# ╔═╡ a65da87a-0689-48c0-816d-c415c71d6bc9
xmean_ev

# ╔═╡ bcb2c7e6-cb17-4048-86b7-3a854fc24629
begin
	pt1 = 1
	
	md"""
	### Transport point $(pt1)  of event $(event_number)
	"""
end

# ╔═╡ c49b70bc-1e09-4571-b9a4-31cd5e303db9
begin
	xx, yy, zz  = get_xyz_point(xn_ev, yn_ev, zn_ev, pt1) 
	zhst = Hist1D(zz,binedges = 1:201)
	plot(zhst)
end

# ╔═╡ 6e1e25bf-5c11-470c-a739-1214243c8438
begin
	tbins = bincounts(zhst)./nentries(zhst)
	plot(tbins, xlabel="time bin", ylabel="weight")
end

# ╔═╡ a84d9f3f-ba81-4817-8f66-97c8c65adb5e
md"""
#### tbins
- tbins is a vector of weights that is used to distribute the events recorded by a SiPM in a time series. It is obtained by taking the times of arrival to the ELCC (normalizing to the first electron arriving) and binning in $(length(tbins)) bins (1mm = 1 μs)
"""

# ╔═╡ 99c6d9e7-30f5-4dd3-a95d-14afa511bc53
begin
	pt2 = 10
	
	md"""
	### Transport point $(pt2)  of event $(event_number)
	"""
end

# ╔═╡ 8a8d1789-c1cb-4c47-a578-da926bf34edc
begin
	
	xx2, yy2, zz2  = get_xyz_point(xn_ev, yn_ev, zn_ev, pt2) 

	zhst2 = Hist1D(zz2,binedges = 1:201)
	plot(zhst2)
end

# ╔═╡ ce84c2a4-53d7-4bc7-871e-51aea25093b8
begin
	tbins2 = bincounts(zhst2)./nentries(zhst2)
	plot(tbins2, xlabel="time bin", ylabel="weight")
end

# ╔═╡ bbd23f58-592d-4c96-b2e6-9449396da78f
function display2d(h1, h2, 
	               title1, label1, alpha1, title2, label2, alpha2;
				   size=(900,400))
	# Create separate plots for each histogram
	p1 = plot(h1, title=title1,  alpha=alpha1)
	p2 = plot(h2, title=title2,  alpha=alpha2)

	# Combine p1 and p2 into subplots with layout=(1,2)
	plot(p1, p2, 
    layout=(1,2),  # 1 row, 2 columns
    size=size) # optional: control figure size

end

# ╔═╡ 9695043a-7610-4827-ae47-f600768bf939
begin
	hek = Hist1D(pardf.ekin; binedges = 0:0.2:3);
	hl = Hist1D(pardf.length; binedges = 0:5:200);
	display2d(hek, hl, 
	      "Kinetic energy (MeV)", "bb0nu", 0.5, 
	      "Length (mm)", "bb0nu", 0.5, 
		  size=(900,400))
end

# ╔═╡ bd3638dc-4916-4337-8e7c-f9dda76b6928
begin
	hxy = Hist2D((pardf.x0,pardf.y0));
	hxyf = Hist2D((pardf.xf,pardf.yf));
	display2d(hxy, hxyf, 
	      "X vs Y  (initial, mm)", "bb0nu", 1.0, 
	      "X vs Y  (final, mm)", "bb0nu", 1.'', 
		  size=(900,400))
end

# ╔═╡ 38ff4567-8d5c-4925-994c-22586c8f0ea6
begin
	hxz = Hist2D((pardf.x0,pardf.z0));
	hxzf = Hist2D((pardf.xf,pardf.zf));
	display2d(hxz, hxzf, 
	      "X vs Z  (initial, mm)", "bb0nu", 1.0, 
	      "X vs Z  (final, mm)", "bb0nu", 1.'', 
		  size=(900,400))
end

# ╔═╡ 2b0d31f0-69c3-4a2c-9d6c-7e96d6393571
function display4d(h1, h2, h3, h4,
	               title1, label1, alpha1, 
				   title2, label2, alpha2,
				   title3, label3, alpha3, 
				   title4, label4, alpha4;
				   size=(900,400))
	# Create separate plots for each histogram
	p1 = plot(h1, title=title1,  alpha=alpha1)
	p2 = plot(h2, title=title2,  alpha=alpha2)
	p3 = plot(h3, title=title3,  alpha=alpha3)
	p4 = plot(h4, title=title4,  alpha=alpha4)

	# Combine p1 and p2 into subplots with layout=(1,2)
	plot(p1, p2, p3, p4,
    layout=(2,2),  # 2 row, 2 columns
    size=size) # optional: control figure size

end

# ╔═╡ a5f74f36-a089-4123-8d60-cddf2af6a697
display4d(Hist1D(xstd_ev), Hist1D(ystd_ev), Hist1D(zstd_ev), Hist1D(nel_ev),
	             "std Dtx", " ", 0.5,
	             "std Dty", " ", 0.5,
		         "std Dtz", " ", 0.5,
				 "nel ", " ", 0.5;
				 size=(900,400))

# ╔═╡ a62a307b-28e9-4b15-ac5e-ce2d4745a5cc
function plot_event(boxtpc::jn.JNEXT.Box, 
                    eventid::Int64, 
	                xd::AbstractVector{<:AbstractFloat},
	                yd::AbstractVector{<:AbstractFloat},
	                zd::AbstractVector{<:AbstractFloat};
                    markersize::Int64=1, zoom::Bool=false, 
                    size::Tuple{Int64, Int64}=(900,400))

	if zoom
		xmin = minimum(xd)
		xmax = maximum(xd)
		ymin = minimum(yd)
		ymax = maximum(yd)
		zmin = minimum(zd)
		zmax = maximum(zd)
	else
		xmin = boxtpc.xmin/mm
		xmax = boxtpc.xmax/mm
		ymin = boxtpc.ymin/mm
		ymax = boxtpc.ymax/mm
		zmin = boxtpc.zmin/mm
		zmax = boxtpc.zmax/mm
	end

	
	s3d = scatter3d(xd, yd, zd, 
		            markersize=markersize, 
		            label="xyz, event=$(eventid)",
	                xlims = (xmin, xmax),
                    ylims = (ymin, ymax),
                    zlims = (zmin, zmax))
	sxy = scatter(xd, yd, 
		            markersize=markersize, 
		            label="xy, event=$(eventid)",
		            xlims = (xmin, xmax),
                    ylims = (ymin, ymax))
	sxz = scatter(xd, zd, 
		          markersize=markersize, label="xz, event=$(eventid)",
		          xlims = (xmin, xmax),
                  ylims = (zmin, zmax))
	syz = scatter(yd, zd, 
		          markersize=markersize, label="yz, event=$(eventid)",
		          xlims = (ymin, ymax),
                  ylims = (zmin, zmax))
	
	plot(s3d, sxy,sxz,syz,  
    layout=(2,2),  # 1 row, 2 columns
    size=size)
end

# ╔═╡ 6831ccc5-3607-4ace-9f37-da203502b1b4
begin
	                 
	plot_event(boxtpc, edf.event_id[1], edf.x, edf.y, edf.z;
		       markersize=1, zoom=true, size=(900,400))
end

# ╔═╡ bcb22069-18eb-44aa-ba00-26244be4e3fb
plot_event(boxtpc, edf.event_id[1], xmean_ev, ymean_ev, zmean_ev;
		       markersize=1, zoom=true, size=(900,400))

# ╔═╡ 68f875be-3126-4041-90a8-14403b8def72
"""
Algorithm to Generate Hole Positions
"""
function generate_hole_positions(xmin::AbstractFloat, 
	                             xmax::AbstractFloat, 
	                             ymin::AbstractFloat, 
	                             ymax::AbstractFloat, 
	                             p::AbstractFloat, 
	                             d::AbstractFloat)
	
    # Calculate number of holes along X and Y
    nx = floor(Int, (xmax - xmin + 2 * p) / p) #+ 1
    ny = floor(Int, (ymax - ymin + 2 * p) / p) #+ 1
    
	#println("nx = $(nx), ny = $(ny)")
	nn = maximum((nx,ny))
	hxy = zeros(2, nn*nn)

	n=1
	for i in 1:nn
		for j in 1:nn
			hxy[1,n] = xmin - p + (i-1)*p
			hxy[2,n] = ymin - p + (j-1)*p
			n+=1
		end
	end

	
    hxy
end





# ╔═╡ b59aa31a-c222-457c-8b46-73d745173885
begin
	# Define Surface and Hole Parameters
	
	xbmin = minimum(xmean_ev)
	xbmax = maximum(xmean_ev)
	ybmin = minimum(ymean_ev)
	ybmax = maximum(ymean_ev)
	
	# Generate Hole Positions
	holes = generate_hole_positions(xbmin, xbmax, ybmin, ybmax, ph, dh)
	
	nh = size(holes)[2]
	
	md"""

	### ELCC Detection 
	
	Dimensions of ELCC detection area for event = $(event_number)

	xmin = $( @sprintf("%.2f", xbmin) ) mm, xmax = $( @sprintf("%.2f", xbmax) ) mm

	ymin = $( @sprintf("%.2f", ybmin) ) mm, ymax = $( @sprintf("%.2f", ybmax) ) mm

	size of ELCC matrix (holes = SiPMs) = $(size(holes))

	"""
	
end


# ╔═╡ 88d90e38-e01c-4938-9e6c-cc9609d8fb50
begin
	sipm1, tsipm1 = transport_xy_to_sipm(xx, yy, zz, holes, tbins)
	tsipm1_sum =[sum(x) for x in tsipm1]
	isipmx = argmax(tsipm1_sum)
	plot(tsipm1_sum, xlabel="sipm (hole) number", ylabel="# electrons reaching ELCC hole")
end

# ╔═╡ 37ede59f-64fb-4297-9634-1810919e6608
plot(tsipm1[isipmx], xlabel="time (μs)", ylabel="# electrons reaching ELCC hole")

# ╔═╡ 5cabdb81-5aa3-404a-9c7a-28d259a06f67
scatter(holes[1,:], holes[2,:], marker_z=sipm1, 
			color=:viridis, xlabel="x", ylabel="y", label=false)

# ╔═╡ 9c6856e5-c063-497c-b15b-d78d34421fbb
begin
	sipm2, tsipm2 = transport_xy_to_sipm(xx2, yy2, zz2, holes, tbins2)
	tsipm2_sum =[sum(x) for x in tsipm2]
	isipmx2 = argmax(tsipm2_sum)
	plot(tsipm2_sum, xlabel="sipm (hole) number", ylabel="# electrons reaching ELCC hole")
end

# ╔═╡ 23704a7a-cfef-482f-b240-4a9ad54026f7
plot(tsipm2[isipmx2], xlabel="time (μs)", ylabel="# electrons reaching ELCC hole")

# ╔═╡ 0c040aa4-78b5-49e3-a12d-2a2974133168
begin
	tsipm = tsipm1 .+ tsipm2
	tsipm_sum =[sum(x) for x in tsipm]
	isipmxx = argmax(tsipm_sum)
	p1 = plot(tsipm_sum, xlabel="sipm (hole) number", ylabel="# electrons reaching ELCC hole")
	p2 = plot(tsipm[isipmxx], xlabel="time (μs)", ylabel="# electrons reaching ELCC hole")
	plot(p1, p2, zone=(1,2))
end

# ╔═╡ c318420a-c068-4bb4-9c59-fb31256c22e6
scatter(holes[1,:], holes[2,:], marker_z=sipm2, 
			color=:viridis, xlabel="x", ylabel="y", label=false)

# ╔═╡ 5376610a-ae34-403c-b7b1-c2e1907cc8a5
"""
- Generates ionisation electrons (IE) in each point of the track.
- Transport the IE to the ELCC (diffusion DL and DT)
- For each IE find the ELCC hole (thus SiPM) where IE goes 
- Count the IE in each SiPM and arrange them in a time vector (given by DL diffusion) of ntb bins (intervals of tb mus, use the approx relation 1μs = 1mm)
"""
function elcc(hitdf, event_number, Wi, ph, dh; ntb=200, tb=1)
	edf                  =select_event(hitdf, event_number) 
	nel_ev               = edf.energy * 1E+6/Wi
	xn_ev, yn_ev, zn_ev  = transport_to_el(edf, boxtpc)
	xmean_ev, xstd_ev    = mean_diff(xn_ev)
	ymean_ev, ystd_ev    = mean_diff(yn_ev)
	zmean_ev, zstd_ev    = mean_diff(zn_ev)

	xbmin = minimum(xmean_ev)
	xbmax = maximum(xmean_ev)
	ybmin = minimum(ymean_ev)
	ybmax = maximum(ymean_ev)
		
	# Generate Hole Positions
	holes = generate_hole_positions(xbmin, xbmax, ybmin, ybmax, ph, dh)

	println("Wi = $(Wi) ev")
	println(""" ### ELCC Detection 
	
	Dimensions of ELCC detection area for event = $(event_number)

	xmin = $( @sprintf("%.2f", xbmin) ) mm, xmax = $( @sprintf("%.2f", xbmax) ) mm

	ymin = $( @sprintf("%.2f", ybmin) ) mm, ymax = $( @sprintf("%.2f", ybmax) ) mm

	size of ELCC matrix (holes = SiPMs) = $(size(holes))
	""")
	
	nt = size(edf)[1]
	txsipm = Vector{Vector{Vector{Float64}}}(undef, nt)

	for ip in 1:size(edf)[1]
		xx, yy, zz  = get_xyz_point(xn_ev, yn_ev, zn_ev, pt1) 
		zhst = Hist1D(zz,binedges = 1:tb:ntb+1)
		tbins = bincounts(zhst)./nentries(zhst)
		sipm, tsipm = transport_xy_to_sipm(xx, yy, zz, holes, tbins)
		#println("tispm = $(typeof(tsipm))")
		txsipm[ip] = tsipm
	end
	xbmin, xbmax, ybmin, ybmax, holes, txsipm
end

# ╔═╡ af0341d5-70b5-4fce-9ec5-c65cb4ec066d
xbm, xbmx, ybm, ybmx, hlx, txsipm = elcc(hitdf, event_number, Wi, ph, dh; ntb=200, tb=1)

# ╔═╡ 8aff1c8a-1b5f-4224-ac26-ffab1edb9a22
begin
	npoints = length(txsipm)
	nsi = length(txsipm[1])
	nst = length(txsipm[1][1])
md"""
- Vector txsipm has $(npoints) vector, corresponding to the numper of points in the event.
- Each vector has $(nsi) vectors, corresponding to the matrix os SiPMs.
- Each vector of the matrix has $(nst) numbers, corresponding to the number of electrons recorded in the SiPM in a time series of $(nst) elements. 
"""
end

# ╔═╡ b30aa2b9-9557-4ff1-9790-88757aeab321
begin
	ts = add_time_series(txsipm)
	
	
md"""
-  Vector ts is obtained adding the time series for all points in each SiPM.
- Each element of ts has $(length(ts)) SiPMs.
- Each SiPM has a time series of length $(length(ts[1]))
"""
end

# ╔═╡ 49da6468-0b78-4c16-ac87-2dd06f15c481
begin
	emax, emean, etot = statsts(ts)
	md"""
	- emax computes the maximum number of electron in the time series, and thus one element for SiPM with length $(length(emax))
	- emean computes the mean number of electron in the time series, and thus one element for SiPM with length $(length(emax))
	- etot adds all electron in the time series, and thus one element for SiPM with length $(length(emax))
	"""
end

# ╔═╡ 0fd805b8-d659-44be-8da5-e4378d974f36
begin
	px4 = plot(Hist1D(edf.energy * 1E+6/Wi, binedges = 0:2:200), title="# electrons per point")
	px1 = plot(Hist1D(emean, binedges = 0:2:200), title=" <# electrons> per sipm")
	px2 = plot(Hist1D(emax, binedges = 0:10:1000), title=" max(electrons) per sipm")
	px3 = plot(Hist1D(etot, binedges = 0:400:40000), title=" sum(electrons) per sipm")
	
	plot(px4, px1,px2,px3, zone=(2,2))
	
end

# ╔═╡ dd1b911f-3f6c-43a1-87f2-8e046835b480
begin
	maxpe = emax * gt
	avpe = emean * gt
	mxpecl = maxpe/ncl
	avpecl = avpe/ncl
	totpe  = etot * gt
	pestot = sum(totpe)
	sigma_pes = 100/sqrt(pestot)
	pxx1 = plot(Hist1D(maxpe, binedges = 0:50:5000), title=" max pes per sipm")
	pxx2 = plot(Hist1D(avpe, binedges = 0:50:5000), title=" average pes per sipm")
	pxx3 = plot(Hist1D(mxpecl, binedges = 0:0.05:1), title=" max pes sipm/cell")
	pxx4 = plot(Hist1D(avpecl, binedges = 0:0.05:1), title=" average pes sipm/cell")
	
	plot(pxx1,pxx2,pxx3, pxx4, zone=(2,2), size = (900,600))
	
end

# ╔═╡ a815824c-4b55-442e-9634-28731944f23c
md"""
- total number of pes per event = $(@sprintf("%.2f", pestot) )
- relative resolution = $(@sprintf("%.2f", sigma_pes) ) %
"""

# ╔═╡ 7c0587b4-00d1-4f47-89f1-6a13b022ec43
begin
	hmxpe = Hist1D(maxpe, binedges = 0:250:50000)
	mxpe = bincounts(hmxpe)
	lmxpe = length(mxpe)
	xss = sum(mxpe[4:lmxpe])/sum(mxpe)

	md"""
	- Fraction of SiPMs with one or more saturated sample (histo counting) = $(@sprintf("%.2f", xss))
	
	"""
end

# ╔═╡ 976f69f8-c159-4de9-8e48-b1dfad22669b
begin 
	xfs = fraction_saturated_sipm(ts,mxp,gt)
md"""
- Fraction of SiPMs with one or more saturated sample = $(@sprintf("%.2f", xfs))

"""
end

# ╔═╡ 520341e4-d79f-41f9-9ffd-2ac85a68fcde
scatter(hlx[1,:], hlx[2,:], marker_z=emean, 
			color=:viridis, xlabel="x", ylabel="y", label=false)


# ╔═╡ 4b7b91be-fdf3-4ab8-9d8b-35c6749c2475
scatter(hlx[1,:], hlx[2,:], marker_z=emax, 
			color=:viridis, xlabel="x", ylabel="y", label=false)

# ╔═╡ 6d8da4dd-572f-4c8e-8959-0f67fdb3f692
scatter(hlx[1,:], hlx[2,:], marker_z=etot, 
			color=:viridis, xlabel="x", ylabel="y", label=false)

# ╔═╡ 34dee5c2-bd5c-4be7-9ab4-b09c5867d125
begin
	xbm, xbmx, ybm, ybmx, hlx
	rx, ry = xbm:1.0:xbmx, ybm:1.0:ybmx
	wgts = totpe
    hxyevt = Hist2D((hlx[1,:], hlx[2,:]); weights = wgts, binedges = (rx, ry))
	plot(hxyevt)
end

# ╔═╡ a1a4f09b-a8b5-4a2f-b281-68d36b5939ef
xbx, yby, zxpes, zypes = mat_hxz(hlx, ts)

# ╔═╡ 40e5aefd-505b-4225-8e0e-687dad35387b
zxpes

# ╔═╡ f658dc54-4c7d-4e49-af5f-5df749c50852
zypes

# ╔═╡ fe61e9ed-a279-4e08-b224-ddd4eea4ade1
heatmap(zxpes)

# ╔═╡ 05a314de-2eec-419a-97d6-8bf55dfa9938
heatmap(zypes)

# ╔═╡ 1e23c6cb-853b-43a9-ae97-e4b144d04fcb
"""
Function to define a circle as a parametric surface
(xc,yc) is the circle center, r is the radius.
"""
function circle(xc,yc,r)
	θ = LinRange(0, 2* π, 500)
	xc .+ r*sin.(θ), yc .+ cos.(θ)
end

# ╔═╡ ed4a219b-e405-450c-a006-a2dbcffb5f7d
"""
Plotting the Surface and Holes
"""
function plot_surface_with_holes(xmin, xmax, ymin, ymax, holes, d; sample=1)
    # Plot the surface as a rectangle
    p1 = plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], lw=2,
		      xlims=(xmin, xmax), ylims=(ymin, ymax), aspect_ratio=1,
              legend=false, title="Surface with Circular Holes", xlabel="X",   
		      ylabel="Y")

	
    # Plot each (sampled) hole as a circle 
	for n in 1:sample:size(holes)[2]
		x = holes[1,n]
		y= holes[2,n]
        p1 = plot!(p1, circle(x, y, dh/2), color=:blue, linecolor=:red, lw=1,
			       legend=false, aspect_ratio=1)
    end
	p1
end

# ╔═╡ e18be5a0-ff16-4552-996c-cfe7ef52a36a
"""
Plot the event in the EL Plane 
"""
function plot_event_elplane(xmin, xmax, ymin, ymax, xx, yy, holes, dh)
	#xmin = minimum(xx)
	#xmax = maximum(xx)
	#ymin = minimum(yy)
	#ymax = maximum(yy)
	p1 = plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], lw=2,
		      xlims=(xmin, xmax), ylims=(ymin, ymax), aspect_ratio=1,
              legend=false, title="EL plane", xlabel="X",   
		      ylabel="Y")

	# Plot each (sampled) hole as a circle 
	for n in 1:size(holes)[2]
		x = holes[1,n]
		y= holes[2,n]
        p1 = plot!(p1, circle(x, y, dh/2), color=:blue, linecolor=:red, lw=1,
			       legend=false, aspect_ratio=1)
    end

	# Plot drift electrons
	scatter!(p1, xx, yy, markersize=1)
	p1
end

# ╔═╡ 6fa739ae-6e98-4378-93c5-41c2e68b971d
plot_event_elplane(xbmin, xbmax, ybmin, ybmax, xx, yy, holes, dh)

# ╔═╡ 22b78ecd-e1bc-4dad-983b-f514dfc77ac6
plot_event_elplane(xbmin, xbmax, ybmin, ybmax, xx2, yy2, holes, dh)

# ╔═╡ 430e46a6-5cad-4031-ad6c-e94f7cf70c2a
"""
Binning xyz
"""
function binxyz(x,y,z, n)
	
	function assign_bin(zi, bins, n)
	    for i in 1:n
	        if zi >= bins[i] && zi < bins[i+1]
	            return i
	        end
	    end
	    return n  # Assign to last bin if zi == z_max
	end

	# Define bin edges
	z_min, z_max = minimum(z), maximum(z)
	bin_width = (z_max - z_min) / n
	bins = [z_min + i * bin_width for i in 0:n]


	# Create a DataFrame
	df = DataFrame(x = x, y = y, z = z)

	# Assign each z to a bin
	df.bin = [assign_bin(zi, bins, n) for zi in df.z]


	# Define bin edges and assign bins
	#cut(0:10, 0:5:15, labels=[2.5, 7.5, 12.5])
	#df.bin = cut(df.z, bins = n, labels = false)
	#df.bin = clamp.(df.bin, 1, n)  # Ensure bin indices are within [1, n]

	# Group by 'bin'
	grouped = groupby(df, :bin)

	# Collect grouped data
	grouped_x = [group.x for group in grouped]
	grouped_y = [group.y for group in grouped]
	grouped_z = [group.z for group in grouped]
	return grouped_x, grouped_y, grouped_z

end


# ╔═╡ f370d35b-633e-4287-9178-c2a88f7372ad
md"""
 ## GPT queries
"""

# ╔═╡ d34cb5ad-ab8c-459c-a5b6-09b3f227fb9a
md"""
Consider a surface defined by (xmin, xmax), (ymin, ymax)
I want to define circular holes across all the surface. The holes have a diameter d and are spaced at a pitch p. I want to place n holes in positions x1 = xmin + p/2, x2 = x1 + p, x3 = x3 + p...and so on, with the constrain the the last hole must be inside the surface (e.g., xn + d/2 < xmax). Same for y coordinate. 
1) Provide an algorithm to fill the positions of the holes, so that hx[i] = xi, hy[i] = yi.
2) Given an arbitrary position in the plane, (xc, yc), provide a fast algorithm to find the hole which is nearest that position.
3) Provide code to plot the surface and the holes.
4) Output the algorithm in latex format
5) all the code must be in Julia
"""

# ╔═╡ b7e4b911-4078-4849-8d46-53ac8f21b149
md"""
I have three vectors of the same length x, y, z. I need to group z in n bins. I need julia code so that x and y are also grouped in the same n bins. 
"""

# ╔═╡ Cell order:
# ╠═56471816-de2f-11ef-0099-9b1a882f96c8
# ╠═5769b663-7790-45eb-b5b1-dd513a284458
# ╠═2dd7f02d-73f6-45c2-a904-482efdd56b28
# ╠═d57ed4ef-8b54-4b55-9c56-47345aea1a26
# ╠═775d3ebb-4e86-4e91-9954-628cbec2529b
# ╠═0f23f280-f816-4aca-9c20-5425e4641b8b
# ╠═a32930ae-120f-4f6e-8e6f-300580ec9ade
# ╠═cc1ac654-e7b0-4441-ae9e-6faddb9cfbb0
# ╠═91d39551-02db-42da-9eca-f9623506e761
# ╠═9695043a-7610-4827-ae47-f600768bf939
# ╠═bd3638dc-4916-4337-8e7c-f9dda76b6928
# ╠═38ff4567-8d5c-4925-994c-22586c8f0ea6
# ╠═ce05f403-d0bf-41e9-bcc7-f6ed7768bf71
# ╠═08935cb7-3f21-4e8e-92a1-a8d5e60254f6
# ╠═a5f74f36-a089-4123-8d60-cddf2af6a697
# ╠═c255e0b0-bfe1-49e7-9259-ba87ce4ec55f
# ╠═aabb2413-4512-4c39-8070-5e853c9c0127
# ╠═2e3cb152-50cb-42e7-a02d-7533e781686d
# ╠═55be668d-4af3-4d32-b048-495ab020e113
# ╠═94f49857-c6b4-4991-8846-d57b93fd0901
# ╠═2cb0bcef-1892-4b3b-9a4b-89c290eb57b1
# ╠═a65da87a-0689-48c0-816d-c415c71d6bc9
# ╠═6831ccc5-3607-4ace-9f37-da203502b1b4
# ╠═bcb22069-18eb-44aa-ba00-26244be4e3fb
# ╠═b59aa31a-c222-457c-8b46-73d745173885
# ╠═bcb2c7e6-cb17-4048-86b7-3a854fc24629
# ╠═c49b70bc-1e09-4571-b9a4-31cd5e303db9
# ╠═a84d9f3f-ba81-4817-8f66-97c8c65adb5e
# ╠═6e1e25bf-5c11-470c-a739-1214243c8438
# ╠═6fa739ae-6e98-4378-93c5-41c2e68b971d
# ╠═88d90e38-e01c-4938-9e6c-cc9609d8fb50
# ╠═37ede59f-64fb-4297-9634-1810919e6608
# ╠═5cabdb81-5aa3-404a-9c7a-28d259a06f67
# ╠═99c6d9e7-30f5-4dd3-a95d-14afa511bc53
# ╠═8a8d1789-c1cb-4c47-a578-da926bf34edc
# ╠═ce84c2a4-53d7-4bc7-871e-51aea25093b8
# ╠═22b78ecd-e1bc-4dad-983b-f514dfc77ac6
# ╠═9c6856e5-c063-497c-b15b-d78d34421fbb
# ╠═23704a7a-cfef-482f-b240-4a9ad54026f7
# ╠═c318420a-c068-4bb4-9c59-fb31256c22e6
# ╠═5f1de077-147b-4fe1-84b3-0327d795214e
# ╠═0c040aa4-78b5-49e3-a12d-2a2974133168
# ╠═24d3a2bc-3406-4705-9eaa-a3244c9b9244
# ╠═86149e81-ed71-4c57-aeb8-fe525be4ad64
# ╠═af0341d5-70b5-4fce-9ec5-c65cb4ec066d
# ╠═8aff1c8a-1b5f-4224-ac26-ffab1edb9a22
# ╠═b30aa2b9-9557-4ff1-9790-88757aeab321
# ╠═49da6468-0b78-4c16-ac87-2dd06f15c481
# ╠═8d1e1d84-a7d0-44e0-8ddf-2dc3b2f43add
# ╠═520341e4-d79f-41f9-9ffd-2ac85a68fcde
# ╠═921e6046-ff59-43d2-a834-a74320331ef9
# ╠═4b7b91be-fdf3-4ab8-9d8b-35c6749c2475
# ╠═2ceb0f10-33c1-45fc-a5a8-e456f481f3ac
# ╠═6d8da4dd-572f-4c8e-8959-0f67fdb3f692
# ╠═0fd805b8-d659-44be-8da5-e4378d974f36
# ╠═8fd4e626-f8f4-4541-b520-19fe8beeef1c
# ╠═dd1b911f-3f6c-43a1-87f2-8e046835b480
# ╠═34dee5c2-bd5c-4be7-9ab4-b09c5867d125
# ╠═a1a4f09b-a8b5-4a2f-b281-68d36b5939ef
# ╠═40e5aefd-505b-4225-8e0e-687dad35387b
# ╠═f658dc54-4c7d-4e49-af5f-5df749c50852
# ╠═fe61e9ed-a279-4e08-b224-ddd4eea4ade1
# ╠═05a314de-2eec-419a-97d6-8bf55dfa9938
# ╠═a815824c-4b55-442e-9634-28731944f23c
# ╠═ae4ae06f-a756-4cf6-94d0-d08dac29ed48
# ╠═7c0587b4-00d1-4f47-89f1-6a13b022ec43
# ╠═976f69f8-c159-4de9-8e48-b1dfad22669b
# ╠═dde3b9ff-3f28-49c8-b3cb-efe9f9827e85
# ╠═5376610a-ae34-403c-b7b1-c2e1907cc8a5
# ╠═df6e502e-7892-4158-bfb7-cc38214618a1
# ╠═f7b2fbef-f189-4c4e-b7ce-ee46688d530d
# ╠═4c58b596-9da3-4e31-ae3e-5acf18e206a4
# ╠═a442b7d0-9035-48a9-bbce-98c66c09e05f
# ╟─c446556a-239d-4e95-a3e7-3f7f3679e038
# ╠═a01e9701-dce9-496e-85f9-cc511fb274ed
# ╠═b9c13dfc-614d-4f95-9a71-8c3152a0c5fd
# ╠═9ed19e6c-2f9b-4d16-940f-6269427ac8ee
# ╠═8b4ddbbf-a22b-4249-8980-630bc2b9bf2b
# ╠═6ba99a14-a13e-44ea-a873-4597c9b0fd13
# ╠═b38a1a09-00ea-4ad4-ac9d-c74a5a1f7700
# ╠═166dd216-79be-4974-bab9-b2a64bd2d817
# ╠═79b86295-13e1-4277-8dc0-55106ba2e887
# ╠═4e0634f1-3b46-47fe-b742-a815ea1aef4e
# ╠═790948af-bed5-4f68-a52a-de7d6af932fb
# ╠═2a0ffd4a-0a8a-45a3-8f98-3893f8628ce4
# ╠═5ebd4bf9-b53f-4610-8260-0c42b302a6ea
# ╠═69b72ac6-c947-44bf-a10f-6aa7726a7a10
# ╠═bbd23f58-592d-4c96-b2e6-9449396da78f
# ╠═2b0d31f0-69c3-4a2c-9d6c-7e96d6393571
# ╠═a62a307b-28e9-4b15-ac5e-ce2d4745a5cc
# ╠═68f875be-3126-4041-90a8-14403b8def72
# ╠═1e23c6cb-853b-43a9-ae97-e4b144d04fcb
# ╠═ed4a219b-e405-450c-a006-a2dbcffb5f7d
# ╠═e18be5a0-ff16-4552-996c-cfe7ef52a36a
# ╠═430e46a6-5cad-4031-ad6c-e94f7cf70c2a
# ╠═f370d35b-633e-4287-9178-c2a88f7372ad
# ╠═d34cb5ad-ab8c-459c-a5b6-09b3f227fb9a
# ╠═b7e4b911-4078-4849-8d46-53ac8f21b149
