### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7884ab66-f9e2-11ef-03ea-f10faf671dba
using Pkg; Pkg.activate(ENV["JNEXT"])


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

# ╔═╡ a333497c-c5fc-49a7-a8ca-d82a7dcd27ad
begin
	import PyPlot
end

# ╔═╡ 57248e63-9e36-4644-a6c4-5a3aa1808e29
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

# ╔═╡ eada2802-68a7-4663-a2c0-c1ca41b74601
jn = ingredients(string(ENV["JNEXT"],"/src/JNEXT.jl"))

# ╔═╡ 2617ae05-e1db-473a-9b0f-befeea6d0e12
md"""
# Simulation
"""

# ╔═╡ a891cff0-6910-4f78-8fc5-ff4e90163a7e
begin
	V = 1e+3
	kV = 1e+3V
	mm = 1.0
	cm = 10.0
end

# ╔═╡ ac79ab2e-af61-499a-94e7-964a8f04b111
begin

	# GALA geometry. The collector is located at 10 mm, the gate at 0 mm and the anode at -5 mm (the SiPMs at -10 mm)
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
	trfile ="trj_d_3_p_6.jld2"

	R = d_hole/2
	dd = d_hole
	pp = (pitch - d_hole)/2
	x0 = y0 = pp + R

	sipmXY = 6mm
	sipmPitch = 10mm
	
	md"""
	- Size of ELCC $(X) x $(Y) mm
	- hole center in ($(x0), $(y0)) mm
	- hole diameter $(dd) mm, pitch $(pitch) mm
	- Zc = $(Zc), Zg = $(Zg), Za = $(Za), Zs = $(Zs)
	-
	"""
end

# ╔═╡ b9df9120-e258-4a63-9dfa-2b0ecf9c5ceb
begin
	ymm = jn.JNEXT.GSim.yield_mm(Zg/mm, Za/mm, Vg/kV, Va/kV; p=10)
	md"""
	- yield/mm = $(ymm)
	"""
end

# ╔═╡ 321fb432-4464-47b8-94ac-30d466670224
md"""
## ELCC geometry
"""

# ╔═╡ dfd7cbf8-adaa-454f-957e-ecc6eee905d3
begin
	elcc = jn.JNEXT.GSim.ELCCGeometry(X/mm, Y/mm, Zc/mm, Zg/mm, Za/mm, Zs/mm, Vg/kV, Va/kV, d_hole/mm, pitch/mm)
	
	sipm = jn.JNEXT.GSim.SiPMGeometry(sipmXY/mm, sipmPitch/mm, X/mm, Y/mm)
	
	jn.JNEXT.GSim.plot_gala_geom(elcc, sipm)
end

# ╔═╡ 16e4221a-4dd8-4571-8ce2-ef259400562a
begin
	ndx = jn.JNEXT.GSim.ndicex(elcc)
	nsix = jn.JNEXT.GSim.nsipmx(sipm)
	ndy = jn.JNEXT.GSim.ndicey(elcc)
	nsiy = jn.JNEXT.GSim.nsipmy(sipm)
md"""

- ELCC structure created with ( $(ndx) x $(ndy)) dices and ( $(nsix), $(nsiy)) sipms.
"""
end

# ╔═╡ a340f566-c9c0-4293-988e-11b7e69e8e4a
md"""
## Load trajectories
"""

# ╔═╡ c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
@load  trfile ftrj btrj

# ╔═╡ 7c38062b-0671-451c-911e-f88272f97937
begin
	zftrj = [ftrj[i][1,1:2] for i in range(1, length(ftrj))]
	xmax = maximum([z[1] for z in zftrj])
	xmin = minimum([z[1] for z in zftrj])
	ymax = maximum([z[2] for z in zftrj])
	ymin = minimum([z[2] for z in zftrj])
	md"""
	#### Trajectories:
	- in X, between $(xmin) and $(xmax)
	- in Y, between $(ymin) and $(ymax)
	
	"""
end

# ╔═╡ 7d32dbb6-04d7-456d-9db8-b4b98ff70de3
@bind RandomPosition CheckBox()

# ╔═╡ c5924aa7-a04b-4820-aafb-2c71a5bb289d
@bind SimulatePhotons CheckBox()

# ╔═╡ b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
md"""
# Functions
"""

# ╔═╡ c7704f94-2ab5-4111-ac7c-63f978c7ee4c
function float_to_str(number, fmt::String)
    io = IOBuffer()
    Printf.format(io, Printf.Format(fmt), number)
    return String(take!(io))
end

# ╔═╡ fcbf9e5a-b7f2-400d-87af-7448fd348071
function vect_to_str(vec::AbstractVector, fmt::String)
    # Helper function to format a single element
    function format_element(x, fmt)
        if x isa AbstractFloat
            float_to_str(x, fmt)
        else
            return string(x)
        end
    end
	formatted_elements = [format_element(x, fmt) for x in vec]
	return "[" * join(formatted_elements, ", ") * "]"
end

# ╔═╡ 2273c136-4709-43a6-bf68-1184493fbb70
begin
	if RandomPosition
		electron_xy0 = jn.JNEXT.GSim.generate_electron_positions_random(20, 0.0, elcc.X, 0.0, elcc.Y)
		i = 1 # electron number to run 
	else
		electron_xy0 = jn.JNEXT.GSim.generate_electron_positions(200, 0.0, elcc.X, 0.0, elcc.Y)
		i = 100 # electron number to run 
	end
	
	electron_pos = electron_xy0[i, :]  
	md"""
	#### Generate electron
	- Absolute position position $(vect_to_str(electron_pos, "%.2f"))
	"""
end

# ╔═╡ 9eb86c8c-4347-46c4-9111-793f921fac56
jn.JNEXT.GSim.plot_impact_xy(elcc, sipm, electron_pos)

# ╔═╡ 5b23138d-32e7-4ec1-8032-e00ce1848459
dice_i, dice_o, xxlocal = jn.JNEXT.GSim.find_dice(electron_pos, elcc)
	

# ╔═╡ 33391da0-e8a4-4c3b-ba9e-1d5ca55e69da
sipm_i, sipm_o =jn.JNEXT.GSim.find_sipm(electron_pos, sipm)

# ╔═╡ ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
begin 
	dice_indices, dice_origin, xlocal = jn.JNEXT.GSim.find_dice(electron_pos, elcc)
	sipm_indices, sipm_origin =jn.JNEXT.GSim.find_sipm(electron_pos, sipm)
	xl = collect(xlocal)
	izfr = argmin([norm(xl -zftrj[i]) for i in 1:length(zftrj)])
	md"""
	- dice indices = $(dice_indices), sipm indices = $(sipm_indices)
	- local coordinates = $(vect_to_str(xl, "%.2f"))
	- Closer trajectory number $(izfr), with coordinates $(vect_to_str(ftrj[izfr][1,:], "%.2f"))
	"""
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

# ╔═╡ 2144e49f-1505-4c19-a047-7733c7cfc0c1
jn.JNEXT.GSim.p_trajectory(electron_pos, elcc, sipm, tr) 

# ╔═╡ d474342a-81ca-4504-86a9-52925211b685
tr[end-10:end,3]

# ╔═╡ 1ef1b221-da82-4852-bfb3-ffe1b2b50600
typeof(tr)

# ╔═╡ 27669660-d21b-4e10-904d-b8142e8447dd
if SimulatePhotons
	gcounter, cij, gammas, jsteps = jn.JNEXT.GSim.simulate_photons_along_trajectory(electron_pos,
		                                                        tr, 
	                                                            elcc,
	                                                            sipm; 
										                        ymm=ymm,
										                        p1=0.00, 
										                        p2=0.5,  
										                        samplet=1000, 
										                        maxgam=200,
	                                                            saveg=true,
	                                                            savet = true,
																        eps=1e-14)
	
end

# ╔═╡ 10bfa4fb-b245-4467-805e-2ffd9314b58f
typeof(jsteps)

# ╔═╡ 1dad2fcb-836e-46a8-bb2b-8d43f25c4767
typeof(gammas)

# ╔═╡ fbf61158-272f-435e-aeba-ace8ee442c71
jn.JNEXT.GSim.p_trajectory(electron_pos, elcc, sipm, jsteps, gammas)

# ╔═╡ 6dd673b2-1c77-46eb-bf65-375bb11c7c99
maximum(cij)

# ╔═╡ f0f0f9bb-4c29-4e78-bc92-4a441d99c58b
jn.JNEXT.GSim.p_hitmatrix(cij, sipm)

# ╔═╡ b5247e3b-1197-4cb2-ad9e-50b4c7fd8c4e
function getmax(c::AbstractMatrix)
	max_val, lin_idx = findmax(c)
	ci = CartesianIndices(size(c))[lin_idx]
	i, j = Tuple(ci)
	i,j, max_val
end

# ╔═╡ ae488bf8-706a-4d57-8ac7-412f0a43bd08
let
	eff = gcounter.bottom/gcounter.total
	ic,jc, cmax = getmax(cij)
	Qmx = cmax/gcounter.bottom
	cu =cij[ic,jc+1]
	cd =cij[ic,jc-1]
	cr =cij[ic+1,jc]
	cl =cij[ic-1,jc]
	Qu = cu/gcounter.bottom
	Qd = cd/gcounter.bottom
	Qr = cr/gcounter.bottom
	Ql = cl/gcounter.bottom
	md"""
	- ELCC efficieny = $(float_to_str(eff, "%.2f"))
	- max signal SiPM: ( $(ic), $(jc)): Qmx = $(float_to_str(Qmx, "%.2f"))
	- max signal SiPM: ( $(ic), $(jc+1)): Qr = $(float_to_str(Qr, "%.2f"))
	- max signal SiPM: ( $(ic), $(jc-1)): Ql = $(float_to_str(Ql, "%.2f"))
	- max signal SiPM: ( $(ic+1), $(jc)): Qu = $(float_to_str(Qu, "%.2f"))
	- max signal SiPM: ( $(ic-1), $(jc)): Qd = $(float_to_str(Qd, "%.2f"))
	"""
end

# ╔═╡ 42d12a7c-e067-4342-860e-ad3530913094
md"""
### Intersection of a Line with a Finite Cylinder

#### 1. Equation of the Line
The line is given in parametric form as:

``
\begin{aligned}
x(t) &= x + v_x\,t,\\,
y(t) &= y + v_y\,t,\\,
z(t) &= z + v_z\,t,
\end{aligned}
``

where ``(x,y,z)`` is a point on the line and ``(v_x,v_y,v_z)`` are the direction cosines.

#### 2. Description of the Cylinder

The cylinder is defined by:

- **Lateral Surface:** All points satisfying

  ``
  (x - x_0)^2 + (y - y_0)^2 = R^2,
  ``

  where ``(x_0, y_0)`` is the center of the cylinder’s circular cross-section in the ``xy``-plane and ``R`` is the cylinder’s radius.

- **End-Caps:** Two horizontal planes at:

  ``
  z = z_a \quad (\text{top cap}) \quad \text{and} \quad z = z_b \quad (\text{bottom cap}),
  ``

  with ``z_a > z_b``.

#### 3. Intersection with the Lateral Surface
Substitute the parametric equations of the line into the cylinder’s equation:

``
\bigl(x + v_x\,t - x_0\bigr)^2 + \bigl(y + v_y\,t - y_0\bigr)^2 = R^2.
``

Expanding, we obtain a quadratic in \(t\):

``
(v_x^2 + v_y^2)t^2 + 2\bigl[v_x(x-x_0) + v_y(y-y_0)\bigr]t + \Bigl[(x-x_0)^2+(y-y_0)^2 - R^2\Bigr] = 0.
``

Let

``
a = v_x^2 + v_y^2,\quad b = 2\bigl[v_x(x-x_0) + v_y(y-y_0)\bigr],\quad c = (x-x_0)^2 + (y-y_0)^2 - R^2.
``

Then the quadratic equation is:

``
a\,t^2 + b\,t + c = 0.
``

The solutions are:

``
t = \frac{-b \pm \sqrt{b^2-4ac}}{2a}.
``

For real intersections, the discriminant \(D = b^2 - 4ac\) must be non-negative. For each valid \(t\), the \(z\)-coordinate is given by:

``
z(t) = z + v_z\,t,
``

and must satisfy ``z_b \le z(t) \le z_a`` to lie within the finite cylinder.

#### 4. Intersection with the End-Caps

**Top End-Cap (at ``z = z_a``):**

- Set the ``z`` equation equal to ``z_a``:

``
  z + v_z\,t = z_a \quad \Longrightarrow \quad t = \frac{z_a - z}{v_z}\quad (v_z \neq 0).
``

- The corresponding ``x`` and ``y`` coordinates are:

  ``
  x(t) = x + v_x\,t,\quad y(t) = y + v_y\,t.
  ``

- This intersection is valid if:

  ``
  (x(t)-x_0)^2 + (y(t)-y_0)^2 \le R^2.
  ``


**Bottom End-Cap (at ``z = z_b``):**
- Similarly, set:

  ``
  z + v_z\,t = z_b \quad \Longrightarrow \quad t = \frac{z_b - z}{v_z}\quad (v_z \neq 0).
  ``

- The intersection is valid if:

  ``
  (x(t)-x_0)^2 + (y(t)-y_0)^2 \le R^2.
  ``


"""

# ╔═╡ 569026e1-b82f-4230-b8f5-4fe60afd2cb7


# ╔═╡ Cell order:
# ╠═7884ab66-f9e2-11ef-03ea-f10faf671dba
# ╠═108bd997-6d2d-416d-b2df-ec034273d62e
# ╠═a333497c-c5fc-49a7-a8ca-d82a7dcd27ad
# ╠═57248e63-9e36-4644-a6c4-5a3aa1808e29
# ╠═eada2802-68a7-4663-a2c0-c1ca41b74601
# ╠═2617ae05-e1db-473a-9b0f-befeea6d0e12
# ╠═a891cff0-6910-4f78-8fc5-ff4e90163a7e
# ╠═ac79ab2e-af61-499a-94e7-964a8f04b111
# ╠═b9df9120-e258-4a63-9dfa-2b0ecf9c5ceb
# ╠═321fb432-4464-47b8-94ac-30d466670224
# ╠═dfd7cbf8-adaa-454f-957e-ecc6eee905d3
# ╠═16e4221a-4dd8-4571-8ce2-ef259400562a
# ╠═a340f566-c9c0-4293-988e-11b7e69e8e4a
# ╠═c26b4fc3-3d16-45fc-bffc-934ccfd9deb9
# ╠═7c38062b-0671-451c-911e-f88272f97937
# ╠═7d32dbb6-04d7-456d-9db8-b4b98ff70de3
# ╠═2273c136-4709-43a6-bf68-1184493fbb70
# ╠═9eb86c8c-4347-46c4-9111-793f921fac56
# ╠═ce9fe145-efa7-4421-9c90-1d9ee73c3e1e
# ╠═951ea570-d232-47a3-bbe8-b216de1469a8
# ╠═2144e49f-1505-4c19-a047-7733c7cfc0c1
# ╠═d474342a-81ca-4504-86a9-52925211b685
# ╠═1ef1b221-da82-4852-bfb3-ffe1b2b50600
# ╠═c5924aa7-a04b-4820-aafb-2c71a5bb289d
# ╠═27669660-d21b-4e10-904d-b8142e8447dd
# ╠═10bfa4fb-b245-4467-805e-2ffd9314b58f
# ╠═1dad2fcb-836e-46a8-bb2b-8d43f25c4767
# ╠═5b23138d-32e7-4ec1-8032-e00ce1848459
# ╠═33391da0-e8a4-4c3b-ba9e-1d5ca55e69da
# ╠═fbf61158-272f-435e-aeba-ace8ee442c71
# ╠═6dd673b2-1c77-46eb-bf65-375bb11c7c99
# ╠═f0f0f9bb-4c29-4e78-bc92-4a441d99c58b
# ╠═ae488bf8-706a-4d57-8ac7-412f0a43bd08
# ╠═b56be6ba-7e8a-48c2-a2d3-7b4e27e4f03c
# ╠═c7704f94-2ab5-4111-ac7c-63f978c7ee4c
# ╠═fcbf9e5a-b7f2-400d-87af-7448fd348071
# ╠═b5247e3b-1197-4cb2-ad9e-50b4c7fd8c4e
# ╠═42d12a7c-e067-4342-860e-ad3530913094
# ╠═569026e1-b82f-4230-b8f5-4fe60afd2cb7
