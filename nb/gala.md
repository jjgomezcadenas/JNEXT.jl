PlutoUI.TableOfContents(title="", indent=true)
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")
begin
	using PlutoUI
	using Plots
	using Random
	using Test
	using Printf
	using InteractiveUtils
	using SparseArrays, LinearAlgebra
	using Unitful 
end
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
begin
	cd("/Users/jjgomezcadenas/Projects/JNEXT/nb")
	jn = ingredients("../src/JNEXT.jl")
end

begin

	d = 2.5      # hole diameter in mm 
	rmax = 5.0    # simulation domain in r: 
	l1 = 2.0    # anode position
	l2 = 7.0    # gate position
	l3 = 8.0    # top electrode position
	Va = 5e+6    # anode potential (V) : V scaled up so that V/m makes sense
	#Va = 0.0    # anode potential (V) : V scaled up so that V/m makes sense
	Vg = -12e+6  # gate potential (V)
	V0 = -12.5e+6   # top electrode potential (V)
	
	
	gparams = jn.JNEXT.GALAParams(d, rmax, l1, l2, l3, Va, Vg, V0)
		#(A_mat, b_vec, rgrid, zgrid), (phi_mat, rgrid, zgrid)
	
	Nr=301 
	Nz=301
end

begin 
	la = 0.0
	if Va > 0
		la = l1
	end
end

t = @elapsed begin
    #phi_mat, rgrid, zgrid = solve_laplace_gala(params; Nr=51, Nz=101)
	#AA, PHI = jn.JNEXT.solve_laplace_gala(gparams; Nr=Nr, Nz=Nz, returnMatrix=false)
	phi_mat, rgrid, zgrid = jn.JNEXT.solve_laplace_gala(gparams; Nr=Nr, Nz=Nz, returnMatrix=false)
end

println("Nr = $(Nr), Nz = $(Nz): Elapsed time (Julia): ", t, " seconds")

#############################
# Example Usage
#############################
# Define parameters for the 2D problem with two holes.
# Hole 1: centered at x1; Hole 2: centered at x1+p.
# Domain in x: [x1 - p/2, x1 + p + p/2]
params2D = GALAParams2D(
    x1 = 0.8e-3,    # example: hole 1 center at 0.8 mm
    d1 = 1e-3,      # hole diameter: 1 mm
    p  = 1e-3,      # pitch: 1 mm (so hole 2 center at 1.8 mm)
    l1 = 0.5e-3,    # anode at 0.5 mm
    l2 = 1.5e-3,    # gate at 1.5 mm
    l3 = 2.0e-3,    # top electrode at 2.0 mm
    Va = -100.0,    # anode potential
    Vg = -10000.0,  # gate potential
    V0 = -10400.0   # top potential
)

# Domain in x is from x1 - p/2 to (x1+p) + p/2.
x_min = params2D.x1 - params2D.p/2
x_max = (params2D.x1 + params2D.p) + params2D.p/2



# Compute the potential.
phi_mat, xgrid, zgrid = solve_laplace_gala_2d(params2D; Nx=51, Nz=101)
# Transport electrons from top electrode (z = l3) until they reach the anode (z = params2D.l1).
trajectories = transport_electrons_2d(phi_mat, xgrid, zgrid, params2D.l1; N_electrons=20, ds=1e-6)
# Plot electron trajectories.
plot_electron_trajectories_2d(trajectories; xlims=(minimum(xgrid), maximum(xgrid)), ylims=(minimum(zgrid), maximum(zgrid)))

#############################
# 4. Plot Electron Trajectories
#############################
"""
    plot_electron_trajectories_2d(trajectories; xlims, ylims)

Plots the electron trajectories on an (x,z) plane.
Trajectories are drawn in blue with thin lines.

Arguments:
  - trajectories: Vector of trajectories (each a vector of [x, z] points).
  - xlims: Tuple for x-axis limits.
  - ylims: Tuple for z-axis limits.
Returns the plot.
"""
function plot_electron_trajectories_2d(trajectories; xlims=(0.0, 2e-3), ylims=(0.0, 2e-3))
    plt = plot(legend=false, xlabel="x [m]", ylabel="z [m]",
               title="Electron Trajectories", aspect_ratio=:equal,
               xlims=xlims, ylims=ylims)
    for traj in trajectories
        x_traj = [p[1] for p in traj]
        z_traj = [p[2] for p in traj]
        plot!(plt, x_traj, z_traj, lw=1, color=:blue)
    end
    display(plt)
    return plt
end


md"""
# GALA Problem in 2D (x,z) with Two Holes

In this formulation the problem is set up in the (x,z) plane. The vertical (z) dimension is the same as before:
- \( z \in [0, l_3] \), with the bottom at \( z=0 \) (ground, \(\phi=0\)) and the top electrode at \( z=l_3 \) (with \(\phi=V_0\)).
- Intermediate electrode planes (anode at \( z=l_1 \) and gate at \( z=l_2 \)) have prescribed potentials only within the holes.

In the horizontal (x) direction the device contains two holes:
- **Hole 1** is centered at \( x_1 \) and has a diameter \( d_1 \).
- **Hole 2** is centered at \( x_2 = x_1 + p \), where \( p \) is the pitch between holes.
- The region of interest in \( x \) is defined as
  \[
  x \in \left[x_1 - \frac{p}{2},\; x_2 + \frac{p}{2}\right].
  \]
  
**Boundary conditions in x:**  
At \( x = x_{\min} = x_1 - p/2 \) and \( x = x_{\max} = x_2 + p/2 \), Neumann conditions (zero normal derivative) are imposed.

**Electrode Conditions in z:**  
- At \( z=0 \): \(\phi(x,0)=0\).  
- At \( z=l_3 \): \(\phi(x,l_3)=V_0\).  
- At \( z=l_1 \) (anode) and \( z=l_2 \) (gate), Dirichlet conditions are imposed **only within the holes**. A point \((x,z)\) on one of these planes is inside a hole if:
  - For Hole 1: \( x \in \left[x_1 - \frac{d_1}{2},\; x_1 + \frac{d_1}{2}\right] \).
  - For Hole 2: \( x \in \left[x_2 - \frac{d_1}{2},\; x_2 + \frac{d_1}{2}\right] \).

**Finite Difference Algorithm:**
1. **Grid Generation:**  
   Uniform grids are created in \(x\) (from \(x_{\min}\) to \(x_{\max}\)) and in \(z\) (from \(0\) to \(l_3\)).

2. **Assembly:**  
   For interior points, central differences approximate the second derivatives:
   - In \(z\):  
     \(\frac{\partial^2 \phi}{\partial z^2} \approx \frac{\phi(x,z+\Delta z) - 2\phi(x,z) + \phi(x,z-\Delta z)}{\Delta z^2}\).
   - In \(x\):  
     \(\frac{\partial^2 \phi}{\partial x^2} \approx \frac{\phi(x+\Delta x,z) - 2\phi(x,z) + \phi(x-\Delta x,z)}{\Delta x^2}\).
   
   At boundaries in \(x\), one-sided differences are used to enforce Neumann (zero derivative) conditions; at \(z=0\) and \(z=l_3\) the potential is fixed. At \(z=l_1\) and \(z=l_2\) the potential is fixed only for grid points inside the holes.
   
   This yields a sparse linear system \(A\phi = b\), which is solved to get \(\phi(x,z)\).

3. **Electric Field:**  
   The electric field is given by  
   \[
   \mathbf{E}(x,z) = -\nabla \phi(x,z),
   \]
   and its components are approximated by finite differences (forward/backward at boundaries, central in the interior).

4. **Electron Transport:**  
   Electrons are initialized at \(z = l_3\) at uniformly spaced \(x\)-positions (within the domain). Then, using Euler integration, each electron is transported along the negative gradient of the potential (i.e. opposite to the electric field, since electrons are negatively charged) until the electron reaches the anode (i.e. \(z \le l_1\)).

---
"""