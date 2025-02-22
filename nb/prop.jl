### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f4813204-e644-11ef-0ce3-8bb694f90094
using Pkg; Pkg.activate("/Users/jjgomezcadenas/Projects/JNEXT")

# ╔═╡ 2a1ae100-8989-46bf-82a3-dfda83b19ac7
begin
	using PlutoUI
	using CSV
	using DataFrames
	using Images
	using Plots
	using Random
	using Test
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
	using OrdinaryDiffEq
end

# ╔═╡ 0d5a3deb-d370-4fb6-aaa6-e513c87e6a3b
import Unitful:
nm, μm, mm, cm, m, km,
mg, g, kg,
fs, ps, ns, μs, ms, s, minute, hr, d, yr, Hz, kHz, MHz, GHz,
eV, keV, MeV,
μJ, mJ, J,
μW, mW, W,
A, N, mol, mmol, V, L, M

# ╔═╡ 3d9e9dff-2ff9-4165-9387-69e79a31e6e9
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

# ╔═╡ fe9bcc95-993a-45b8-a7a9-89433195b6a0
jn = ingredients("../src/JNEXT.jl")

# ╔═╡ 460ac14b-a4e0-402c-982b-7e16e50bc655
PlutoUI.TableOfContents(title="Propagator", indent=true)

# ╔═╡ b42c1e00-31ee-4bac-8988-139c1bffa1bf
md"""
# GALA (Cylindrical Coordinates)

## 1. Physical Setup

- **Conducting Plates:**
  - **Anode:** Located at ``z = 0`` (ground potential).
  - **Gate:** Located at  ``z = l`` (voltage ``V`` ).
- A dielectric fills the space between the plates.
- A cylindrical hole (of diameter ``d`` ) is drilled along the center.
- Under the assumption of cylindrical symmetry, the system is described using two coordinates:
  - The radial coordinate ``\rho``
  - The vertical coordinate ``z``

## 2. Electric Potential

We model the electric potential as:
``
\phi(\rho,z) = \frac{V}{l}\,z + A\,\exp\left(-\frac{\rho^2}{w^2}\right) \sin\left(\pi\,\frac{z}{l}\right),
``
where:
- ``\frac{V}{l}\,z`` represents the linear potential variation between the plates,
- ``A`` is the amplitude of the perturbation (typically a fraction of ``V``),
- ``w`` is a characteristic width (typically of order ``d``).

## 3. Electric Field

The electric field is the negative gradient of ``\phi``. In cylindrical coordinates (and neglecting any ``\theta``-dependence), the components are:

- **Radial Component:**
  ``
  E_\rho(\rho,z) = -\frac{\partial \phi}{\partial \rho} = \frac{2A\,\rho}{w^2}\,\exp\left(-\frac{\rho^2}{w^2}\right) \sin\left(\pi\,\frac{z}{l}\right)
  ``
- **Vertical Component:**
  ``
  E_z(\rho,z) = -\frac{\partial \phi}{\partial z} = -\frac{V}{l} - \frac{A\pi}{l}\,\exp\left(-\frac{\rho^2}{w^2}\right) \cos\left(\pi\,\frac{z}{l}\right)
  ``

## 4. Electron Trajectory Integration

Assuming that electrons follow the electric field lines, their trajectories satisfy:
``
\frac{d\rho}{ds} = \frac{E_\rho(\rho,z)}{\sqrt{E_\rho^2(\rho,z) + E_z^2(\rho,z)}}, \qquad \frac{dz}{ds} = \frac{E_z(\rho,z)}{\sqrt{E_\rho^2(\rho,z) + E_z^2(\rho,z)}},
``
where ``s`` is a parameter along the field line. Integration (using, for example, a fourth-order Runge–Kutta method) is carried out until the electron reaches the anode (i.e. when ``z \le 0``) or a predetermined maximum number of steps is reached.

## 5. Seed Point Generation

Seed points for the electron trajectories are generated at ``z = 2l``. In this simplified ``(\rho,z)`` formulation, the radial coordinate ``\rho`` is chosen uniformly in the interval ``[0, 2d]``.

## Summary

1. **Input:**
   - Physical parameters: ``V``, ``l``, ``d``, ``A``, and ``w``.

2. **Electric Potential:**
   ``
   \phi(\rho,z) = \frac{V}{l}\,z + A\,\exp\left(-\frac{\rho^2}{w^2}\right) \sin\left(\pi\,\frac{z}{l}\right)
   ``

3. **Electric Field Components:**
   ``
   E_\rho(\rho,z) = \frac{2A\,\rho}{w^2}\,\exp\left(-\frac{\rho^2}{w^2}\right) \sin\left(\pi\,\frac{z}{l}\right)
   ``
   ``
   E_z(\rho,z) = -\frac{V}{l} - \frac{A\pi}{l}\,\exp\left(-\frac{\rho^2}{w^2}\right) \cos\left(\pi\,\frac{z}{l}\right)
   ``

4. **Seed Points:**
   - Generate seed points at ``z = 2l`` with ``\rho`` uniformly distributed in ``[0, 2d]``.

5. **Integrate Trajectories:**
   - For each seed point, integrate:
     ``
     \frac{d\rho}{ds} = \frac{E_\rho(\rho,z)}{\sqrt{E_\rho^2(\rho,z) + E_z^2(\rho,z)}}, \qquad \frac{dz}{ds} = \frac{E_z(\rho,z)}{\sqrt{E_\rho^2(\rho,z) + E_z^2(\rho,z)}}
     ``
     until the electron reaches ``z \le 0``.

This simplified two-dimensional model in ``(\rho,z)`` captures the essential physics of the original GALA problem under the assumption of cylindrical symmetry.
"""

# ╔═╡ f08d6c33-20c0-4d97-bdd1-68ddb5c4e104
md"""
# Revised GALA Problem: Modified Boundary Conditions and Potential

In this revised version of the GALA problem the device is immersed in an insulating medium (for example, xenon gas) and we impose four different horizontal boundary conditions in the ``z``–direction. The conductors are placed at

- ``z = 0``: An infinite conductor at ground (``V=0``).
- ``z = l``: The anode at potential ``V_a`` (e.g. ``V_a = -\varepsilon_2``, say ``-100`` V).
- ``z = 2l``: The GALA gate at potential ``V_g`` (e.g. ``10^4`` V).
- ``z = 3l``: Another infinite conductor at potential ``V_0`` (with the extra condition ``V_0 = V_g + \varepsilon``, e.g. ``10^4 + 400 = 10400`` V).

We wish to compute the electric field (and then field lines and electron trajectories) in the region
``
\rho \in [0, d] \quad \text{and} \quad z \in [0, 3l],
``
where ``d`` is the diameter of the hole.

---

## 1. Constructing the Potential

### a) The Background Potential `` \phi_{\text{bg}}(z) ``

Without any lateral (radial) perturbation the simplest solution to Laplace’s equation (in one dimension, along ``z``) that satisfies the prescribed boundary conditions is an interpolation between the values at ``z=0``, ``l``, ``2l``, and ``3l``. For example, one may choose a cubic (or higher–order) polynomial `` \phi_{\text{bg}}(z) `` such that

- ``\phi_{\text{bg}}(0) = 0``,
- ``\phi_{\text{bg}}(l) = V_a``,
- ``\phi_{\text{bg}}(2l) = V_g``,
- ``\phi_{\text{bg}}(3l) = V_0``.

This function describes the “global” voltage profile in the ``z``–direction that is imposed by the conductors. (In practice one might solve for the unique cubic interpolant or even use piecewise linear segments.)

### b) The Localized Perturbation

The presence of the hole (an aperture in the dielectric) introduces a disturbance in the electric field that is most significant near the hole and decays away radially. To model this, we add a perturbation term that is both localized in the radial direction and “damped” at the ``z``–boundaries so as not to disturb the imposed potentials.

A natural choice is a product of:
- A **Gaussian** in the radial coordinate,
  ``
  \exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr),
  ``
  where ``w`` is a width parameter (typically on the order of ``d``).
- A function ``\Psi(z)`` that vanishes at the boundaries ``z=0``, ``z=l``, ``z=2l``, and ``z=3l``. One simple (though not unique) choice is to use a polynomial with roots at these points. For example,
  ``
  \Psi(z) = (z)(z-l)(z-2l)(z-3l).
  ``
  (Any function with these zeros works; sometimes one prefers trigonometric functions, but here a polynomial is a simple choice.)

### c) The Full Potential

By superposition, we can write the overall potential as:
``
\phi(\rho,z) = \phi_{\text{bg}}(z) + A\,\exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr) \,\Psi(z),
``
or explicitly,
``
\phi(\rho,z) = \phi_{\text{bg}}(z) + A\,\exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr) \,(z)(z-l)(z-2l)(z-3l).
``
Here, the amplitude ``A`` controls the strength of the perturbation. Notice that because ``\Psi(z)`` vanishes at ``z=0``, ``z=l``, ``z=2l``, and ``z=3l``, the boundary values of ``\phi`` remain exactly those prescribed by ``\phi_{\text{bg}}(z)``.

---

## 2. Physical Justification and Derivation

### a) Satisfying Boundary Conditions

- **At ``z=0``:**  
  ``\phi( \rho,0 ) = \phi_{\text{bg}}(0) + A\,\exp(-\rho^2/w^2)\,\Psi(0) = 0 + A \cdot (\text{anything}) \cdot 0 = 0.``
  
- **At ``z=l``:**  
  ``\phi( \rho,l ) = \phi_{\text{bg}}(l) + A\,\exp(-\rho^2/w^2)\,\Psi(l) = V_a + 0 = V_a.``
  
- **At ``z=2l``:**  
  ``\phi( \rho,2l ) = V_g,`` since ``\Psi(2l)=0.``
  
- **At ``z=3l``:**  
  ``\phi( \rho,3l ) = V_0,`` since ``\Psi(3l)=0.``

Thus, the extra perturbation does not alter the potential at the conductors.

### b) Derivation from Laplace’s Equation

In regions without free charge, the potential must satisfy Laplace’s equation,
``
\nabla^2 \phi = 0.
``
In a region with cylindrical symmetry the equation is
``
\frac{1}{\rho}\frac{\partial}{\partial \rho}\!\Bigl(\rho\,\frac{\partial \phi}{\partial \rho}\Bigr) + \frac{\partial^2 \phi}{\partial z^2} = 0.
``
A common method of solution is separation of variables. For the unperturbed case (with only ``z`` dependence) one obtains the linear solution ``\phi(z) = \phi_{\text{bg}}(z)``. The perturbation due to the hole, however, introduces ``\rho``–dependence. A complete solution would involve an expansion in Bessel functions in ``\rho`` and sine (or cosine) functions in ``z``.  
  
Our choice
``
\exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr)
``
mimics the rapid decay of higher–order Bessel functions for large ``\rho``, while the factor ``\Psi(z)`` (with zeros at the boundaries) is consistent with the separation–of–variables solution in ``z``. Although this potential is an approximation and does not represent the full solution of Laplace’s equation for the complicated geometry, it is a “sound” model in that it:
  
- **Respects the boundary conditions.**
- **Captures the localized nature of the perturbation** due to the hole.
- **Is simple enough for analytical and numerical study** of field lines and electron trajectories.

---

## 3. Field Lines and Electron Trajectories

With the potential defined as above, the electric field is given by
``
\mathbf{E}(\rho,z) = -\nabla \phi(\rho,z),
``
with components:
- **Radial:**
  ``
  E_\rho(\rho,z) = -\frac{\partial \phi}{\partial \rho} = -\frac{\partial \phi_{\text{bg}}}{\partial \rho} - A \left(-\frac{2\rho}{w^2}\right) \exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr) \Psi(z).
  ``
  (Note that the background potential ``\phi_{\text{bg}}`` is independent of ``\rho``, so its derivative is zero.)
- **Vertical:**
  ``
  E_z(\rho,z) = -\frac{\partial \phi}{\partial z} = -\phi_{\text{bg}}'(z) - A\,\exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr) \Psi'(z).
  ``

Electron trajectories are then computed by integrating
``
\frac{d\rho}{ds} = \frac{E_\rho(\rho,z)}{\sqrt{E_\rho^2 + E_z^2}}, \qquad
\frac{dz}{ds} = \frac{E_z(\rho,z)}{\sqrt{E_\rho^2 + E_z^2}},
``
which gives the path of an electron moving along the field line.

The simulation is performed in the region
``
\rho \in [0, d] \quad \text{and} \quad z \in [0, 3l],
``
where the lateral boundary at ``\rho = d`` approximates the edge of the hole and the ``z`` boundaries are set by the conductors.

---

## 4. Conclusion

The proposed potential
``
\phi(\rho,z) = \phi_{\text{bg}}(z) + A\,\exp\!\Bigl(-\frac{\rho^2}{w^2}\Bigr) (z)(z-l)(z-2l)(z-3l)
``
is justified because:

1. **Boundary Conditions:** It satisfies ``\phi(0,z) = \phi_{\text{bg}}(z)`` and ``\phi(\rho,0) = 0``, ``\phi(\rho,l) = V_a``, ``\phi(\rho,2l) = V_g``, and ``\phi(\rho,3l) = V_0`` by design.
2. **Local Perturbation:** The Gaussian factor confines the perturbation near the center (the hole), while the polynomial ``\Psi(z)`` ensures that the perturbation does not alter the imposed conductor potentials.
3. **Physical Motivation:** Although derived approximately, this form mimics the type of solution one would obtain by separating variables in Laplace’s equation for a system with these boundary conditions.

This potential is thus a physically reasonable starting point for computing the electric field lines and electron trajectories in the revised GALA problem.
"""

# ╔═╡ d5792afc-7e93-47f1-a8f5-98f007da6140
md"""
# Analysis
"""

# ╔═╡ 42138717-98f3-432f-9481-0c1d0281a4b5


# ╔═╡ 142c0965-5c05-49c4-bc2c-d84f8b2e84b7
#electron_trajectory_mod(sp[2], params; dt=1e-1, max_steps=1000)

# ╔═╡ 3163342f-f107-4587-9f84-b617545082ce
#plot_field_lines(params; grid_points=20, dt=1e-2, max_steps=10000)

# ╔═╡ 66459668-4944-4934-8015-a859867110cd
#plot_field_lines_mod(params; grid_points=20, dt=1e-2, max_steps=1000)

# ╔═╡ 72e7015c-3337-440e-bf1c-8ac54a76c8c4
md"""
# Functions
"""

# ╔═╡ 6e4890c2-9203-46fa-b40f-092b5d7c31af
"""
    struct GALAParams

Holds simulation parameters for the revised GALA problem:
- `Vg`: Gate potential at z = l₂ (e.g. -10⁴ V)
- `Va`: Anode potential at z = l₁ (e.g. -100 V)
- `V0`: Conductor potential at z = l₃, with V₀ = Vg - ε (e.g. -10400 V if Vg = -10⁴ V and ε = 400 V)
- `l1`: Position of the anode (z = l₁)
- `l2`: Position of the gate (z = l₂)
- `l3`: Position of the upper conductor (z = l₃)
- `d` : Diameter of the hole (sets the radial domain: ρ ∈ [0, d])
- `A` : Amplitude of the perturbation term
- `w` : Width parameter for the Gaussian (typically of order d)
"""
struct GALAParams
    Vg::Float64
    Va::Float64
    V0::Float64
    l1::Float64
    l2::Float64
    l3::Float64
    d::Float64
    A::Float64
    w::Float64
end

# ╔═╡ 1040845e-c934-4b2a-8be5-e2180ddfc3f9
begin
	Vg = -10000.0
	V0 = Vg - 400.0
	Va = -100.0
	ll1 = 5
	ll2 = 10
	ll3 = 10.5
	lh = 5.0
	dh = 5.0
	Ap = -0.01 * Vg
	w = dh
	
	params = GALAParams(Vg, Va, V0, ll1, ll2, ll3, dh, Ap, w)
	md"""
	Parameters: 
	- `Vg`: Gate potential at z = l₂ (e.g. $(params.Vg) V)
	- `Va`: Anode potential at z = l₁ (e.g. $(params.Va) V)
	- `V0`: Conductor potential at z = l₃ (e.g. $(params.V0) V)
	- `l1`: Position of the anode (z = $(params.l1) mm)
	- `l2`: Position of the gate (z = $(params.l2) mm)
	- `l3`: Position of the upper conductor (z = $(params.l3) mm)
	- `d` : Diameter of the hole (sets the radial domain: ρ ∈ [0, d]: =$(params.d) mm)
	- `A` : Amplitude of the perturbation term: A = $(params.A)
	- `w` : Width parameter for the Gaussian: w =$(params.w) mm
	
	"""
end

# ╔═╡ d8f89f9e-aaf2-401b-8205-0848789c265f
begin
	# ========== 1) Compute E(r,z) analytically ===========

"""
    E_rz(r, z, params) -> (E_r, E_z)

Compute the electric field vector in the (r,z) plane from the potential:
    E = -∇φ = -(∂φ/∂r, ∂φ/∂z).

We assume:
  φ(r,z) = φ_bg(z) + A * radial_factor(r) * shape_z(z).

Hence:
  E_r(r,z) = -d/dr [φ_bg(z)] - A * d/dr [ radial_factor(r)*shape_z(z) ]
            = -0  (because φ_bg depends only on z)
              - A * [ d/dr radial_factor(r) ] * shape_z(z)
  E_z(r,z) = -d/dz [φ_bg(z)] - A * radial_factor(r)* d/dz shape_z(z).
"""
function E_rz(r::Float64, z::Float64, params::GALAParams)
    # 1. E_r(r,z) = - ∂φ/∂r
    # The background potential φ_bg(z) depends only on z => ∂φ_bg/∂r = 0
    # So the radial part only comes from the perturbation A*radial_factor(r)*shape_z(z).
    # radial_factor(r) = exp(-r^2 / w^2), so d/dr radial_factor(r) =  -2r/w^2 * exp(-r^2/w^2).
    
    # shape_z(z) is independent of r => factor is shape_z(z).
    
    # => E_r(r,z) = - A * d/dr[ radial_factor(r) ] * shape_z(z).
    
    # 2. E_z(r,z) = - ∂φ/∂z
    #   = - [ d/dz φ_bg(z) + A * radial_factor(r)* d/dz shape_z(z) ].
    
    # => E_z(r,z) = - d/dz φ_bg(z) - A * radial_factor(r)* shape_z'(z).

    # We'll retrieve the needed derivatives from the same logic used in phi_bg or shape_z, etc.

    # (a) d/dr of radial_factor(r):
    w = params.w
    rf = radial_factor(r, params)   # = exp(-r^2 / w^2)
    drf_dr = - (2*r)/(w^2) * rf
    
    # (b) d/dz of shape_z(z):
    # shape_z(z) = z*(z-l1)*(z-l2)*(z-l3)
    # => derivative is easily computed by product rule or we can define function shape_z_prime
    dsz_dz = shape_z_prime(z, params)
    
    # (c) d/dz of φ_bg(z):
    dbg_dz = dphi_bg_dz2(z, params)
    
    # E_r:
    E_r = - params.A * drf_dr * shape_z(z, params)
    
    # E_z:
    E_z = -  dbg_dz + params.A * rf * dsz_dz 
    
    return (E_r, E_z)
end

"""
    shape_z_prime(z, params)

Compute derivative of shape_z(z) = z*(z-l1)*(z-l2)*(z-l3).
"""
function shape_z_prime(z::Float64, params::GALAParams)
    l1, l2, l3 = params.l1, params.l2, params.l3
    # shape_z(z) = z*(z-l1)*(z-l2)*(z-l3)
    # We'll do a direct symbolic expansion or the product rule. Let's do a product rule:

    # d/dz [z*(z-l1)*(z-l2)*(z-l3)]
    #   = (z-l1)*(z-l2)*(z-l3) +
    #     z * [ d/dz( (z-l1)*(z-l2)*(z-l3) ) ]
    # That sub-derivative is  sum of partial expansions. For brevity, let's just define it:

    # Or define a short approach:
    val_plus  = (z - l1)*(z - l2)*(z - l3)
    # partial derivative of (z-l1)*(z-l2)*(z-l3) w.r.t. z => sum of 3 terms
    partial   = (z - l2)*(z - l3) + (z - l1)*(z - l3) + (z - l1)*(z - l2)
    
    return val_plus + z*partial
end

"""
    dphi_bg_dz2(z, params)

Compute derivative of the cubic polynomial background φ_bg(z).
Used in E_z(r,z).
"""
function dphi_bg_dz2(z::Float64, params::GALAParams)
    # We solved for a, b, c s.t. φ_bg(z) = a*z^3 + b*z^2 + c*z
    # => d/dz φ_bg(z) = 3a z^2 + 2b z + c
    Va, Vg, V0 = params.Va, params.Vg, params.V0
    l1, l2, l3 = params.l1, params.l2, params.l3
    
    M = [l1^3  l1^2  l1;  l2^3  l2^2  l2;  l3^3  l3^2  l3]
    bcs = [Va, Vg, V0]
    coeffs = M \ bcs
    a, b, c = coeffs
    
    return 3a*z^2 + 2b*z + c
end
	"""
    radial_factor(r, params)

Returns a radial "weight" function, for example:
  exp(-r^2 / w^2)
Localizing the perturbation near r=0 (the hole region).
"""
function radial_factor(r::Float64, params::GALAParams)
    return exp(- (r^2)/(params.w^2))
end

"""
    shape_z(z, params)

Define a shape function in z that vanishes at z=0, l1, l2, l3 so that
the perturbation does not alter the boundary conditions. For instance:
  (z)*(z - l1)*(z - l2)*(z - l3).
"""
function shape_z(z::Float64, params::GALAParams)
    return z * (z - params.l1)*(z - params.l2)*(z - params.l3)
end
end

# ╔═╡ 30220ad6-6710-450d-99e0-753b3b256585
function test_field_boundaries(params)
    @testset "Electric Field near boundaries" begin
        
        
        # We'll check E(r=0, z=0).
        # This might be interesting: we expect that at z=0 => potential=0, so we might not necessarily
        # expect E=some big number or zero. But let's just ensure we don't blow up:

        E00 = E_rz(0.0, params.l3, params)
		println(E00)
        @test isfinite(E00[1]) && isfinite(E00[2])
        
        # Similarly check at z=l1 => We might see some finite E. We can do a quick check:
        El1 = E_rz(0.0, params.l1, params)
        @test isfinite(El1[1]) && isfinite(El1[2])
        
        # etc. If we want more rigorous checks, we can add them. For instance, near r=0, we expect E_r=0
        # from the radial factor derivative. Let's do a small check at r=0:
        
        # Actually let's do a functional check: at r=0 => d/d r radial_factor(0)=0 => E_r(0,z) might be 0?
        # Because drf_dr(0)= -2*0/(w^2) * exp(...) = 0 => E_r => 0 => check that:
        
        # We'll just test for some z in [0, l3].
        for z in [0.0, params.l1, params.l2, params.l3]
            Etest = E_rz(0.0, z, params)
            @test isapprox(Etest[1], 0.0, atol=1e-8)
        end
    end
end


# ╔═╡ f06c35aa-1970-4077-9e6f-b224e9f0386e
test_field_boundaries(params)

# ╔═╡ dd46c25c-75ad-4e0c-88fe-c70228d1fb18
"""
    plot_field_arrows(params; rmax, nr, nz, scale)

Visualize the electric field in the (r,z) plane using quiver (arrow) plots.
- We'll sample (r,z) in a grid [0, rmax] x [0, l3].
- Then compute E_r and E_z.
- The "scale" controls arrow size for readability.
"""
function plot_field_arrows(params::GALAParams; rmax=5.0, nr=20, nz=40, scale=0.1)
    # We'll define a grid
    rs = range(0, rmax, length=nr)
    zs = range(0, params.l3, length=nz)
    
    # We'll accumulate arrays for Plots.jl's quiver functionality:
    # quiver(x, y, quiver=(u, v))
    X = Float64[]
    Y = Float64[]
    U = Float64[]
    V = Float64[]
    
    for z in zs
        for r in rs
            push!(X, r)
            push!(Y, z)
            (Er, Ez) = E_rz(r, z, params)
            push!(U, scale*Er)
            push!(V, scale*Ez)
        end
    end
    
    plt = quiver(X, Y, quiver=(U, V), aspect_ratio=:equal,
                 title="Electric Field (r,z)", xlabel="r", ylabel="z")
end

# ╔═╡ 31934950-dd0a-4758-82d8-700a3e9f7605
plot_field_arrows(params; rmax=5.0, nr=20, nz=40, scale=0.1)

# ╔═╡ bc54303b-3ac6-43a9-a241-11e967ab2d40
begin
	# ========== (A) ODE Definition for Electron ==========

"""
    drift_electron(r, z, params)

Return the 2D drift vector for an electron at (r, z), i.e.
  dr/ds, dz/ds = -E(r,z)/||E(r,z)||,
where E(r,z) = (E_r, E_z).
We use a small check for E=0 to avoid NaNs.
"""
function drift_electron(r::Float64, z::Float64, params::GALAParams)
    (Er, Ez) = E_rz(r, z, params)  # Suppose you have a function E_rz(r,z)->(Er,Ez)

    normE = sqrt(Er^2 + Ez^2)
    if normE < 1e-12
        # If field is extremely small, return zero drift => electron "stalls"
        return (0.0, 0.0)
    end
    # Electron moves opposite to E => factor -1

    return (-Er / normE, -Ez / normE)
end

# ========== (B) RK4 Integrator for (r,z) ==========

"""
    rk4_step_2D(rz, ds, params, driftfun)

Perform a single 4th-order Runge–Kutta step for the ODE:
   d(r,z)/ds = driftfun(r,z,params),
with step size ds. Return the new (r,z).
"""
function rk4_step_2D(rz::NTuple{2,Float64}, ds::Float64,
                     params::GALAParams,
                     driftfun::Function)
    # current point
    (r0, z0) = rz
   
    k1 = driftfun(r0, z0, params)
	
    r1 = (r0 + 0.5*ds*k1[1], z0 + 0.5*ds*k1[2])
	 
    k2 = driftfun(r1[1], r1[2], params)
    r2 = (r0 + 0.5*ds*k2[1], z0 + 0.5*ds*k2[2])
    
    k3 = driftfun(r2[1], r2[2], params)
    r3 = (r0 + ds*k3[1], z0 + ds*k3[2])
    
    k4 = driftfun(r3[1], r3[2], params)
    
    new_r = r0 + ds/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    new_z = z0 + ds/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    
    return (new_r, new_z)
end

# ========== (C) Single Electron Trajectory ==========

"""
    electron_trajectory_mod(initial, params; ds, max_steps)

Integrate the field–line ODE starting at `initial` = (r,z).
We do steps of size `ds`. The electron stops when `z <= l1`
(meaning it reached or crossed the anode).
Returns an array of positions (r,z).
"""
function electron_trajectory_modx(initial::NTuple{2,Float64},
                                 params::GALAParams;
                                 ds=1e-3, max_steps=10_000)
    # store the trajectory
    traj = Vector{NTuple{2,Float64}}()
    push!(traj, initial)
    rcur, zcur = initial

    for _ in 1:max_steps
        # stop if electron reached or crossed the anode
        if zcur <= params.l1
            break
        end
        (rnew, znew) = rk4_step_2D((rcur,zcur), ds, params, drift_electron)
		
        # optionally ensure we don't get negative r
        if rnew < 0
            rnew = 0
        end
        push!(traj, (rnew, znew))
        rcur, zcur = rnew, znew
    end
    return traj
end

# ========== (D) Multiple Electrons (seed at z=l3) ==========

"""
    generate_seed_points(n, params)

Generate n seed points at z=l3, with r in [0, d].
"""
function generate_seed_points(n::Int, params::GALAParams)
    seeds = Vector{NTuple{2,Float64}}()
    for _ in 1:n
        r = (params.d)*rand()  # uniform in [0, d]
        z = params.l3
        push!(seeds, (r, z))
    end
    return seeds
end


"""
    electron_trajectories_modx(n, params; ds, max_steps)

Compute electron trajectories for `n` seed points (r in [0,d], z=l3).
Returns a vector of trajectories (each one is a vector of (r,z) positions).
"""
function electron_trajectories_modx(n::Int, params::GALAParams;
                                   ds=1e-3, max_steps=10_000)
    seeds = generate_seed_points(n, params)
	#println("seeds =$(seeds)")
    trajs = [ electron_trajectory_modx(seed, params, ds=ds, max_steps=max_steps)
              for seed in seeds ]
    return trajs
end

# ========== (E) Test ==========


end

# ╔═╡ 27a8e777-37d1-44a5-8747-9579fdbcc18b
function test_trajectories(params; n=1, ds=1e-1, max_steps=10)
    @testset "Electron Trajectories" begin
        # Let's do e.g. 5 seeds
        n=5
       
        trajs = electron_trajectories_modx(n, params, ds=ds, max_steps=max_steps)
        
        # Check: each trajectory ends with z <= l1
        for tr in trajs
            @test tr[end][2] <= params.l1 + 1e-9
        end
    end
end

# ╔═╡ 4354c71c-b20b-4da1-b4a5-ca8d05906ade
test_trajectories(params; n=1, ds=1e-5, max_steps=100000)

# ╔═╡ 7972c25a-db03-4aca-905c-88fc183cae37
begin
	# 1) Electron ODE in (r,z)
function electron_ode!(du, u, p, t)
    # u = (r, z)
    r, z = u
    params = p  # pass GALAParams as "p" in the ODE problem

    Er, Ez = E_rz(r, z, params)
    normE = sqrt(Er^2 + Ez^2)
    if normE < 1e-12
        # If field is extremely small => zero drift
        du[1] = 0.0
        du[2] = 0.0
    else
        # electron velocity = -E / ||E||
        du[1] = -Er / normE
        du[2] = -Ez / normE

		println("erNorm = $(du[1]), ezNorm=$(du[2])")
    end
end

# 2) Condition to stop when z <= l1 (the anode)
function electron_stop_condition(u, t, integrator)
    # stop if z <= l1
    z = u[2]
    return z - integrator.p.l1
end

# Return true if event is triggered (zero crossing)
function electron_stop_affect!(integrator, event_index)
    terminate!(integrator)
end

# 3) Solve for one electron:
function electron_trajectory_diffeq(initial::NTuple{2,Float64},
                                    params::GALAParams;
                                    max_t=100.0)
    # initial => e.g. (r0, z0) at top
    # We'll treat "s" as a pseudo-time in the ODE
    # create ODEProblem

	u0 = [initial[1], initial[2]]  # (r=0, z=10.5) for example

    prob = ODEProblem(electron_ode!, u0, (0.0, max_t), params)
    
    # define a callback for stopping at z <= l1
    condition = electron_stop_condition
    affect!   = electron_stop_affect!
    cb = ContinuousCallback(condition, affect!)
    
    # solve using e.g. Tsit5
    sol = solve(prob, Tsit5(); callback=cb, dt=1e-2, maxiters=1e3, abstol=1e-7, reltol=1e-5)
    
    # The solution is a trajectory in "time" s => we can gather (r,z) from sol.
    # We'll return them as a vector of points.
    traj = [ (sol(t)[1], sol(t)[2]) for t in sol.t ]
    return traj
end

# 4) Multiple seeds
function electron_trajectories_diffeq(n, params; max_t=100.0)
    seeds = [ (params.d * rand(), params.l3) for _ in 1:n ]  # r in [0,d], z=l3
    [ electron_trajectory_diffeq(s, params; max_t=max_t) for s in seeds ]
end

# 5) Plot
function plot_electron_trajectories_diffeq(trajs)
    plt = plot(title="Electron Trajectories in (r,z)", xlabel="r", ylabel="z", legend=false)
    for tr in trajs
        rs = [p[1] for p in tr]
        zs = [p[2] for p in tr]
        plot!(plt, rs, zs, lw=2)
    end
    plt
end

end

# ╔═╡ ca19c8a0-426a-4f4b-9ca6-9230f453f572
trajs = electron_trajectories_diffeq(1, params; max_t=200.0)

# ╔═╡ 9355cac7-eff5-4c2d-bae8-58fc1c598019
"""
    phi_bg2(z, params)

Compute the background potential φ_bg(z) as a cubic polynomial in the scaled variable
  x = z / l₃.

We assume:

  φ\\_bg(z) = a*(z/l₃)^3 + b*(z/l₃)^2 + c*(z/l₃),

with the conditions:

  φ\\_bg(l₁) = Vₐ,   φ\\_bg(l₂) = Vg,   φ\\_bg(l₃) = V₀.

Let x₁ = l₁/l₃, x₂ = l₂/l₃, and x₃ = 1.

Then we solve:
  a*x₁^3 + b*x₁^2 + c*x₁ = Vₐ,

  a*x₂^3 + b*x₂^2 + c*x₂ = Vg,

  a*1^3   + b*1^2   + c*1   = V₀.

"""
function phi_bg2(z::Float64, params::GALAParams)
    x1 = params.l1 / params.l3
    x2 = params.l2 / params.l3
    x3 = 1.0
    bcs = [params.Va, params.Vg, params.V0]
    M = [x1^3  x1^2  x1;
         x2^3  x2^2  x2;
         x3^3  x3^2  x3]
    coeffs = vec(M \ bcs)   # convert to a standard vector
    a, b, c = coeffs
    x = z / params.l3
    return a*x^3 + b*x^2 + c*x
end



# ╔═╡ 3eff442d-a523-44b1-883e-76e03796ad2b
"""
    dphi_bg_dz(z, params)

Compute the derivative of the background potential with respect to z.
Given:

  φ\\_bg(z) = a*(z/l₃)^3 + b*(z/l₃)^2 + c*(z/l₃),

its derivative is:

  φ\\_bg'(z) = [3a*(z/l₃)^2 + 2b*(z/l₃) + c] / l₃.
"""
function dphi_bg_dz(z::Float64, params::GALAParams)
    x = z / params.l3
    x1 = params.l1 / params.l3
    x2 = params.l2 / params.l3
    x3 = 1.0
    bcs = [params.Va, params.Vg, params.V0]
    M = [x1^3  x1^2  x1;
         x2^3  x2^2  x2;
         x3^3  x3^2  x3]
    coeffs = vec(M \ bcs)
    a, b, c = coeffs
    return (3*a*x^2 + 2*b*x + c) / params.l3
end

# ╔═╡ 451e6ffb-f05a-408a-be17-12ad8e1605e3
# =================== Background Potential ===================

"""
    phi_bg(z, params)

Compute the background potential φ\\_bg(z) as the unique cubic polynomial
(with zero constant term, so that φ\\_bg(0)=0) that satisfies:

  φ\\_bg(l₁) = Va,  φ\\_bg(l₂) = Vg,  φ\\_bg(l₃) = V0.

We assume:
  φ\\_bg(z) = a*z^3 + b*z^2 + cz.

The coefficients a, b, c are determined by solving:

  a*l₁^3 + b*l₁^2 + cl₁ = Va,

  a*l₂^3 + b*l₂^2 + cl₂ = Vg,

  a*l₃^3 + b*l₃^2 + cl₃ = V0.

The backslash operator solves the system M x = bcs for the vector x (which, in this case, contains the coefficients a, b, and c). Thus, after executing

coeffs = M \\ bcs

the variable coeffs is a vector containing the values of a, b, and c that satisfy the equations above

"""
function phi_bg(z::Float64, params::GALAParams)
    l1, l2, l3 = params.l1, params.l2, params.l3
    bcs = [params.Va, params.Vg, params.V0]
    M = [l1^3  l1^2  l1;
         l2^3  l2^2  l2;
         l3^3  l3^2  l3]
    coeffs = M \ bcs  # Solve for [a, b, c]
    a, b, c = coeffs
    return a*z^3 + b*z^2 + c*z
end

# ╔═╡ 23f35b47-d898-463d-a141-16814d88bd4c
function ϕ(r::Float64, z::Float64, params::GALAParams)
    # Background
    φ_bg = phi_bg(z, params)
    # Radial factor
    rf   = exp( - (r^2)/(params.w^2) )
    # Sine-based shape in z
    shz  = shape_z_sin(z, params.l1, params.l3)
    # Combine
    return φ_bg + params.A * rf * shz
end

# ╔═╡ 5d035c7b-c382-4c23-a163-ce107e195447
# =================== Perturbation Functions ===================

"""
    poly_z(z, params)

Compute the polynomial Ψ(z) = z*(z - l₁)*(z - l₂)*(z - l₃) which vanishes at 
z = 0, l₁, l₂, and l₃.
"""
function poly_zx(z::Float64, params::GALAParams)
    return z * (z - params.l1) * (z - params.l2) * (z - params.l3)
end




# ╔═╡ ee2d4035-c4b0-4dc1-b690-a3876b7eba4a
"""
    dpoly_zx(z, params)

Compute the derivative of Ψ(z) = z*(z - l₁)*(z - l₂)*(z - l₃).

Using the product rule, one obtains:

  Ψ'(z) = (z - l₁)*(z - l₂)*(z - l₃)
          + z * [ (z - l₂)*(z - l₃) + (z - l₁)*(z - l₃) + (z - l₁)*(z - l₂) ].
"""
function dpoly_zx(z::Float64, params::GALAParams)
    term1 = (z - params.l1) * (z - params.l2) * (z - params.l3)
    term2 = (z - params.l2) * (z - params.l3) +
            (z - params.l1) * (z - params.l3) +
            (z - params.l1) * (z - params.l2)
    return term1 + z * term2
end

# ╔═╡ dd9048a5-e730-4809-8d79-eea8ae9239ce
# =================== Revised Perturbation Functions ===================

"""
    poly_z(z, params)

Compute a revised perturbation function Ψ(z) that vanishes with multiplicity 2 at z = 0, l₁, l₂, and l₃.
We choose:
  Ψ(z) = z^2 * (z - l₁)^2 * (z - l₂)^2 * (z - l₃)^2.
"""
function poly_z(z::Float64, params::GALAParams)
    return z^2 * (z - params.l1)^2 * (z - params.l2)^2 * (z - params.l3)^2
end



# ╔═╡ 12383f9f-3cbb-4f2d-98d3-a00346f7f1cf
"""
    dpoly_z(z, params)

Compute the derivative of the revised perturbation function Ψ(z) using a finite difference.
This ensures that both Ψ(z) and Ψ'(z) vanish at z = 0, l₁, l₂, l₃.
"""
function dpoly_z(z::Float64, params::GALAParams; δ=1e-8)
    return (poly_z(z+δ, params) - poly_z(z-δ, params)) / (2δ)
end

# ╔═╡ f28cab00-4ec8-4f6f-b23c-f224d400ae26
# Test that the perturbation function and its derivative vanish at z = 0, l1, l2, l3.
@testset "Perturbation Vanishing Tests" begin
    
    # Define the boundaries where the perturbation should vanish.
    boundaries = [0.0, params.l1, params.l2, params.l3]
    
    for z in boundaries
        p_val = poly_z(z, params)
        dp_val = dpoly_z(z, params; δ=1e-8)
        println("At z = $(z): poly_z = $(p_val), dpoly_z = $(dp_val)")
        
        # We use a small tolerance since floating point arithmetic might not give exact zero.
        @test isapprox(p_val, 0.0, atol=1e-10)
        @test isapprox(dp_val, 0.0, atol=1e-6)
    end
end

# ╔═╡ edb12f79-36ee-474e-8a87-3766ba966c3a
let

z_test = 10.26
# At ρ=0, exp(-0) = 1.
total_dphi_dz = dphi_bg_dz(z_test, params) + params.A * dpoly_z(z_test, params; δ=1e-8)
println("At z = $z_test, dφ_bg/dz = ", dphi_bg_dz(z_test, params))
println("At z = $z_test, A*dΨ/dz = ", params.A * dpoly_z(z_test, params; δ=1e-8))
println("At z = $z_test, total dφ/dz = ", total_dphi_dz)
end

# ╔═╡ e71eb72d-9051-45bd-89c0-afd3bde94d0e
let
	for z in [params.l3, params.l3 - 0.005, params.l3 - 0.01]
    tot_dphi = dphi_bg_dz(z, params) + params.A * dpoly_z(z, params; δ=1e-8)
    println("z = ", z, ": total dφ/dz = ", tot_dphi)
end
end

# ╔═╡ 6438baef-aeca-4cc5-8ccc-665e1eca427a
# =================== Full Potential and Electric Field ===================

"""
    potential_mod(ρ, z, params)

Compute the full potential:

  φ(ρ,z) = φ_bg(z) + A * exp(-ρ²/w²) * Ψ(z),

with Ψ(z) = z*(z-l₁)*(z-l₂)*(z-l₃).
"""
function potential_mod(ρ::Float64, z::Float64, params::GALAParams)
    return phi_bg2(z, params) + params.A * exp(- (ρ^2) / (params.w^2)) * poly_z(z, params)
end

# ╔═╡ d4e54bf8-dbd7-4e4f-83ba-2e5f517d9d47
function plot_potential_contour(params::GALAParams; rmax=1.0, nr=50, nz=50)
    rs = range(0, rmax, length=nr)
    zs = range(0, params.l3, length=nz)
    phimat = [potential_mod(r,z, params) for z in zs, r in rs]
    # Note: we used the array comprehension style. This yields (nz x nr).
    # The "Contour" approach in Plots.jl expects x to map columns, y to map rows, so be mindful
    # we might need to transpose or pass the correct coordinates.
    # We'll define x => rs, y => zs, so we pass phimat but note that row=zs, col=rs => (nz,nr)
    
    contour(rs, zs, phimat,
        xlabel="r", ylabel="z",
        title="Contour of φ(r,z)",
        fill=true,
        color=:viridis, size=(900,600))
end

# ╔═╡ b84ca364-a8c7-42ca-b735-f16a35fb550d
plot_potential_contour(params; rmax=5.0, nr=50, nz=50)

# ╔═╡ 1aedb785-5198-4d32-858b-d9427293fc7e
"""
    E_field_mod(ρ, z, params)

Compute the electric field from the potential via:
  E\\_ρ(ρ,z) = -∂φ/∂ρ,    E\\_z(ρ,z) = -∂φ/∂z.

Here:

  ∂φ/∂ρ = A * (-2ρ/w²) * exp(-ρ²/w²) * Ψ(z),

  ∂φ/∂z = φ_bg'(z) + A * exp(-ρ²/w²) * Ψ'(z).
"""
function E_field_mod(ρ::Float64, z::Float64, params::GALAParams)
    # Radial derivative:
    dphi_dρ = params.A * (-2 * ρ / (params.w^2)) * exp(- (ρ^2) / (params.w^2)) * poly_z(z, params)
    Eρ = -dphi_dρ  # E = -∇φ
    
    # Vertical derivative:
    dphi_dz = dphi_bg_dz(z, params) + params.A * exp(- (ρ^2) / (params.w^2)) * dpoly_z(z, params)
    Ez = -dphi_dz
    return (Eρ, Ez)
end

# ╔═╡ 8cdb2ae6-8952-4d61-9252-f63c2ffcbb16
"""
    E_field_mod(ρ, z, params)

Compute the electric field from the potential via:

  E_ρ(ρ,z) = -∂φ/∂ρ,    E_z(ρ,z) = -∂φ/∂z.

Here:
  ∂φ/∂ρ = A * (-2ρ/w²) * exp(-ρ²/w²) * Ψ(z),

  ∂φ/∂z = φ\\_bg'(z) + A * exp(-ρ²/w²) * Ψ'(z),

with φ\\_bg'(z) computed from the cubic polynomial.
"""
function E_field_modx(ρ::Float64, z::Float64, params::GALAParams)

	#println("E_field: ρ=$(ρ), z = $(z)")
	
    # Radial derivative:
    dphi_dρ = params.A * (-2 * ρ / (params.w^2)) * exp(- (ρ^2) / (params.w^2)) * poly_z(z, params)
    Eρ = -dphi_dρ  # negative gradient

	#println("Eρ=$(Eρ)")
    
    # Vertical derivative:
    # Compute φ_bg'(z) using the same coefficients as in phi_bg:
    l1, l2, l3 = params.l1, params.l2, params.l3
    bcs = [params.Va, params.Vg, params.V0]
    M = [l1^3  l1^2  l1;
         l2^3  l2^2  l2;
         l3^3  l3^2  l3]
    coeffs = M \ bcs
    a, b, c = coeffs
    dphi_bg_dz = 3 * a * z^2 + 2 * b * z + c
    dphi_dz = dphi_bg_dz + params.A * exp(- (ρ^2) / (params.w^2)) * dpoly_z(z, params)
    Ez = -dphi_dz
	#println("Ez=$(Ez)")
    return (Eρ, Ez)
end

# ╔═╡ 13133baf-d118-4cb6-a12f-0724ea168f97

"""Return the Euclidean norm of a 2-vector."""
function norm2(v::Tuple{Float64,Float64})
    return sqrt(v[1]^2 + v[2]^2)
end

# ╔═╡ 943bcd94-2716-41cd-9d9a-b54711a2e749
# =================== ODE Integrator (RK4) in 2D ===================

"""
    rk4_step_mod(r, dt, params)

Take one RK4 step for the 2D system:
   d(ρ, z)/ds = f(ρ, z) = E(ρ,z) / ||E(ρ,z)||,
where E(ρ,z) is computed from the modified potential.
"""
function rk4_step_mod(r::NTuple{2,Float64}, dt::Float64, params::GALAParams)
    # Here, we integrate along -E/||E|| so that electrons move opposite to the electric field.
    f(r) = begin
        E = E_field_mod(r[1], r[2], params)
        n = sqrt(E[1]^2 + E[2]^2)
        return (-E[1] / n, -E[2] / n)
    end
    k1 = f(r)
    r2 = (r[1] + 0.5 * dt * k1[1], r[2] + 0.5 * dt * k1[2])
    k2 = f(r2)
    r3 = (r[1] + 0.5 * dt * k2[1], r[2] + 0.5 * dt * k2[2])
    k3 = f(r3)
    r4 = (r[1] + dt * k3[1], r[2] + dt * k3[2])
    k4 = f(r4)
    new_r = (r[1] + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
             r[2] + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))
    return new_r
end

# ╔═╡ 845ca241-c861-4505-8ad2-b25052f5d8bf
let
	# Set up default parameters and the initial point (ρ=0, z=l₃)

r = (0.5, params.l3)  # ρ = 0, z = l₃

println("Initial point: ", r)

# Compute the electric field at (ρ=0, z=l₃)
E = E_field_mod(r[1], r[2], params)
println("Electric field E(ρ=0, z=l₃) = ", E)

# Compute the normalized field
normE = sqrt(E[1]^2 + E[2]^2)
f_val = (E[1] / normE, E[2] / normE)
println("Normalized field = ", f_val)

# Take one RK4 step with a small step size dt
dt = 1e-2
r_new = rk4_step_mod(r, dt, params)
println("New point after one RK4 step (dt = $(dt)): ", r_new)
	for i in 1:1000
		r = r_new
		r_new = rk4_step_mod(r, dt, params)
		if i%10 == 0
			println("New point after $(i) RK4 step (dt = $(dt)): ", r_new)
		end
	end
end

# ╔═╡ 72ccfbb1-e847-46c5-b442-48affae0b973
let

r = (0.0, params.l3)  # starting at ρ = 0, z = l₃
dt = 0.01             # choose a test step size

println("Testing RK4 integration starting at (ρ, z) = (0, l₃) = (0, $(params.l3))")
# Compute the electric field at the initial point:
E = E_field_mod(r[1], r[2], params)
nE = sqrt(E[1]^2 + E[2]^2)
# For electron drift, we want to integrate along -E/||E||
normalized_electron_field = (-E[1] / nE, -E[2] / nE)
println("At (ρ, z) = (0, l₃):")
println("  Electric field E = ", E)
println("  |E| = ", nE)
println("  Normalized electron field (-E/|E|) = ", normalized_electron_field)

# Take one RK4 step:
r_new = rk4_step_mod(r, dt, params)
println("After one RK4 step (dt=$(dt)): r_new = ", r_new)
end

# ╔═╡ de40bca9-535b-4f7d-ae52-6029e4118be9
# =================== Electron Trajectory Integration in 2D ===================

"""
    electron_trajectory_mod(initial, params; dt, max_steps)

Integrate the 2D field-line ODE starting at `initial` (a tuple (ρ, z)).
Integration proceeds in steps of size `dt` until the electron reaches the anode 
(i.e. when z ≤ 0) or when `max_steps` is exceeded.
Returns an array of positions (tuples) representing the trajectory.
"""
function electron_trajectory_mod(initial::NTuple{2,Float64}, params::GALAParams; dt=1e-5, max_steps=10000)
    traj = Vector{NTuple{2,Float64}}()
    push!(traj, initial)
    r = initial

	#println("max steps =$(max_steps)")
	#println("initial =$(initial)")
	
    for i in 1:max_steps
        r = rk4_step_mod(r, dt, params)
		#println("r =$(r)")
        push!(traj, r)
        # Stop if electron reaches the anode (z ≤ 0)
        if r[2] <= 0
            break
        end
    end
    return traj
end

# ╔═╡ efc08fe1-b64a-455b-947b-36c3b3f4da94
let
    seed = generate_seed_points(20, params)
    
    traj = electron_trajectory_mod(seed[1], params, dt=1e-1, 
		                                  max_steps=100) 
   
    ρs = [t[1] for t in traj] 
	zs = [t[2] for t in traj] 
	println("ρs = $(ρs[end]), zs = $(zs[end])")
    plot(ρs, zs, lw=2)
    
end


# ╔═╡ f5b6f90f-df29-48e6-ac83-4ae57e88ed19
# =================== Seed Point Generation in 2D ===================

"""
    generate_seed_points_mod(n, params)

Generate `n` seed points in the (ρ, z) plane. For this problem, choose seed points at
z = l₃ (upper boundary) with ρ uniformly distributed in [0, d].
"""
function generate_seed_points_mod(n::Int, params::GALAParams)
    seeds = Vector{NTuple{2,Float64}}()
    for i in 1:n
        ρ = params.d * rand()  # uniformly in [0, d]
        z = params.l3          # fixed at z = l₃
        push!(seeds, (ρ, z))
    end
    return seeds
end


# ╔═╡ 2d6e215f-9882-41f3-ba57-c1a4ceba3eae


# ╔═╡ 8c18b522-b741-4109-a9e1-d33e5fc85796
"""
    electron_trajectories_mod(n, params; dt, max_steps)

Compute electron trajectories for `n` seed points in the 2D (ρ, z) geometry.
"""
function electron_trajectories_mod(n::Int, params::GALAParams; dt=1e-5, max_steps=10000)
    seeds = generate_seed_points_mod(n, params)
    trajs = [electron_trajectory_mod(seed, params, dt=dt, max_steps=max_steps) for seed in seeds]
    return trajs
end

# ╔═╡ 4130b7f9-52bb-411f-9915-cbb50c7e1380
"""
    plot_field_lines_mod(params; grid_points, dt, max_steps)

Plot representative field lines in the (ρ, z) plane.
Seed points are generated at z = 2l (forward integration) and at z = 0 (reverse integration).
"""
function plot_field_lines_mod(params::GALAParams; grid_points=20, dt=1e-5, max_steps=10000)
    seeds_top = generate_seed_points_mod(grid_points, params)
    seeds_bottom = [(seed[1], 0.0) for seed in generate_seed_points_mod(grid_points, params)]
    
    field_lines = [electron_trajectory_mod(seed, params, dt=dt, max_steps=max_steps) for seed in seeds_top]
    # Reverse integration: use negative dt starting from z = 0.
    field_lines_rev = [electron_trajectory_mod(seed, params, dt=-dt, max_steps=max_steps) for seed in seeds_bottom]
    
    plt = plot(title="Field Lines in (ρ, z) Plane",
               xlabel="ρ (m)", ylabel="z (m)", legend=false)
    for traj in field_lines
        ρs = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        plot!(plt, ρs, zs, lw=2)
    end
    for traj in field_lines_rev
        ρs = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        plot!(plt, ρs, zs, lw=2, ls=:dash)
    end
    plt
end

# ╔═╡ 3fd1735f-de52-4d5f-afb2-22f11c629bfe
function plot_field_lines(params::GALAParams; grid_points=20, dt=1e-2, 
                          max_steps=10000)
	
    seeds_top = generate_seed_points_mod(grid_points, params)
    
    field_lines = [electron_trajectory_mod(seed, params, dt=dt, max_steps=max_steps) for seed in seeds_top]
	
    
    plt = plot(title="Field Lines in (ρ, z) Plane",
               xlabel="ρ (m)", ylabel="z (m)", legend=false)
    for traj in field_lines
        ρs = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        plot!(plt, ρs, zs, lw=2)
    end
    
    plt
end

# ╔═╡ f3036b4f-463d-4201-8023-e1eba4c10307
function plot_phi(phi, rmin, rmax, zmin, zmax; ns=30)

	r_vals = range(rmin, rmax, length=ns)
	z_vals = range(zmin, zmax, length=ns)
	phirz = [phi(r, z) for r in r_vals, z in z_vals]


	p1 = plot(r_vals, z_vals, phirz,
    	  linetype=:surface,
          legend=false,
    	  xlabel="ρ",
    	  ylabel="z",
    	  zlabel="phi(ρ,z)",
    	  alpha=0.6
	)

	p2 = plot(r_vals, z_vals, phirz,
    	  linetype=:contour,
          legend=false,
    	  xlabel="ρ",
    	  ylabel="z",
    	  zlabel="phi(ρ,z)",
    	  alpha=0.6
	)

	plot(p1,p2, layout=(1,2), size=(1000,600))
end

# ╔═╡ f81b3873-c702-4ff9-9e76-f075286b47b6
function test_potential_mod(params)
   
    # At z = 0, perturbation vanishes and φ_bg(0)=0.
    @test isapprox(potential_mod(0.0, 0.0, params), 0.0, atol=1e-8)
    # At z = l₁, Ψ(l₁)=0 so φ(ρ,l₁)=φ_bg(l₁)=Va.
    @test isapprox(potential_mod(0.0, params.l1, params), params.Va, atol=1e-8)
    # At z = l₂, φ should equal Vg.
    @test isapprox(potential_mod(0.0, params.l2, params), params.Vg, atol=1e-8)
    # At z = l₃, φ should equal V0.
    @test isapprox(potential_mod(0.0, params.l3, params), params.V0, atol=1e-8)
end

# ╔═╡ b24367b3-1ab5-4d28-bfb5-0c956fc3a6da
test_potential_mod(params)

# ╔═╡ 8215e89c-f73b-449c-a3df-9f9ed35b771a
plot_phi(phi, 0.0, params.d, -1.0, 2*params.l; ns=30)

# ╔═╡ 40161863-9103-4a23-ab5a-271a4509356a
function test_E_field_mod(params)
   
    # At ρ = 0, the radial derivative of the perturbation is zero.
    E = E_field_mod(0.0, params.l1, params)
    @test isapprox(E[1], 0.0, atol=1e-8)
end

# ╔═╡ f7ddf4ec-b915-4ffe-9564-8a8dcdd0c294
test_E_field_mod(params)

# ╔═╡ 370f0b3c-719e-4249-bb06-095bd79fb3b4
function test_electron_trajectory_mod(params; dt=1e-2, max_steps=10000)
   
    traj = electron_trajectory_mod((0.0, params.l3), params, dt=dt, max_steps=max_steps)
    # The trajectory should eventually reach z ≤ 0 (anode/ground region)
    @test traj[end][2] <= 0
end

# ╔═╡ c7de7228-87c2-42a3-b153-039cc1be6678
test_electron_trajectory_mod(params; dt=1e-4, max_steps=10000)

# ╔═╡ 848c5c36-43ad-43d1-a9ec-26c6d825fb53
md"""# CGPT prompts"""

# ╔═╡ aa9b3f30-9a14-49c0-ac14-4aa091d9e2a9
md"""
### 

## Field lines in GALA.

Context: you are a physicist with strong programming skills in Julia language.

Consider the following structure (which we call GALA). 
1. two infinte planes of a conductor (for example, copper). They are very thin (20 microns). Called them "copper plates". The upper copper plate (Gate) is at a voltage V, and the lower copper plate (Anode) is at ground.
2. The Gate and the Anode are separated by a distance l (5 mm)
3. A dielectric (for example acrylic) is sandwiched between the two copper plates.
4. A hole of diameter d is made with coordinates (0,0,0). The walls of the hole are, therefore, made of acrylic. 

We want to model the electric field in the vicinity of the hole. Assuming that the hole entrance is in (0,0,0) and the hole exit is in (0,0,l), the cylinder where we want to find the electric field has the upper encap in (0,0, -l/2) and the lower endcap in (0,0, (3/2)*l), and a diameter D = 2*d. We call this cylinder, Fiducial Volume (FV)

Consider now electrons entering the FV. The electrons will propagate following the electric field lines. We want to produce a mathematical description of those field lines, so that we can follow the trajectory of the electrons from the point of entrance in the FV, until they reach the Anode.

To produce those field lines (e.g, solving the electric field) you can use any sound approximation, and/or numerical approaches. 

The algorithm must be described and you must output it in latex format. Then code it in julia. 

Divide the algorithm in small functions and provide test functions for each one of the main functions you need. 

Include plot functions to plot the field lines and the electron trajectories in the FV. 
"""

# ╔═╡ 136dee09-57e2-4913-a478-8a6f7087376d
md"""
Refer to the GALA problem. We need to add extra boundary conditions.The GALA structure is surrounded by an isolant (gas xenon for example). We have defined the Gate of the GALA to be at a potencial V and the anode to be at ground. 

We now modify the boundary conditions as follows.

At z = 3l, we have an infinite conductor at potencial V0.
At z = 2l, we have the GALA gate, at potential Vg.
At z = l, we have the anode, at potential Va
At z = 0, we have another infinite conductor at potential V=0 (ground)

As an extra condition, V0 = Vg + eps. Example: Vg = 10^4 V, V0 = Vg + 400 V
Va = -eps2; Example, Va = -100 V.

We want to compute field lines and electron trajectories in the region defined by  [0, d] (where d is the diameter of the hole), and [0,3l]
"""

# ╔═╡ 7f536e77-4e8c-43da-ae57-9536555b7f51
md"""
Refer to the GALA problem. We need to modify the boundary conditions as follows.

At z = 0, we have an infinite conductor at potential V=0 (ground)
At z = l1, we have the anode, at potential Va
At z = l2, we have the GALA gate, at potential Vg.
At z = l3, we have an infinite conductor at potencial V0.

As an extra condition, V0 = -Vg - eps. Example: Vg = -10^4 V, V0 = Vg - 400 V
Va = -eps2; Example, Va = -100 V.

We want to compute field lines and electron trajectories in the region defined by  [0, d] (where d is the diameter of the hole), and [0,l3]
"""

# ╔═╡ df12e950-f394-4947-9c4c-5e297605ba0b


# ╔═╡ f70bc7d7-5f64-4d62-ba0c-eb3c1b446688
md"""
### Test functions
"""

# ╔═╡ 3613a7b5-9071-4ff3-bc30-06fd1d42bf31
md"""
Context: you are a physicist with strong programming skills in Julia language.

Consider the following structure (which we call GALA).

At z = 0, we have an infinite conductor at potential V=0 (ground)
At z = l1, we have the anode, held at potential Va
At z = l2, we have the gate, at potential Vg.
At z = l3, we have an infinite conductor at potencial V0.

Example values: V =0, Va = -100 V, Vg = -10000 V, V0 = -10400 V.

The GALA structure: 

1. Inserts a dielectric (e.g. Acrilic), between the gate and the anode planes
2. opens up a hole of diameter d that connects the gate with the anode.


We want to model:

1. The potential Phi(r,z), where r is the radial coordinate and z the longitudinal coordinate.

2. Write the corresponding Julia code, including:

a) tests to verify that Phi behaves as expected and respect all relevant boundary conditions.

b) plot functions to visualize Phi (including 3D and contour plots)

2. The electric field.

3. The trajectories followed by electrons that start in the vicinity of the  


We want to compute field lines and electron trajectories in the region defined by  [0, d] (where d is the diameter of the hole), and [0,l3]

1. two infinte planes of a conductor (for example, copper). They are very thin (20 microns). Called them "copper plates". The upper copper plate (Gate) is at a voltage V, and the lower copper plate (Anode) is at ground.
2. The Gate and the Anode are separated by a distance l (5 mm)
3. A dielectric (for example acrylic) is sandwiched between the two copper plates.
4. A hole of diameter d is made with coordinates (0,0,0). The walls of the hole are, therefore, made of acrylic. 

We want to model the electric field in the vicinity of the hole. Assuming that the hole entrance is in (0,0,0) and the hole exit is in (0,0,l), the cylinder where we want to find the electric field has the upper encap in (0,0, -l/2) and the lower endcap in (0,0, (3/2)*l), and a diameter D = 2*d. We call this cylinder, Fiducial Volume (FV)

Consider now electrons entering the FV. The electrons will propagate following the electric field lines. We want to produce a mathematical description of those field lines, so that we can follow the trajectory of the electrons from the point of entrance in the FV, until they reach the Anode.

To produce those field lines (e.g, solving the electric field) you can use any sound approximation, and/or numerical approaches. 

The algorithm must be described and you must output it in latex format. Then code it in julia. 

Divide the algorithm in small functions and provide test functions for each one of the main functions you need. 

Include plot functions to plot the field lines and the electron trajectories in the FV. 
"""

# ╔═╡ f0fba670-3376-491b-be1f-7d1e0b35566a
begin
	get_potential(pars) = (r, z) -> potential_mod(r, z, pars)
	phi = get_potential(params)
	test_potential_mod(params)
end

# ╔═╡ d552f63d-4772-432e-aa68-8f2e680da1ab
function shape_z_sin(z::Float64, l1::Float64, l3::Float64)
    if l3 ≈ l1
        return 0.0
    end
    return sin(pi * (z - l1)/(l3 - l1))
end

# ╔═╡ 4f045c17-acd8-4242-9f56-08a3b2b09688
begin
	# =============== 1) Radial Factor ===============
"""
    radial_factor(r, w)

Compute a Gaussian radial factor, e.g. exp(-r^2 / w^2).
"""
function radial_factor(r::Float64, w::Float64)
    return exp(- (r^2)/(w^2))
end

# =============== 2) Sine-based Shape Function ===============
"""
    shape_z_sin(z, l1, l3)

Return a sine-based shape function that vanishes at z = l1 and z = l3:

    shape_z_sin(z) = sin(π * (z - l1)/(l3 - l1))

This ensures shape_z_sin(l1) = 0 and shape_z_sin(l3) = 0.
"""
function shape_z_sin(z::Float64, l1::Float64, l3::Float64)
    if abs(l3 - l1) < 1e-14
        return 0.0
    end
    return sin(pi * (z - l1)/(l3 - l1))
end

"""
    shape_z_sin_prime(z, l1, l3)

Compute the derivative d/dz of shape_z_sin(z,l1,l3).

    d/dz [ sin( π*(z-l1)/(l3-l1) ) ]
  = cos( π*(z-l1)/(l3-l1) ) * [π/(l3-l1)]
"""
function shape_z_sin_prime(z::Float64, l1::Float64, l3::Float64)
    if abs(l3 - l1) < 1e-14
        return 0.0
    end
    return (pi/(l3 - l1)) * cos(pi*(z - l1)/(l3 - l1))
end

# =============== 3) Full Potential φ(r,z) ===============
"""
    phi(r, z, params)

Compute the full potential = phi_bg(z) + A * radial_factor(r,w) * shape_z_sin(z, l1, l3).

Requires:
  phi_bg(z, params) -> background potential
  params.A, params.w, params.l1, params.l3
"""
function phi(r::Float64, z::Float64, params)
    ϕ_bg = phi_bg(z, params)
    rf   = radial_factor(r, params.w)
    shz  = shape_z_sin(z, params.l1, params.l3)
    return ϕ_bg + params.A * rf * shz
end

# =============== 4) Electric Field E(r,z) = -∇φ(r,z) ===============
"""
    E_rz(r, z, params) -> (E_r, E_z)

Compute the electric field in (r,z), using:

    E_r = -∂φ/∂r
    E_z = -∂φ/∂z

where:
  φ(r,z) = φ_bg(z) + A * exp(-r^2/w^2) * shape_z_sin(z).
  
Assumes:
- φ_bg depends on z only (so ∂/∂r φ_bg=0).
- radial_factor(r) = exp(-r^2/w^2).
- shape_z_sin(z) as above.
"""
function E_rz(r::Float64, z::Float64, params) 
    # 1) Background derivative w.r.t. r is zero, so
    #    E_r = -∂/∂r [ A*rf(r)*shape_z_sin(z) ].
    # radial_factor(r) = exp(-r^2/w^2),
    # => d/dr radial_factor(r) = -2r/w^2 * exp(-r^2/w^2).
    #
    # => E_r = - [ A * shape_z_sin(z) * d/dr radial_factor(r) ] 
    #
    w  = params.w
    A  = params.A
    
    # radial factor
    rf  = radial_factor(r, w)
    # derivative w.r.t. r
    drf_dr = - (2*r)/(w^2) * rf
    
    shz = shape_z_sin(z, params.l1, params.l3)
    
    # E_r = - A * shz * drf_dr
    E_r = - A * shz * drf_dr
    
    # 2) E_z = - d/dz [ φ_bg(z) + A*rf(r)*shape_z_sin(z) ]
    # => E_z = - d/dz φ_bg(z) - A*rf(r)* d/dz shape_z_sin(z).
    # (since rf(r) depends only on r, so no ∂/∂z of that factor.)
    
    dphi_bg_dz_val = dphi_bg_dz(z, params)  # e.g. 3a z^2 + ...
    shz_prime      = shape_z_sin_prime(z, params.l1, params.l3)
    
    E_z = - ( dphi_bg_dz_val + A * rf * shz_prime )
    
    return (E_r, E_z)
end

"""
    dphi_bg_dz(z, params)

Compute ∂/∂z of the background potential (whatever it is).
This depends on your phi_bg definition. Example for a cubic polynomial:
"""
function dphi_bg_dz(z::Float64, params)
    # Suppose phi_bg(z) = a*z^3 + b*z^2 + c*z
    # We'll do the same approach used in your code:
    Va, Vg, V0 = params.Va, params.Vg, params.V0
    l1, l2, l3 = params.l1, params.l2, params.l3
    # Solve for a,b,c from boundary conditions
    M = [l1^3 l1^2 l1;
         l2^3 l2^2 l2;
         l3^3 l3^2 l3]
    bcs = [Va, Vg, V0]
    coeffs = M \ bcs
    a, b, c = coeffs
    return 3a*z^2 + 2b*z + c
end
end

# ╔═╡ Cell order:
# ╠═f4813204-e644-11ef-0ce3-8bb694f90094
# ╠═2a1ae100-8989-46bf-82a3-dfda83b19ac7
# ╠═0d5a3deb-d370-4fb6-aaa6-e513c87e6a3b
# ╠═3d9e9dff-2ff9-4165-9387-69e79a31e6e9
# ╠═fe9bcc95-993a-45b8-a7a9-89433195b6a0
# ╠═460ac14b-a4e0-402c-982b-7e16e50bc655
# ╟─b42c1e00-31ee-4bac-8988-139c1bffa1bf
# ╟─f08d6c33-20c0-4d97-bdd1-68ddb5c4e104
# ╠═d5792afc-7e93-47f1-a8f5-98f007da6140
# ╠═1040845e-c934-4b2a-8be5-e2180ddfc3f9
# ╠═b24367b3-1ab5-4d28-bfb5-0c956fc3a6da
# ╠═f7ddf4ec-b915-4ffe-9564-8a8dcdd0c294
# ╠═f28cab00-4ec8-4f6f-b23c-f224d400ae26
# ╠═845ca241-c861-4505-8ad2-b25052f5d8bf
# ╠═edb12f79-36ee-474e-8a87-3766ba966c3a
# ╠═e71eb72d-9051-45bd-89c0-afd3bde94d0e
# ╠═d4e54bf8-dbd7-4e4f-83ba-2e5f517d9d47
# ╠═b84ca364-a8c7-42ca-b735-f16a35fb550d
# ╠═d8f89f9e-aaf2-401b-8205-0848789c265f
# ╠═30220ad6-6710-450d-99e0-753b3b256585
# ╠═4f045c17-acd8-4242-9f56-08a3b2b09688
# ╠═d552f63d-4772-432e-aa68-8f2e680da1ab
# ╠═23f35b47-d898-463d-a141-16814d88bd4c
# ╠═f06c35aa-1970-4077-9e6f-b224e9f0386e
# ╠═dd46c25c-75ad-4e0c-88fe-c70228d1fb18
# ╠═31934950-dd0a-4758-82d8-700a3e9f7605
# ╠═bc54303b-3ac6-43a9-a241-11e967ab2d40
# ╠═7972c25a-db03-4aca-905c-88fc183cae37
# ╠═ca19c8a0-426a-4f4b-9ca6-9230f453f572
# ╠═27a8e777-37d1-44a5-8747-9579fdbcc18b
# ╠═4354c71c-b20b-4da1-b4a5-ca8d05906ade
# ╠═72ccfbb1-e847-46c5-b442-48affae0b973
# ╠═c7de7228-87c2-42a3-b153-039cc1be6678
# ╠═f0fba670-3376-491b-be1f-7d1e0b35566a
# ╠═8215e89c-f73b-449c-a3df-9f9ed35b771a
# ╠═efc08fe1-b64a-455b-947b-36c3b3f4da94
# ╠═42138717-98f3-432f-9481-0c1d0281a4b5
# ╠═142c0965-5c05-49c4-bc2c-d84f8b2e84b7
# ╠═3163342f-f107-4587-9f84-b617545082ce
# ╠═66459668-4944-4934-8015-a859867110cd
# ╠═72e7015c-3337-440e-bf1c-8ac54a76c8c4
# ╠═6e4890c2-9203-46fa-b40f-092b5d7c31af
# ╠═9355cac7-eff5-4c2d-bae8-58fc1c598019
# ╠═3eff442d-a523-44b1-883e-76e03796ad2b
# ╠═451e6ffb-f05a-408a-be17-12ad8e1605e3
# ╠═5d035c7b-c382-4c23-a163-ce107e195447
# ╠═ee2d4035-c4b0-4dc1-b690-a3876b7eba4a
# ╠═dd9048a5-e730-4809-8d79-eea8ae9239ce
# ╠═12383f9f-3cbb-4f2d-98d3-a00346f7f1cf
# ╠═6438baef-aeca-4cc5-8ccc-665e1eca427a
# ╠═1aedb785-5198-4d32-858b-d9427293fc7e
# ╠═8cdb2ae6-8952-4d61-9252-f63c2ffcbb16
# ╠═13133baf-d118-4cb6-a12f-0724ea168f97
# ╠═943bcd94-2716-41cd-9d9a-b54711a2e749
# ╠═de40bca9-535b-4f7d-ae52-6029e4118be9
# ╠═f5b6f90f-df29-48e6-ac83-4ae57e88ed19
# ╠═2d6e215f-9882-41f3-ba57-c1a4ceba3eae
# ╠═8c18b522-b741-4109-a9e1-d33e5fc85796
# ╠═4130b7f9-52bb-411f-9915-cbb50c7e1380
# ╠═3fd1735f-de52-4d5f-afb2-22f11c629bfe
# ╠═f3036b4f-463d-4201-8023-e1eba4c10307
# ╠═f81b3873-c702-4ff9-9e76-f075286b47b6
# ╠═40161863-9103-4a23-ab5a-271a4509356a
# ╠═370f0b3c-719e-4249-bb06-095bd79fb3b4
# ╠═848c5c36-43ad-43d1-a9ec-26c6d825fb53
# ╠═aa9b3f30-9a14-49c0-ac14-4aa091d9e2a9
# ╠═136dee09-57e2-4913-a478-8a6f7087376d
# ╠═7f536e77-4e8c-43da-ae57-9536555b7f51
# ╠═df12e950-f394-4947-9c4c-5e297605ba0b
# ╠═f70bc7d7-5f64-4d62-ba0c-eb3c1b446688
# ╠═3613a7b5-9071-4ff3-bc30-06fd1d42bf31
