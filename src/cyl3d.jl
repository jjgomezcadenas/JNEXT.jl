import PyPlot

struct Cylinder
    r::Float64      # radius
    x0::Float64     # cylinder center
    y0::Float64     # cylinder center
    zmin::Float64   # lower z-boundary
    zmax::Float64   # upper z-boundary
    p0::Vector{Float64}  # point on one end (computed)
    p1::Vector{Float64}  # point on the other end (computed)
    
end


function meshgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(reshape(x, 1, length(x)), length(y), 1)
    Y = repeat(reshape(y, length(y), 1), 1, length(x))
    return X, Y
end

"""Outer constructor computes additional fields."""
function Cylinder(r::Float64, x0::Float64, y0::Float64, 
                    zmin::Float64, zmax::Float64)
    p0 = [x0, y0, zmin]   # point at one end
    p1 = [x0, y0, zmax]   # point at the other end
    cyl = Cylinder(r, x0, y0, zmin, zmax, p0, p1)
    return cyl
end


clength(c::Cylinder) = c.zmax - c.zmin
perimeter(c::Cylinder) = 2 * π * c.r
area_barrel(c::Cylinder) = 2 * π * c.r * length(c)
area_endcap(c::Cylinder) = π * c.r^2
area(c::Cylinder) = area_barrel(c) + 2 * area_endcap(c)
volume(c::Cylinder) = π * c.r^2 * length(c)


function surfaces(c::Cylinder)

	mag, v, n1, n2 = unit_vectors(c)
	
	# Create parameter ranges:
	t = collect(range(0, stop=mag, length=2))   # for the axis (2 sample points)
	theta = collect(range(0, stop=2*pi, length=100))    # angular parameter
	rsample = collect(range(0, stop=c.r, length=2))     # for endcaps

	# Create meshgrid arrays.
	T, Theta2 = meshgrid(t, theta)        # for the barrel
	R, Theta  = meshgrid(rsample, theta)   # for the endcaps


	# Barrel ("tube"): generate coordinates over the lateral surface.
	    X = c.p0[1] .+ v[1]*T .+ c.r .* sin.(Theta2) .* n1[1] .+ c.r .* cos.(Theta2) .* n2[1]
	    Y = c.p0[2] .+ v[2]*T .+ c.r .* sin.(Theta2) .* n1[2] .+ c.r .* cos.(Theta2) .* n2[2]
	    Z = c.p0[3] .+ v[3]*T .+ c.r .* sin.(Theta2) .* n1[3] .+ c.r .* cos.(Theta2) .* n2[3]
	    
	    # Bottom endcap (at zmin)
	    X2 = c.p0[1] .+ R .* sin.(Theta) .* n1[1] .+ R .* cos.(Theta) .* n2[1]
	    Y2 = c.p0[2] .+ R .* sin.(Theta) .* n1[2] .+ R .* cos.(Theta) .* n2[2]
	    Z2 = c.p0[3] .+ R .* sin.(Theta) .* n1[3] .+ R .* cos.(Theta) .* n2[3]
	    
	    # Top endcap (at zmax)
	    X3 = c.p0[1] .+ v[1]*mag .+ R .* sin.(Theta) .* n1[1] .+ R .* cos.(Theta) .* n2[1]
	    Y3 = c.p0[2] .+ v[2]*mag .+ R .* sin.(Theta) .* n1[2] .+ R .* cos.(Theta) .* n2[2]
	    Z3 = c.p0[3] .+ v[3]*mag .+ R .* sin.(Theta) .* n1[3] .+ R .* cos.(Theta) .* n2[3]
	    
	    return (X, Y, Z), (X2, Y2, Z2), (X3, Y3, Z3)
end


function unit_vectors(c::Cylinder)
    v = c.p1 .- c.p0
    mag = norm(v)
    v = v / mag
    not_v = [1.0, 0.0, 0.0]
    # Check if v is approximately (1,0,0)
    if all(isapprox.(v, not_v))
        not_v = [0.0, 1.0, 0.0]
    end
    n1 = cross(v, not_v)
    n1 /= norm(n1)
    n2 = cross(v, n1)
    return mag, v, n1, n2
end


"""Normal to the cylinder barrel"""
function normal_to_barrel(c::Cylinder, P::Vector{Float64})    
    [P[1], P[2], 0] ./ c.r
end


"""
Uses equation of cylynder: 

F(x,y,z) = x^2 + y^2 - r^2 = 0
"""
function cylinder_equation(c::Cylinder, P::Vector{Float64})
    P[1]^2 + P[2]^2 - c.r^2
end


function draw_cylinder(c; alpha=0.2, figsize=(16,16), 
                        DWORLD=false, 
                        WDIM=((0.0,4.5), (0.0,4.5), (-10.0,0)),
                        barrelColor="blue", cupColor="red")
	
	fig = PyPlot.figure(figsize=figsize)
    ax = PyPlot.subplot(111, projection="3d")

	P, P2, P3 = surfaces(c)

	if DWORLD
        ax.set_xlim3d(0.0, 10.0)
        ax.set_ylim3d(0.0, 10.0)
        ax.set_zlim3d(-10.0, 0.0)
    end

	ax.plot_surface(P[1], P[2], P[3], color=barrelColor, alpha=alpha)
	ax.plot_surface(P2[1], P2[2], P2[3], color=cupColor, alpha=alpha)
    ax.plot_surface(P3[1], P3[2], P3[3], color=cupColor, alpha=alpha)

	
	ax, fig
end


function draw_cylinder2(c, c2; alpha=0.3, figsize=(16,16), 
    DWORLD=false, 
    WDIM=((0.0,4.5), (0.0,4.5), (-10.0,0)),
    barrelColor="blue", cupColor="red")

    fig = PyPlot.figure(figsize=figsize)
    ax = PyPlot.subplot(111, projection="3d")

    P, P2, P3 = surfaces(c2)

    if DWORLD
    ax.set_xlim3d(0.0, 10.0)
    ax.set_ylim3d(0.0, 10.0)
    ax.set_zlim3d(-10.0, 0.0)
    end

    ax.plot_surface(P[1], P[2], P[3], color=barrelColor, alpha=alpha)
    ax.plot_surface(P2[1], P2[2], P2[3], color=cupColor, alpha=alpha)
    ax.plot_surface(P3[1], P3[2], P3[3], color=cupColor, alpha=alpha)

    PP, _, _ = surfaces(c)
    ax.plot_surface(PP[1], PP[2], PP[3], color="orange", alpha=0.1)

    ax, fig
end


function draw_cylinder_proj(jsteps, gammas, r, x0, y0, Za, Zg, Zs, pitch; 
    alpha=0.3, figsize=(8,8))


    c = Cylinder(r, x0, y0, Za, Zg)
    P, _, _ = surfaces(c)

    X = vec(P[1])
    Y = vec(P[1])
    Z = vec(P[3])

    # Extract x, y, z coordinates for each step in the jsteps.
    xs = [step[1] for step in jsteps]
    ys = [step[2] for step in jsteps]
    zs = [step[3] for step in jsteps]



    # Choose a color for this gamma (cycling if needed).
    #col = colors[mod1(i, length(colors))]



    fig, axs = PyPlot.subplots(1, 2, figsize=figsize)
    # Draw a vertical line at x = 5
    axs[1].axhline(y=Zg, color="black", linestyle="--", linewidth=2, label=false)
    axs[1].axhline(y=Za, color="black", linestyle="--", linewidth=2, label=false)
    axs[1].axhline(y=Zs, color="black", linestyle="solid", linewidth=2, label=false)

    axs[1].axvline(x=x0-r, color="black", linestyle="--", linewidth=2, label=false)
    axs[1].axvline(x=x0+r, color="black", linestyle="--", linewidth=2, label=false)

    # X-Z projection: plot X vs Z.
    #axs[1].scatter(X, Z, s=1, color="blue")
    axs[1].plot(xs, zs, linestyle=:solid, color=:red, linewidth=3)

    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Z (mm)")
    axs[1].set_title("X–Z Projection")
    axs[1].set_xlim(0, pitch)
    axs[1].set_ylim(Zs, Zg)
    for gamma in gammas
    # Extract x, y, z coordinates for each step in the gamma.
    xgs = [step[1] for step in gamma]
    zgs = [step[3] for step in gamma]

    axs[1].scatter(xgs, zgs, s=1, color="cyan")
    axs[1].plot(xgs, zgs, linestyle="--", color=:cyan, linewidth=1)
    end

    # Y-Z projection: plot Y vs Z.
    axs[2].scatter(Y, Z, s=1, color="blue")
    axs[2].plot(ys, zs, linestyle=:solid, color=:cyan, linewidth=1)
    axs[2].set_xlabel("Y (mm)")
    axs[2].set_ylabel("Z (mm)")
    axs[2].set_title("Y–Z Projection")
    axs[2].set_xlim(0, pitch)
    axs[2].set_ylim(Zs, Zg)
    PyPlot.tight_layout()
    fig
end

function plot_cylinder(gammas, jsteps, x0, y0, zg, zb, za, r; 
    num_plot=5, figsize=(16,16))

    c = Cylinder(r, x0, y0, za, zg)
    c2 = Cylinder(r, x0, y0, zb, zg)
    ax, fig = draw_cylinder2(c, c2; alpha=0.2, figsize=figsize, DWORLD=false)

    # Define a set of colors to cycle through for different gammas.
    colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow]

    # Extract x, y, z coordinates for each step in the jsteps.
    xs = [step[1] for step in jsteps]
    ys = [step[2] for step in jsteps]
    zs = [step[3] for step in jsteps]
    ax.plot(xs, ys, zs, linestyle="--", color=:cyan, linewidth=1)

    # Loop over the selected gammas.
    #for i in 1:min(num_plot, length(gammas))
    i = num_plot
    gamma = gammas[i]
    # Extract x, y, z coordinates for each step in the gamma.
    xs = [step[1] for step in gamma]
    ys = [step[2] for step in gamma]
    zs = [step[3] for step in gamma]

    # Choose a color for this gamma (cycling if needed).
    #col = colors[mod1(i, length(colors))]

    ax.scatter(xs, ys, zs, s=25, color=:red)

    #println("xs =",xs)
    #println("ys =",ys)
    #println("zs =",zs)

    # Connect the points with a dashed line.
    ax.plot(xs, ys, zs, linestyle="--", color=:red, linewidth=1)
    #end

    fig
end