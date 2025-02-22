"""function to find the grid index closest to a given z value"""
    function find_z_index(zval, zgrid)
        diffs = abs.(zgrid .- zval)
        return argmin(diffs)
    end

    """
    solve_laplace_gala(params; Nr, Nz)

Solve Laplace’s equation in the domain:
   r ∈ [0, d],  z ∈ [0, l3]
using a finite difference scheme.
 - Nr: Number of grid points in r
 - Nz: Number of grid points in z

Boundary conditions:
   φ(r, 0)   = 0,

   φ(r, l1)  = Va,

   φ(r, l2)  = Vg,

   φ(r, l3)  = V0,

and at r=0 and r=d, use Neumann (∂φ/∂r=0).

Returns (phi\\_mat, rgrid, zgrid) where phi\\_mat is an Nr×Nz matrix.
"""
function solve_laplace_gala(params::GALAParams; Nr=51, Nz=101)
	
	
    rmin, rmax = 0.0, params.d
    zmin, zmax = 0.0, params.l3
    dr = (rmax - rmin)/(Nr - 1)
    dz = (zmax - zmin)/(Nz - 1)
    rgrid = collect(range(rmin, rmax, length=Nr))
    zgrid = collect(range(zmin, zmax, length=Nz))
    
    Ntot = Nr * Nz
    A_mat = spzeros(Float64, Ntot, Ntot)
    b_vec = zeros(Float64, Ntot)
    
    # mapping: index(i,j) = (j-1)*Nr + i, with i=1..Nr, j=1..Nz.
    index(i,j) = (j-1)*Nr + i
    
    
    i_z0 = find_z_index(0.0, zgrid)
    i_zl1 = find_z_index(params.l1, zgrid)
    i_zl2 = find_z_index(params.l2, zgrid)
    i_zl3 = find_z_index(params.l3, zgrid)
    
    # Use i_z0, i_zl1, i_zl2, i_zl3
    # to impose Dirichlet boundary conditions.)
    
    for j in 1:Nz
        for i in 1:Nr
            idx = index(i,j)
            r_val = rgrid[i]
            z_val = zgrid[j]
            
            if j == i_z0
                A_mat[idx, idx] = 1.0
                b_vec[idx] = 0.0
            elseif j == i_zl1
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Va
            elseif j == i_zl2
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.Vg
            elseif j == i_zl3
                A_mat[idx, idx] = 1.0
                b_vec[idx] = params.V0
            else
                # Interior points: discretize Laplace's equation.
                coeff_z = 1/dz^2
                if i == 1
                    # r = 0, use symmetry: φ_{1,j} = φ_{2,j}
                    A_mat[idx, index(1,j)] = 1.0
                    A_mat[idx, index(2,j)] = -1.0
                elseif i == Nr
                    # r = rmax: Neumann: φ_{Nr,j} = φ_{Nr-1,j}
                    A_mat[idx, index(Nr,j)] = 1.0
                    A_mat[idx, index(Nr-1,j)] = -1.0
                else
                    coeff_r = 1/(dr^2)
                    coeff_r_prime = 1/(2*dr)
                    A_mat[idx, index(i+1,j)] += r_val*coeff_r + coeff_r_prime
                    A_mat[idx, index(i-1,j)] += r_val*coeff_r - coeff_r_prime
                    A_mat[idx, idx] += -2*r_val*coeff_r
                end
                A_mat[idx, index(i,j+1)] += coeff_z
                A_mat[idx, index(i,j-1)] += coeff_z
                A_mat[idx, idx] += -2*coeff_z
            end
        end
    end
    
    phi_vec = A_mat \ b_vec
    phi_mat = reshape(phi_vec, Nr, Nz)
    return (phi_mat, rgrid, zgrid)
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

function test_phi_boundaries(params)
    @testset "φ(r,z) Boundary Conditions" begin
        (phi_mat, rgrid, zgrid) = solve_laplace_gala(params; Nr=51, Nz=101)
        
        i0 = find_z_index(0, zgrid)
        il1 = find_z_index(params.l1, zgrid)
        il2 = find_z_index(params.l2, zgrid)
        il3 = find_z_index(params.l3, zgrid)
        
        for i in 1:length(rgrid)
            @test isapprox(phi_mat[i, i0], 0.0; atol=1e-5)
            @test isapprox(phi_mat[i, il1], params.Va; atol=1e-5)
            @test isapprox(phi_mat[i, il2], params.Vg; atol=1e-5)
            @test isapprox(phi_mat[i, il3], params.V0; atol=1e-5)
        end
    end
end

"""
    test_E_field()

Run tests on the computed electric field.
In our geometry (axial symmetry) we expect that at r=0 the radial field vanishes.
"""
function test_E_field(params)
    
    # Here we call the Laplace solver to get phi(r,z) on the grid.
    # (Replace solve_laplace_gala with your actual solver function if necessary.)
    (phi_mat, rgrid, zgrid) = solve_laplace_gala(params; Nr=51, Nz=101)
    E_r, E_z = compute_E_field(phi_mat, rgrid, zgrid)
    
    # Test: For all z, at r=0 the radial derivative should be near zero.
    for j in 1:length(zgrid)
        @test isapprox(E_r[1,j], 0.0; atol=1e-5)
    end
    println("Electric field tests passed: E_r(r=0, z) ≈ 0 for all z.")
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