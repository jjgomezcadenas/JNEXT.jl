from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class GalaParams:
    x1: float     # x-position of Hole 1 center
    d1: float      # hole diameter [m]
    p:  float       # pitxh
    l0: float      # z-position of ground (potential 0) [m]
    l1: float      # z-position of anode (potential Va) [m]
    l2: float      # z-position of gate (potential Vg) [m]
    l3: float      # z-position of top electrode (potential V0) [m]
    Va: float      # anode potential [V]
    Vg: float      # gate potential [V]
    V0: float      # top electrode potential [V]

@dataclass
class X1Params:
    x1: float     # x-position of Hole 1 center
    d1: float     # hole diameter [m]
    p:  float     # xmin = x1 - p  
    zgrnd: float  # z of ground.
    zanode: float # z anode
    zgate: float  # z gate
    zdrft: float  # z gate
    
    dzel: float   # dz of the EL region
    dzdr: float   # dz of the drift region

    ndr:int       # number of virtual electrodes the drift.
    nel:int       # number of virtual electrodes in EL

    vg: np.array  # voltage of virtual planes in EL
    vd: np.array  # voltage of virtual planes in drift

    zdr: np.array # z position  of virtual planes in drift
    zel: np.array # z position  of virtual planes in EL

    Va: float      # anode potential 
    Vg: float      # gate potential 
    Vd: float      # top drift electrode potential [V]


# -------------------------
# Assemble the sparse matrix A and right-hand side b for all nodes.
# We order the nodes with a single index: k = j*Nx + i.
# For Dirichlet nodes, we set the equation: phi = fixed_value.
# For interior (unknown) nodes, we apply finite differences.
# -------------------------
def phixz1h(params: X1Params, Nx: int, Nz: int):

    def find_index(x, ix):
        """Find the indices where x equals ix"""
        indices = np.where(x == ix)[0]

        if indices.size == 0:
            return False, 0
            
        return True, indices[0]
    
    def add_entry(k, kk, coeff):
        rows.append(k)
        cols.append(kk)
        data.append(coeff)

    def is_dirichlet(j, i):
        """
        Determine if the grid point (j, i) is a Dirichlet point.
        Returns a tuple (flag, value) where flag is True if the point is fixed.
        
        Dirichlet conditions are:
        - z = zgrnd: φ = 0   for all z
        - z = zanode: φ = Va for x in hole
        - z = zgate: φ = Vg  for x in hole
        - z = zdrft: φ = Vd  for all x
        - z = zd: φ = vd (for each virtual plane) all x
        - z = zel: φ = vg (for each virtual plane) x in hole
        
    """
        
        if j == j_zgrnd:
            return True, 0
        if j == j_zanode and (in_hole(x[i], x1_center)):
             return True, params.Va
        if j == j_zgate and (in_hole(x[i], x1_center)):
            return True, params.Vg
        if j == j_zdrft:
            return True, params.Vd

        cond, indx = find_index(j_ze, j)
        if cond and (in_hole(x[i], x1_center)):
            return True, vg[indx]

        cond, indx = find_index(j_zd, j)
        if cond:
            return True, vd[indx]

        return False, None

    def in_hole(x_val, hole_center):
        """Return True if x_val is within the hole (centered at hole_center with diameter d1)."""
        return (x_val >= (hole_center - d1/2)) and (x_val <= (hole_center + d1/2))

    
    N = Nx * Nz
    x1 = params.x1 
    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
    x_min = x1-p
    x_max = x1+p
    
    x = np.linspace(x_min, x_max, Nx)
    z = np.linspace(params.zgrnd, params.zdrft, Nz)
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    # Find grid indices for z = l1 and l2 (assume they fall on grid)
    j_zd = np.array([np.argmin(np.abs(z - lz) for lz in params.zdr)])
    j_ze = np.array([np.argmin(np.abs(z - lz) for lz in params.zel)])
    j_zgrnd = np.argmin(np.abs(z - params.zgrnd))
    j_zanode = np.argmin(np.abs(z - params.zanode))
    j_zgate = np.argmin(np.abs(z - params.zgate))
    j_zdrft = np.argmin(np.abs(z - params.zdrft))
    
    print(f"""
    Nx = {Nx}, Nz = {Nz}
    x1 = {x1_center}, p = {p}, d = {d1}
    x_min = {x_min}, x_max = {x_max}
    zdr = {params.zdr}
    j_zd = {j_zd}
    zel = {params.zel}
    j_ze = {j_ze}
    j_zgrnd = {j_zgrnd}
    j_zanode = {j_zanode}
    j_zgate = {j_zgate}
    j_zdrft = {j_zdrft}

    print params = {params}
    """)

    
    data = []    # List to store nonzero matrix entries
    rows = []    # Row indices of nonzero entries
    cols = []    # Column indices of nonzero entries
    b = np.zeros(N)  # Right-hand side vector

    # Loop over all grid points (j for z-index, i for x-index)
    for j in range(Nz):
        for i in range(Nx):
            k = j * Nx + i # Flatten 2D index (j, i) into 1D index k

            # Check if current node has a Dirichlet condition (fixed potential)
            fixed, val = is_dirichlet(j, i)
            if fixed:
                # Dirichlet node: phi = val
                add_entry(k, k, 1.0)
                b[k] = val
                continue # Skip further processing for fixed nodes

            # For interior or Neumann boundary nodes, apply the finite difference stencil.
            # The Laplacian: (d^2φ/dx^2 + d^2φ/dz^2)= 0
            # We use a 5-point stencil:
            #   (phi(i+1,j) - 2*phi(i,j) + phi(i-1,j))/dx^2 +
            #   (phi(i,j+1) - 2*phi(i,j) + phi(i,j-1))/dz^2 = 0
    
             # --- x-direction finite differences ---
            if i == 0:
                 # Left boundary: Neumann condition (zero derivative)
                 # Use a one-sided difference where φ(i-1,j) is replaced by φ(i+1,j)
                coeff_center_x = -2.0/dx**2
                coeff_right = 2.0/dx**2
                # center
                add_entry(k, k, coeff_center_x)
                # right neighbor (i+1)
                k_right = j * Nx + (i+1)
            
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r:
                    b[k] -= coeff_right * val_r
                else:
                    add_entry(k, k_right, coeff_right)
            elif i == Nx-1:
                # Right boundary: Neumann (zero derivative)
                coeff_center_x = -2.0/dx**2
                coeff_left = 2.0/dx**2
                add_entry(k, k, coeff_center_x)
                k_left = j * Nx + (i-1)
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= coeff_left * val_l
                else:
                    add_entry(k, k_left, coeff_left)
            else:
                # Interior in x
                add_entry(k, k,  -2.0/dx**2)
                # right neighbor
                k_right = j * Nx + (i+1)
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r:
                    b[k] -= (1.0/dx**2)*val_r
                else:
                    add_entry(k, k_right, 1.0/dx**2)
                # left neighbor
                k_left = j * Nx + (i-1)
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= (1.0/dx**2)*val_l
                else:
                    add_entry(k, k_left, 1.0/dx**2)
            
            # z-direction
            if j == 0 or j == Nz-1:
                # These boundaries are Dirichlet (should have been caught)
                pass
            else:
                add_entry(k, k, -2.0/dz**2)
                # Up neighbor (j+1)
                k_up = (j+1)*Nx + i
                fixed_up, val_up = is_dirichlet(j+1, i)
                if fixed_up:
                    b[k] -= (1.0/dz**2)*val_up
                else:
                    add_entry(k, k_up, 1.0/dz**2)
                # Down neighbor (j-1)
                k_down = (j-1)*Nx + i
                fixed_down, val_down = is_dirichlet(j-1, i)
                if fixed_down:
                    b[k] -= (1.0/dz**2)*val_down
                else:
                    add_entry(k, k_down, 1.0/dz**2)

    # Build the sparse matrix in CSR format.
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Solve the linear system.
    phi_flat = spsolve(A, b)
    phi = phi_flat.reshape((Nz, Nx))
    return  x, z, phi


# -------------------------
# Assemble the sparse matrix A and right-hand side b for all nodes.
# We order the nodes with a single index: k = j*Nx + i.
# For Dirichlet nodes, we set the equation: phi = fixed_value.
# For interior (unknown) nodes, we apply finite differences.
# -------------------------
def phi2d1h(params: GalaParams, Nx: int, Nz: int):
   
    def add_entry(k, kk, coeff):
        rows.append(k)
        cols.append(kk)
        data.append(coeff)

    def is_dirichlet(j, i):
        """
        Determine if the grid point (j, i) is a Dirichlet point.
        Returns a tuple (flag, value) where flag is True if the point is fixed.
        
        Dirichlet conditions are:
        - z = 0: φ = 0
        - z = l3: φ = V0
        - z = l1 (anode): φ = Va, but only for x-values inside one of the holes.
        - z = l2 (gate): φ = Vg, but only for x-values inside one of the holes.
    """
        # Bottom electrode: z=0
        if j == j_l0:
            return True, 0.0
        # Top electrode: z=l3
        if j == j_l3:
            return True, V0
        # Anode: z = l1, inside either hole => Va
        if j == j_l1 and in_hole(x[i], x1_center):
            return True, Va
        # Gate: z = l2, inside either hole => Vg
        if j == j_l2 and in_hole(x[i], x1_center):
            return True, Vg
        return False, None

    def in_hole(x_val, hole_center):
        """Return True if x_val is within the hole (centered at hole_center with diameter d1)."""
        return (x_val >= (hole_center - d1/2)) and (x_val <= (hole_center + d1/2))

    
    N = Nx * Nz
    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
    
    l0 = params.l0
    l3 = params.l3
    l1 = params.l1
    l2 = params.l2
    Va = params.Va
    Vg = params.Vg
    V0 = params.V0

    # x-domain: from x1 - p/2 to x2 + p/2
    x_min = x1_center - p/2
    x_max = x1_center + p/2
    
    x = np.linspace(x_min, x_max, Nx)
    z = np.linspace(l0, l3, Nz)
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    print(f"""
    Nx = {Nx}, Nz = {Nz}
    x1 = {x1_center}, p = {p}, d = {d1}
    x_min = {x_min}, x_max = {x_max}
    dx = {dx}, dz = {dz}
    """)

    # Find grid indices for z = l1 and l2 (assume they fall on grid)
    j_l1 = np.argmin(np.abs(z - l1))
    j_l2 = np.argmin(np.abs(z - l2))
    j_l3 = np.argmin(np.abs(z - l3))
    j_l0 = np.argmin(np.abs(z - l0))
    data = []    # List to store nonzero matrix entries
    rows = []    # Row indices of nonzero entries
    cols = []    # Column indices of nonzero entries
    b = np.zeros(N)  # Right-hand side vector

    # Loop over all grid points (j for z-index, i for x-index)
    for j in range(Nz):
        for i in range(Nx):
            k = j * Nx + i # Flatten 2D index (j, i) into 1D index k

            print(f"------------j = {j}, i = {i}, k = {k}-----------")

            # Check if current node has a Dirichlet condition (fixed potential)
            fixed, val = is_dirichlet(j, i)
            if fixed:
                print(f"node is dirichlet, b[{k}]= {val} ")
                add_entry(k, k, 1.0)
                b[k] = val
                continue # Skip further processing for fixed nodes

            # For interior or Neumann boundary nodes, apply the finite difference stencil.
            # The Laplacian: (d^2φ/dx^2 + d^2φ/dz^2)= 0
            # We use a 5-point stencil:
            #   (phi(i+1,j) - 2*phi(i,j) + phi(i-1,j))/dx^2 +
            #   (phi(i,j+1) - 2*phi(i,j) + phi(i,j-1))/dz^2 = 0
    
             # --- x-direction finite differences ---

            print(f"------------x direction-----------, i = {i}")
            if i == 0:
                 # Left boundary: Neumann condition (zero derivative)
                 # Use a one-sided difference where φ(i-1,j) is replaced by φ(i+1,j)

                print(f"--left boundary--, i = {i}")
                coeff_center_x = -2.0/dx**2
                coeff_right = 2.0/dx**2

                #print(f"-coeff_center_x ={coeff_center_x}  coeff_right ={coeff_right}")
                # center

                print(f"-->add entry, r = {k} c= {k}, coeff={coeff_center_x}")
                
                add_entry(k, k, coeff_center_x)
                # right neighbor (i+1)
                k_right = j * Nx + (i+1)

                print(f"--#now consider neighbor  i+1 = {i+1} with k_right = j * Nx + (i+1) = {k_right}: is dirichlet?")
            
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r:
                    b[k] -= coeff_right * val_r

                    print(f"--->yes is dirichlet: val_r = {val_r},  b[{k}] = {b[k]}")
                else:
                    add_entry(k, k_right, coeff_right)
                    print(f"<---not dirichlet: add entry, r = {k} c= {k_right}, coeff={coeff_right}")
                    
            elif i == Nx-1:
                # Right boundary: Neumann (zero derivative)

                print(f"--right boundary--, i = {i}")
                coeff_center_x = -2.0/dx**2
                coeff_left = 2.0/dx**2

                #print(f"-coeff_center_x ={coeff_center_x}  coeff_right ={coeff_right}")
                
                add_entry(k, k, coeff_center_x)

                print(f"-->add entry, r = {k} c= {k}, coeff={coeff_center_x}")
                
                k_left = j * Nx + (i-1)

                print(f"--#now consider neighbor  i-1 = {i-1} with k_left = j * Nx + (i-1) = {k_left}: is dirichlet?")
                
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= coeff_left * val_l

                    print(f"--->yes is dirichlet: val_l = {val_l},  b[{k}] = {b[k]}")
                else:
                    add_entry(k, k_left, coeff_left)
                    print(f"<---not dirichlet: add entry, r = {k} c= {k_left}, data={coeff_left}")
            else:
                # Interior in x

                print(f"--interior--, i = {i}")
                
                add_entry(k, k,  -2.0/dx**2)
                print(f"-->add entry, r = {k} c= {k}, -2.0/dx**2={-2.0/dx**2}")
                
                # right neighbor
                k_right = j * Nx + (i+1)
                fixed_r, val_r = is_dirichlet(j, i+1)

                print(f"--#now consider right neighbor  i+1 = {i+1} with k_right = j * Nx + (i+1) = {k_right}: is dirichlet?")
                
                if fixed_r:
                    b[k] -= (1.0/dx**2)*val_r

                    print(f"--->yes is dirichlet: val_r = {val_r},  b[{k}] = {b[k]}")
                    
                else:
                    add_entry(k, k_right, 1.0/dx**2)

                    print(f"<---not dirichlet: add entry, r = {k} c= {k_right}, 1.0/dx**2={1.0/dx**2}")
                # left neighbor
                k_left = j * Nx + (i-1)

                print(f"--#now consider left neighbor  i-1 = {i-1} with k_left = j * Nx + (i-1) = {k_left}: is dirichlet?")
                
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= (1.0/dx**2)*val_l

                    print(f"--->yes is dirichlet: val_l = {val_l},  b[{k}] = {b[k]}")
                else:
                    add_entry(k, k_left, 1.0/dx**2)
                    print(f"<---not dirichlet: add entry, r = {k} c= {k_left}, 1.0/dx**2={1.0/dx**2}")
            
            # z-direction

            print(f"------------z direction-----------, j = {j}, k = {k}")
            if j == 0 or j == Nz-1:
                # These boundaries are Dirichlet (should have been caught)

                print(f"j = {j} is dirichlet, already done ")
                pass
            else:
                
                add_entry(k, k, -2.0/dz**2)

                #print(f"--add entry, r = {k} c= {k},  -2.0/dz**2={ -2.0/dz**2}")
                
                # Up neighbor (j+1)
                k_up = (j+1)*Nx + i

                print(f"--#now consider up neighbor  j+1 = {j+1} with k_up = (j+1) * Nx + i = {k_up}: is dirichlet?")
                
                fixed_up, val_up = is_dirichlet(j+1, i)
                if fixed_up:
                    b[k] -= (1.0/dz**2)*val_up

                    print(f"--->yes is dirichlet: val_up = {val_up},  b[{k}] = {b[k]}")
                else:
                    add_entry(k, k_up, 1.0/dz**2)

                    print(f"<---not dirichlet: add entry, r = {k} c= {k_up},  1.0/dz**2={ 1.0/dz**2}")
                    
                # Down neighbor (j-1)
                k_down = (j-1)*Nx + i

                print(f"--#now consider down neighbor  j-1 = {j-1} with k_down = (j-1) * Nx + i = {k_down}: is dirichlet?")
                
                fixed_down, val_down = is_dirichlet(j-1, i)
                
                if fixed_down:
                    b[k] -= (1.0/dz**2)*val_down

                    print(f"--->yes is dirichlet: val_down = {val_down},  b[{k}] = {b[k]}")
                else:
                    add_entry(k, k_down, 1.0/dz**2)
                    print(f"<---not dirichlet: add entry, r = {k} c= {k_down},  1.0/dz**2={ 1.0/dz**2}")

    # Build the sparse matrix in CSR format.
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Solve the linear system.
    phi_flat = spsolve(A, b)
    phi = phi_flat.reshape((Nz, Nx))
    return b, A, x, z, phi



# -------------------------
# Assemble the sparse matrix A and right-hand side b for all nodes.
# We order the nodes with a single index: k = j*Nx + i.
# For Dirichlet nodes, we set the equation: phi = fixed_value.
# For interior (unknown) nodes, we apply finite differences.
# -------------------------
def phi2d(params: GalaParams, Nx: int, Nz: int):
   
    def add_entry(k, kk, coeff):
        rows.append(k)
        cols.append(kk)
        data.append(coeff)

    def is_dirichlet(j, i):
        """
        Determine if the grid point (j, i) is a Dirichlet point.
        Returns a tuple (flag, value) where flag is True if the point is fixed.
        
        Dirichlet conditions are:
        - z = 0: φ = 0
        - z = l3: φ = V0
        - z = l1 (anode): φ = Va, but only for x-values inside one of the holes.
        - z = l2 (gate): φ = Vg, but only for x-values inside one of the holes.
    """
        # Bottom electrode: z=0
        if j == j_l0:
            return True, 0.0
        # Top electrode: z=l3
        if j == j_l3:
            return True, V0
        # Anode: z = l1, inside either hole => Va
        if j == j_l1 and (in_hole(x[i], x1_center) or in_hole(x[i], x2_center)):
            return True, Va
        # Gate: z = l2, inside either hole => Vg
        if j == j_l2 and (in_hole(x[i], x1_center) or in_hole(x[i], x2_center)):
            return True, Vg
        return False, None

    def in_hole(x_val, hole_center):
        """Return True if x_val is within the hole (centered at hole_center with diameter d1)."""
        return (x_val > (hole_center - d1/2)) and (x_val < (hole_center + d1/2))

    
    N = Nx * Nz
    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
    x2_center = x1_center + p
    l0 = params.l0
    l3 = params.l3
    l1 = params.l1
    l2 = params.l2
    Va = params.Va
    Vg = params.Vg
    V0 = params.V0

    # x-domain: from x1 - p/2 to x2 + p/2
    x_min = x1_center - p/2
    x_max = x2_center + p/2
    
    x = np.linspace(x_min, x_max, Nx)
    z = np.linspace(l0, l3, Nz)
    dx = x[1]-x[0]
    dz = z[1]-z[0]

    print(f"""
    Nx = {Nx}, Nz = {Nz}
    x1 = {x1_center}, x2 = {x2_center}, p = {p}, d = {d1}
    x_min = {x_min}, x_max = {x_max}
    """)

    # Find grid indices for z = l1 and l2 (assume they fall on grid)
    j_l1 = np.argmin(np.abs(z - l1))
    j_l2 = np.argmin(np.abs(z - l2))
    j_l3 = np.argmin(np.abs(z - l3))
    j_l0 = np.argmin(np.abs(z - l0))
    data = []    # List to store nonzero matrix entries
    rows = []    # Row indices of nonzero entries
    cols = []    # Column indices of nonzero entries
    b = np.zeros(N)  # Right-hand side vector

    # Loop over all grid points (j for z-index, i for x-index)
    for j in range(Nz):
        for i in range(Nx):
            k = j * Nx + i # Flatten 2D index (j, i) into 1D index k

            # Check if current node has a Dirichlet condition (fixed potential)
            fixed, val = is_dirichlet(j, i)
            if fixed:
                # Dirichlet node: phi = val
                add_entry(k, k, 1.0)
                b[k] = val
                continue # Skip further processing for fixed nodes

            # For interior or Neumann boundary nodes, apply the finite difference stencil.
            # The Laplacian: (d^2φ/dx^2 + d^2φ/dz^2)= 0
            # We use a 5-point stencil:
            #   (phi(i+1,j) - 2*phi(i,j) + phi(i-1,j))/dx^2 +
            #   (phi(i,j+1) - 2*phi(i,j) + phi(i,j-1))/dz^2 = 0
    
             # --- x-direction finite differences ---
            if i == 0:
                 # Left boundary: Neumann condition (zero derivative)
                 # Use a one-sided difference where φ(i-1,j) is replaced by φ(i+1,j)
                coeff_center_x = -2.0/dx**2
                coeff_right = 2.0/dx**2
                # center
                add_entry(k, k, coeff_center_x)
                # right neighbor (i+1)
                k_right = j * Nx + (i+1)
            
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r:
                    b[k] -= coeff_right * val_r
                else:
                    add_entry(k, k_right, coeff_right)
            elif i == Nx-1:
                # Right boundary: Neumann (zero derivative)
                coeff_center_x = -2.0/dx**2
                coeff_left = 2.0/dx**2
                add_entry(k, k, coeff_center_x)
                k_left = j * Nx + (i-1)
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= coeff_left * val_l
                else:
                    add_entry(k, k_left, coeff_left)
            else:
                # Interior in x
                add_entry(k, k,  -2.0/dx**2)
                # right neighbor
                k_right = j * Nx + (i+1)
                fixed_r, val_r = is_dirichlet(j, i+1)
                if fixed_r:
                    b[k] -= (1.0/dx**2)*val_r
                else:
                    add_entry(k, k_right, 1.0/dx**2)
                # left neighbor
                k_left = j * Nx + (i-1)
                fixed_l, val_l = is_dirichlet(j, i-1)
                if fixed_l:
                    b[k] -= (1.0/dx**2)*val_l
                else:
                    add_entry(k, k_left, 1.0/dx**2)
            
            # z-direction
            if j == 0 or j == Nz-1:
                # These boundaries are Dirichlet (should have been caught)
                pass
            else:
                add_entry(k, k, -2.0/dz**2)
                # Up neighbor (j+1)
                k_up = (j+1)*Nx + i
                fixed_up, val_up = is_dirichlet(j+1, i)
                if fixed_up:
                    b[k] -= (1.0/dz**2)*val_up
                else:
                    add_entry(k, k_up, 1.0/dz**2)
                # Down neighbor (j-1)
                k_down = (j-1)*Nx + i
                fixed_down, val_down = is_dirichlet(j-1, i)
                if fixed_down:
                    b[k] -= (1.0/dz**2)*val_down
                else:
                    add_entry(k, k_down, 1.0/dz**2)

    # Build the sparse matrix in CSR format.
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Solve the linear system.
    phi_flat = spsolve(A, b)
    phi = phi_flat.reshape((Nz, Nx))
    return x, z, phi



def exz(x, z, phi, Nx, Nz):
    # -------------------------
    # Compute the electric field E = -grad(phi)
    # Use central differences in the interior and one-sided at boundaries.
    # -------------------------
    E_x = np.zeros_like(phi)
    E_z = np.zeros_like(phi)

    dx = x[1]-x[0]
    dz = z[1]-z[0]
    
    # x-direction derivative
    for j in range(Nz):
        for i in range(Nx):
            if i == 0:
                E_x[j,i] = - (phi[j, i+1] - phi[j, i]) / dx
            elif i == Nx-1:
                E_x[j,i] = - (phi[j, i] - phi[j, i-1]) / dx
            else:
                E_x[j,i] = - (phi[j, i+1] - phi[j, i-1]) / (2*dx)
    # z-direction derivative
    for j in range(Nz):
        for i in range(Nx):
            if j == 0:
                E_z[j,i] = - (phi[j+1, i] - phi[j, i]) / dz
            elif j == Nz-1:
                E_z[j,i] = - (phi[j, i] - phi[j-1, i]) / dz
            else:
                E_z[j,i] = - (phi[j+1, i] - phi[j-1, i]) / (2*dz)

    return E_x, E_z

def simulate_electron_transportx(x, z, E_x, E_z, electron_x0, start_z, stop_z, dt=0.05, max_steps=2000):
    """
    Simulate electron trajectories given the grid (x, z) and electric field (E_x, E_z).

    Parameters:
        x (1D array): x-grid coordinates.
        z (1D array): z-grid coordinates.
        E_x (2D array): x-component of electric field on grid (shape: [len(z), len(x)]).
        E_z (2D array): z-component of electric field on grid.
        electron_x0 (1D array): initial x-positions for electrons.
        start_z (float): starting z-position for electrons (scalar).
        stop_z (float): stopping z-position (e.g., anode position, scalar).
        dt (float): time step for Euler integration.
        max_steps (int): maximum number of steps for integration.
        
    Returns:
        trajectories (list of np.ndarray): Each element is an array of shape (N_steps, 2)
                                             containing the (x, z) positions of one electron.
    """
    
    # Create interpolators for the electric field components.
    # Note: The grid for interpolation is (z, x) since our field arrays are organized as [len(z), len(x)].
    interp_E_x = RegularGridInterpolator((z, x), E_x)
    interp_E_z = RegularGridInterpolator((z, x), E_z)
    
    trajectories = []
    
    # Loop over each electron's initial x-position.
    for x0 in electron_x0:
        traj = []
        # Initialize electron at (x0, start_z) where both x0 and start_z are scalars.
        pos = np.array([x0, start_z])  # pos: [x, z]
        traj.append(pos.copy())
        
        for _ in range(max_steps):
            # Prepare the point for interpolation in (z, x) order.
            pt = np.array([pos[1], pos[0]])
            try:
                # Extract the scalar electric field components using .item()
                Ex_local = interp_E_x(pt).item()
                Ez_local = interp_E_z(pt).item()
            except ValueError:
                # If the point is outside the interpolation region, stop this trajectory.
                break
            
            # Electrons move opposite to the electric field.
            drift = -np.array([Ex_local, Ez_local])
            
            # Normalize the drift direction to obtain a unit vector (if nonzero).
            norm = np.linalg.norm(drift)
            if norm != 0:
                drift = drift / norm
            
            # Update position: pos is (x, z)
            pos += dt * drift
            traj.append(pos.copy())
            
            # Stop if electron reaches the stopping z position.
            if pos[1] <= stop_z:
                break

                
        trajectories.append(np.array(traj))
    
    return trajectories

def simulate_electron_transport(x, z, E_x, E_z, electron_x0, params, dt=0.05, max_steps=2000):
    """
    Simulate electron trajectories given the grid (x, z) and electric field (E_x, E_z).

    Parameters:
        x (1D array): x-grid coordinates.
        z (1D array): z-grid coordinates.
        E_x (2D array): x-component of electric field on grid (shape: [len(z), len(x)]).
        E_z (2D array): z-component of electric field on grid.
        electron_x0 (1D array): initial x-positions for electrons.
        start_z (float): starting z-position for electrons (scalar).
        stop_z (float): stopping z-position (e.g., anode position, scalar).
        dt (float): time step for Euler integration.
        max_steps (int): maximum number of steps for integration.
        
    Returns:
        trajectories (list of np.ndarray): Each element is an array of shape (N_steps, 2)
                                             containing the (x, z) positions of one electron.
    """

    def in_hole(x_val, hole_center):
        """Return True if x_val is within the hole (centered at hole_center with diameter d1)."""
        return (x_val > (hole_center - d1/2)) and (x_val < (hole_center + d1/2))

    
    # Create interpolators for the electric field components.
    # Note: The grid for interpolation is (z, x) since our field arrays are organized as [len(z), len(x)].
    interp_E_x = RegularGridInterpolator((z, x), E_x)
    interp_E_z = RegularGridInterpolator((z, x), E_z)

    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
    x2_center = x1_center + p

    start_z = params.l3
    stop_z = params.l1
    stop_z0 = params.l0

    tolz = dt

    print(f"start_z ={start_z}, stop_anode_z ={stop_z}, stop_gnd_z = {stop_z0}")
    
    trajectories = []
    btrajectories = []
    
    # Loop over each electron's initial x-position.
    # Loop over each electron's initial x-position.

    #print(f"electron_x0 = {electron_x0}")
    for x0 in electron_x0:
        traj = []
        # Initialize electron at (x0, start_z) where both x0 and start_z are scalars.
        pos = np.array([x0, start_z])  # pos: [x, z]
        traj.append(pos.copy())
        
        for _ in range(max_steps):
            # Prepare the point for interpolation in (z, x) order.
            pt = np.array([pos[1], pos[0]])
            try:
                # Extract the scalar electric field components using .item()
                Ex_local = interp_E_x(pt).item()
                Ez_local = interp_E_z(pt).item()
            except ValueError:
                # If the point is outside the interpolation region, stop this trajectory.
                break
            
            # Electrons move opposite to the electric field.
            drift = -np.array([Ex_local, Ez_local])
            
            # Normalize the drift direction to obtain a unit vector (if nonzero).
            norm = np.linalg.norm(drift)
            if norm != 0:
                drift = drift / norm
            
            # Update position: pos is (x, z)
            pos += dt * drift
            traj.append(pos.copy())
            
            # Stop if electron reaches the stopping z position.
            if pos[1] <= stop_z:
                break
        
                
        trajectories.append(np.array(traj))
        
    #print(f"trajectories = {trajectories}")
    for traj in trajectories:
        pos = traj[-1] 
        pos[1] = pos[1] - 2*dt
        if  not in_hole(pos[0], x1_center):  # only trajectories which end up in hole
            continue
        btraj = []
        
        btraj.append(pos.copy())
        
        #print(f"in anode: pos  = {pos}")
       
        #btraj = []
        for ii in range(max_steps):
            pt = np.array([pos[1], pos[0]])
            try:
                # Extract the scalar electric field components using .item()
                Ex_local = interp_E_x(pt).item()
                Ez_local = interp_E_z(pt).item()
            except ValueError:
                # If the point is outside the interpolation region, stop this trajectory.
                break
            
            # Electrons move opposite to the electric field.
            drift = np.array([Ex_local, Ez_local])
            
            # Normalize the drift direction to obtain a unit vector (if nonzero).
            norm = np.linalg.norm(drift)
            if norm != 0:
                drift = drift / norm
            
            # Update position: pos is (x, z)
            pos += dt * drift
            #print(f"step {ii}: pos  = {pos}")
            btraj.append(pos.copy())
            

            # Stop if electron reaches the stopping z position.
            if pos[1] > stop_z - tolz or pos[1] <=stop_z0:
                break
        
        btrajectories.append(np.array(btraj))
        
        
    return trajectories, btrajectories

def electron_transport(x, z, E_x, E_z, electron_x0, params, dt=0.05, max_steps=2000):
    """
    Simulate electron trajectories given the grid (x, z) and electric field (E_x, E_z).

    Parameters:
        x (1D array): x-grid coordinates.
        z (1D array): z-grid coordinates.
        E_x (2D array): x-component of electric field on grid (shape: [len(z), len(x)]).
        E_z (2D array): z-component of electric field on grid.
        electron_x0 (1D array): initial x-positions for electrons.
        start_z (float): starting z-position for electrons (scalar).
        stop_z (float): stopping z-position (e.g., anode position, scalar).
        dt (float): time step for Euler integration.
        max_steps (int): maximum number of steps for integration.
        
    Returns:
        trajectories (list of np.ndarray): Each element is an array of shape (N_steps, 2)
                                             containing the (x, z) positions of one electron.
    """

    def get_elocal(pos):
        pt = np.array([pos[1], pos[0]])
        #print(f"pos = {pos}")
        #print(f"pt = {pt}")
        try:
            # Extract the scalar electric field components using .item()
            Ex_local = interp_E_x(pt).item()
            Ez_local = interp_E_z(pt).item()
            return Ex_local, Ez_local
        except ValueError:
            print(f"Error in interpolation for pt = {pt}, pt[0] = {pt[0]}, pt[1]={pt[1]}")
                #print(f"interp_E_x([16.8, -2.5])={interp_E_x([16.8, -2.5]).item()}")
                #print(f"interp_E_x([16.8, 2.5])={interp_E_x([16.8, 2.5]).item()}")
                #print(f"interp_E_z([16.8, -2.5])={interp_E_z([16.8, -2.5]).item()}")
                #print(f"interp_E_z([16.8, 2.5])={interp_E_z([16.8, 2.5]).item()}")
            return None 

    def step(pos, Ex, Ez):
        # Electrons move opposite to the electric field.
        drift = -np.array([Ex, Ez])
            
        # Normalize the drift direction to obtain a unit vector (if nonzero).
        norm = np.linalg.norm(drift)
        if norm != 0:
            drift = drift / norm
            
        # Update position: pos is (x, z)
        pos += dt * drift
        return pos

    def in_hole(x_val, hole_center):
        """Return True if x_val is within the hole (centered at hole_center with diameter d1)."""
        return (x_val >= (hole_center - d1/2)) and (x_val <= (hole_center + d1/2))

    
    # Create interpolators for the electric field components.
    # Note: The grid for interpolation is (z, x) since our field arrays are organized as [len(z), len(x)].
    interp_E_x = RegularGridInterpolator((z, x), E_x)
    interp_E_z = RegularGridInterpolator((z, x), E_z)

    #print(f"interp_E_x([16.8, -2.5])={interp_E_x([16.8, -2.5]).item()}")
    #print(f"interp_E_x([16.8, 2.5])={interp_E_x([16.8, 2.5]).item()}")
    #print(f"interp_E_z([16.8, -2.5])={interp_E_z([16.8, -2.5]).item()}")
    #print(f"interp_E_z([16.8, 2.5])={interp_E_z([16.8, 2.5]).item()}")
    

    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
   
    start_z = params.zdrft 
    stop_z = params.zanode

    tolz = dt/2
    
    trajectories = []
    btrajectories = []
    
    # Loop over each electron's initial x-position.
    # Loop over each electron's initial x-position.
    for x0 in electron_x0:
        traj = []
        # Initialize electron at (x0, start_z) where both x0 and start_z are scalars.
        pos = np.array([x0, start_z])  # pos: [x, z]
        traj.append(pos.copy())
        
        for _ in range(max_steps):
            # Prepare the point for interpolation in (z, x) order.
            result = get_elocal(pos)
            if result == None:
                break
            Ex, Ez = result 
            pos = step(pos, Ex, Ez)
            traj.append(pos.copy())
            
            # Stop if electron reaches the stopping z position.
            if pos[1] <= stop_z:
                break
        
                
        trajectories.append(np.array(traj))

    for traj in trajectories:
        pos = traj[-1]
        if  not in_hole(pos[0], x1_center):  # only trajectories which end up in hole
            continue
        btraj = []
        
        btraj.append(pos.copy())
        
        #print(f"in anode: pos  = {pos}")
        #btraj = []
        for _ in range(max_steps):
            # Prepare the point for interpolation in (z, x) order.
            result = get_elocal(pos)
            if result == None:
                break
            Ex, Ez = result 
            pos = step(pos, Ex, Ez)
            btraj.append(pos.copy())
            
            # Stop if electron reaches the stopping z position.
            if pos[1] > stop_z - tolz or pos[1] <=0:
                break
        
        btrajectories.append(np.array(btraj))
        
    return trajectories, btrajectories



def contour_phi(x, z, phi, figsize=(8,6)):

    fig, ax = plt.subplots(figsize=figsize)
    X, Z = np.meshgrid(x, z)

    # Plot equipotential contours
    contours = ax.contour(X, Z, phi, levels=20, linewidths=0.5, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title('Equipotential lines and electron trajectories')


def quiver_Exz(x, z, E_x, E_z, skip=5, figsize=(8,6)):

    fig, ax = plt.subplots(figsize=figsize)
    X, Z = np.meshgrid(x, z)

    # Plot electric field quiver (skip some points for clarity)
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(X[skip], Z[skip], E_x[skip], E_z[skip], color='red', scale=5e4)

def plot_traj(x, z, phi, trajectories, btr, params, figsize=(8,6)):

    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
    x2_center = x1_center + p

    start_z = params.l3
    gate = params.l2
    anode = params.l1
    
    X, Z = np.meshgrid(x, z)
    fig, ax = plt.subplots(figsize=figsize)
    contours = ax.contour(X, Z, phi, levels=20, linewidths=1.0 ,cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title('Equipotential lines and electron trajectories')
    # Plot electron trajectories
    for traj in trajectories:
        ax.plot(traj[:,0], traj[:,1], 'g', lw=0.3 )
    for traj in btr:
        ax.plot(traj[:,0], traj[:,1], 'r', lw=0.3 )

    plt.axvline(x=x1_center - d1/2, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=x1_center + d1/2, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=x2_center - d1/2, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=x2_center + d1/2, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=start_z, color='blue', linestyle='--', linewidth=1)
    plt.axhline(y=gate, color='blue', linestyle='--', linewidth=1)
    plt.axhline(y=anode, color='blue', linestyle='--', linewidth=1)


    plt.tight_layout()
    plt.show()


def plot_traj2(x, z, phi, trajectories, btr, params, figsize=(8,6)):

    x1_center = params.x1    # center of hole 1 (mm)
    d1 = params.d1
    p =  params.p             # pitch between holes (mm)
   

    start_z = params.zdrft 
    gate = params.zgate
    anode = params.zanode
    
    X, Z = np.meshgrid(x, z)
    fig, ax = plt.subplots(figsize=figsize)
    contours = ax.contour(X, Z, phi, levels=20, linewidths=1.0 ,cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title('Equipotential lines and electron trajectories')
    # Plot electron trajectories
    for traj in trajectories:
        ax.plot(traj[:,0], traj[:,1], 'g', lw=0.3 )
    for traj in btr:
        ax.plot(traj[:,0], traj[:,1], 'r', lw=0.3 )

    plt.axvline(x=x1_center - d1/2, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=x1_center + d1/2, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=start_z, color='blue', linestyle='--', linewidth=1)
    plt.axhline(y=gate, color='blue', linestyle='--', linewidth=1)
    plt.axhline(y=anode, color='blue', linestyle='--', linewidth=1)


    plt.tight_layout()
    plt.show()

def heatmap_phi(x, z, phi, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    X, Z = np.meshgrid(x, z)
    
    # Plot a heatmap of the potential using pcolormesh
    # shading='auto' adjusts the grid automatically (or use 'nearest')
    hm = ax.pcolormesh(X, Z, phi, shading='auto', cmap='viridis')
    fig.colorbar(hm, ax=ax, label='Potential (V)')
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title('Heatmap of the Potential')
    
    plt.show()