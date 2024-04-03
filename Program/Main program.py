import numpy as np
import math
import os


# parameter
n = 21
D = 3
LL = 20.0  # satuan Angstroms
dt = 0.001  # satuan ps
BC = 1  # 0 untuk periodic, 1 untuk reflecting
m = [1.0, 16.0]  # mass of particles in Daltons
qs = [0.41, -0.82]  # charges of particles
T0 = 300  # Temperatur mula-mula (Temperatur kamar)

# constants
kb = 0.8314459920816467  # boltzmann constant in useful units
NA = 6.0221409e26  # Avogardos constant x 1000 (g->kg)
ech = 1.60217662e-19  # electron charge in coulombs
kc = (
    8.9875517923e9 * NA * 1e30 * ech * ech / 1e24
)  # electrostatic constant in Daltons, electron charges, picosecond, angstrom units


# Interaction parameters, roughly copied from
# Xin Bo Zhang et al. Fluid Phase Equilibria 262 (2007) 210–216 doi:10.1016/j.fluid.2007.09.005
# note that 1kJ/mol = 100 Dal A^2 ps^-2 (for energy conversions from table 1 in that paper)
eps = [[3.24, 14.2723], [14.2723, 62.87]]
sig = [[0.98, 2.04845], [2.04845, 3.1169]]
Kr = (
    148000.0 / 2
)  # spring potential is usually defined as U = (k/2)(r-r_0)^2. I included the /2 here
bl = 0.9611
Kth = 35300.0 / 2  # same explanation as Kr but with bending energy
th0 = 109.47 * np.pi / 180.0  # angle in rad

# set dimensions of system. For simoplicity all set to LL
L = np.zeros([D]) + LL

# arrays of variables
r = np.random.rand(n, D) * LL  # initialize all random positions
v = np.random.rand(n, D) - 0.5  # initialize with random velocity

outdir = "dumps"
try:  # make directory if it does not exist
    os.mkdir(outdir)
except:
    pass

# bonds
bnd = []
for i in range(int(n / 3)):
    bnd.append([3 * i, 3 * i + 1, bl, Kr])
    bnd.append([3 * i + 1, 3 * i + 2, bl, Kr])
bnd = np.array(bnd)

# angles
angs = []
for i in range(int(n / 3)):
    angs.append([3 * i, 3 * i + 1, 3 * i + 2, th0, Kth])
angs = np.array(angs)

# Types in groups of three
tp = [0] * n
for i in range(int(n / 3)):
    tp[3 * i] = 0
    tp[3 * i + 1] = 1
    tp[3 * i + 2] = 0

# molecule labels
mols = [0] * n
for i in range(int(n / 3)):
    mols[3 * i] = i
    mols[3 * i + 1] = i
    mols[3 * i + 2] = i

# mass and charge arrays
mm = np.array([m[tp[j]] for j in range(n)])
chrg = np.array([qs[tp[j]] for j in range(n)])

# Lennard-Jones potential
# Given particle index, i, returns potential it feels from other particles
def LJpot(r, i, sigg, epss):
    sg = np.delete(np.array([sigg[tp[j]] for j in range(n)]), i)
    ep = np.delete(np.array([epss[tp[j]] for j in range(n)]), i)
    for ii in range(n):  # ignore atoms in the same molecule
        if mols[i] == mols[ii]:
            ep[ii] = 0
    drv = r - r[i]  # distance in each dimension
    drv = np.delete(drv, i, 0)  # remove ith element (no self LJ interactions)
    dr = [
        np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) for a in drv
    ]  # absolute distance
    r6 = (sg / np.array(dr)) ** 6
    r12 = (sg / np.array(dr)) ** 12
    LJP = 4.0 * eps * sum(ep * r6 - ep * r12)
    return LJP


# Gradient of Lennard-Jones potential
def dLJp(r, i, sigl, epsl, bnds):
    sg = np.delete(np.array([sigl[tp[j]] for j in range(n)]), i)
    ep = np.array([epsl[tp[j]] for j in range(n)])
    for ii in range(n):  # ignore atoms in the same molecule
        if mols[i] == mols[ii]:
            ep[ii] = 0
    ep = np.delete(ep, i)
    drv = r - r[i]  # distance in each dimension
    drv = np.delete(drv, i, 0)  # remove ith element (no self LJ interactions)
    dr = [
        np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) for a in drv
    ]  # absolute distance
    r8 = ep * (sg**6) * (1.0 / np.array(dr)) ** 8
    r14 = 2.0 * ep * (sg**12) * (1.0 / np.array(dr)) ** 14
    r8v = np.transpose(np.transpose(drv) * r8)
    r14v = np.transpose(np.transpose(drv) * r14)
    r8vs = np.sum(r8v, axis=0)
    r14vs = np.sum(r14v, axis=0)
    dLJP = 24.0 * (r14vs - r8vs)
    return dLJP


# bond length potential
def BEpot(r, bnds):
    bps = np.zeros(n)
    for i in range(n):  # loop over all particles
        for j in range(len(bnds)):  # check all bonds to see if particle i is bonded
            if bnds[j][0] == i or bnds[j][1] == i:
                if bnds[j][0] == i:  # find particle bonded to i
                    ii = int(bnds[j][1])
                else:
                    ii = int(bnds[j][0])
                dr0 = bnds[j][2]
                e0 = bnds[j][3]
                dr = r[i] - r[ii]
                dr2 = dr * dr
                adr2 = sum(dr2)
                adr = np.sqrt(adr2)
                BE = e0 * (adr - dr0) ** 2
                bps[i] += BE
    return bps


# gradient of bond length potential (negative force)
def dBEpot(r, bnds):
    bps = np.zeros([n, 3])
    for i in range(n):  # loop over all particles
        for j in range(len(bnds)):  # check all bonds to see if particle i is bonded
            if bnds[j][0] == i or bnds[j][1] == i:
                if bnds[j][0] == i:  # find particle bonded to i
                    ii = int(bnds[j][1])
                else:
                    ii = int(bnds[j][0])
                dr0 = bnds[j][2]
                e0 = bnds[j][3]
                dr = r[i] - r[ii]
                dr2 = dr * dr
                adr2 = sum(dr2)
                adr = np.sqrt(adr2)
                dBE = 2.0 * e0 * (adr - dr0) * dr / adr
                bps[i] += dBE
    return bps


# gradient of bond angle potential (negative force)
def dBA(r, angs):
    aps = np.zeros([n, 3])
    for i in range(n):  # loop over all particles
        for j in range(len(angs)):  # check all bonds to see if particle i is bonded
            a1 = int(angs[j][0])
            a2 = int(angs[j][1])
            a3 = int(angs[j][2])
            if i == a1 or i == a2 or i == a3:
                th00 = angs[j][3]  # equilibrium angle
                e0 = angs[j][4]  # bending modulus
                if i == a1 or i == a2:
                    r1 = r[a1] - r[a2]  # bond vector 1 (form middle atom to atom 1)
                    r2 = r[a3] - r[a2]  # bond vector 2 (middle atom to atom 2)
                else:
                    r1 = r[a3] - r[a2]  # bond vector 1 (form middle atom to atom 1)
                    r2 = r[a1] - r[a2]  # bond vector 2 (middle atom to atom 2)
                ar1 = np.sqrt(sum(r1 * r1))  # lengths of bonds
                ar2 = np.sqrt(sum(r2 * r2))
                dot = sum(r1 * r2)  # r1 dot r2
                ndot = dot / (
                    ar1 * ar2
                )  # normalize dot product by vector lengths i.e. get the cos of angle
                th = math.acos(ndot)  # bond angle, theta
                dUdth = -2.0 * e0 * (th - th00)  # -dU/dtheta
                if a1 == i or a3 == i:
                    numerator = (r2 / (ar1 * ar2)) - (
                        dot / (ar1 * ar1 * ar1 * ar2 * 2.0)
                    )
                    denominator = np.sqrt(1.0 - ndot * ndot)
                    dUdr = dUdth * numerator / denominator
                    aps[i] += dUdr
                if i == a2:
                    denominator = np.sqrt(1.0 - ndot * ndot)
                    n1 = -(r2 + r1)
                    n2 = dot * r1 / (ar1 * ar1)
                    n3 = dot * r2 / (ar2 * ar2)
                    numerator = (n1 + n2 + n3) / (ar1 * ar2)
                    dUdr = dUdth * numerator / denominator
                    aps[i] += dUdr
    return aps


# derivative of coulomb potential (negative force)
def coul(r, i, chrgs):
    q0 = chrgs[i]
    qs = 1.0 * np.array(chrgs)
    for j in range(n):
        if mols[i] == mols[j]:
            qs[j] = 0.0
    qs = np.delete(qs, i)
    drv = r - r[i]  # distance in each dimension
    drv = np.delete(drv, i, 0)  # remove ith element (no self LJ interactions)
    dr = [
        np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) for a in drv
    ]  # absolute distance
    r3 = q0 * qs * kc * ((1.0 / np.array(dr)) ** 3.0)
    FF = np.transpose(np.transpose(drv) * r3)
    Fs = np.sum(FF, axis=0)
    return Fs

# =============================================================================
# Output
# =============================================================================

def dump(r, t):
    """
    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fname = outdir + "/t" + str(t) + ".dump"
    f = open(fname, "w")
    f.write("ITEM: TIMESTEP\n")
    f.write(str(t) + "\n")  # time step
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(str(len(r)) + "\n")  # number of atoms
    f.write("ITEM: BOX BOUNDS pp pp pp\n")  # pp = periodic BCs
    f.write("0 " + str(L[0]) + "\n")
    f.write("0 " + str(L[1]) + "\n")
    f.write("0 " + str(L[2]) + "\n")
    f.write("ITEM: ATOMS id mol type x y z\n")
    for i in range(len(r)):
        f.write(
            str(i)
            + " "
            + str(mols[i])
            + " "
            + str(tp[i])
            + " "
            + str(r[i][0])
            + " "
            + str(r[i][1])
            + " "
            + str(r[i][2])
            + "\n"
        )
    f.close

# =============================================================================
# Boundary conditions
# =============================================================================

def BC_periodic(position, velocity):
    position_new = position % L
    velocity_new = 1.0 * velocity
    return position_new, velocity_new

def BC_reflective(position, velocity):
    velocity_new = 1.0 * velocity
    position_new = 1.0 * position
    for i in range(n):
        for j in range(D):
            if position_new[i][j] < 0:
                position_new[i][j] = -position_new[i][j]
                velocity_new[i][j] = abs(velocity[i][j])
            if position_new[i][j] > L[j]:
                position_new[i][j] = 2.0 * L[j] - position_new[i][j]
                velocity_new[i][j] = -abs(velocity[i][j])
    return position_new, velocity_new

# =============================================================================
# Particle state updaters
# =============================================================================

def update_velocity(r, v, dt, sigg, epss):
    # calculate acceleration:
    F = -np.array([dLJp(r, i, sigg[tp[i]], epss[tp[i]], bnd) for i in range(n)])  # LJ
    F = F - dBEpot(r, bnd)  # Bonds
    F = F - dBA(r, angs)  # Bonds angles
    F = F - np.array([coul(r, i, chrg) for i in range(n)])  # Coulomb
    a = np.transpose(np.transpose(F) / mm)  # Force->acceleration
    # update velocity
    newv = v + dt * a
    return newv, a

def rescale_velocity(v, T):
    KE = 0.5 * sum(sum(mm * np.transpose(v * v)))
    avKE = KE / n
    Tnow = (2.0 / 3) * avKE / kb
    lam = np.sqrt(T / Tnow)
    lam = (lam - 1.0) * 0.5 + 1.0  # update slowly
    vnew = lam * v
    return vnew

def update_position(position, velocity, dt):
    position_new = position + dt * velocity
    if BC == 0:
        position_new, velocity_new = BC_periodic(position_new, velocity)
    if BC == 1:
        position_new, velocity_new = BC_reflective(position_new, velocity)
    return position_new, velocity_new

# =============================================================================
# MAIN
# =============================================================================

skip = 20
STEPS = 50 * skip
for i in range(STEPS):
    # Update particle states
    v, a = update_velocity(r, v, dt, sig, eps)
    v = rescale_velocity(v, T0)
    r, v = update_position(r, v, dt)
    
    # Dump output?
    if i % skip == 0:
        dump(r, int(i / skip))
        print(int(i / skip))
