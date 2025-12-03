import numpy as np
from pyquaternion import Quaternion
import hoomd
import hoomd.md
#import hoomd.deprecated
import argparse
import os
import scipy.spatial.distance
from scipy.spatial.distance import cdist

seed = 123

hoomd.context.initialize("--mode=gpu")
hoomd.util.quiet_status()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--N")
parser.add_argument("-l", "--runlength")
parser.add_argument("-t", "--time")
parser.add_argument("-c", "--phi")
args = parser.parse_args()
N = np.int(args.N)
runlength = np.float(args.runlength)
conc = np.float(args.phi)
time = np.float(args.time)

# without loss of generality, sigma, friction and swim speed are set to unity
U = 1.0 # swim speed
sigma = 1.0 # LJ diameter
zeta = 1.0 # translational drag

LR = runlength * sigma #the runlength provided in the input is in units of sigma
tauR = LR / U #reorientation time
DR = 1 / tauR #rotary diffusion
zeta_R = 1.0 / DR #rotational drag required to ensure correct reoriantion time
swim_force = zeta * U #swim force
total_time = np.float(time) #time inputted is in units of convection time

tau = sigma / U #convection time
epsilon = 50 * swim_force * sigma #this stiffness ensures effective hard-sphere statistics 

total_time = tau * total_time
tequil = tau * 200 #time to run simulation before compression
Ucompress = 0.01 * (U) #compression speed is in units of swim speed
conc0 = 0.74 #initial volume fraction


scratch_path = f"/home/dan/Research/Data/ABP/"
if not os.path.exists(scratch_path):
    os.makedirs(scratch_path)

# write output
gsd_file = scratch_path + f"ABP_slab_coexistence_{str(runlength)}_N_{str(N)}_phi_{str(conc)}_expansion.gsd"

hoomd.context.initialize("--mode=gpu")

#if not os.path.exists(gsd_file):

unit_cell = np.cbrt(4 * 4.0 / 3.0 * np.pi * (2.0**(1 / 6) * sigma / 2)**3.0 / conc0)
ncopy = np.int(np.cbrt(N / 4.0))

system = hoomd.init.create_lattice(unitcell=hoomd.lattice.fcc(a=unit_cell),
                                   n=[int(0.5 * np.cbrt(4.0) * ncopy),
                                      int(0.5 * np.cbrt(4.0) * ncopy),
                                      int(np.cbrt(4.0) * ncopy)]) 

#system = hoomd.init.create_lattice(unitcell=hoomd.lattice.fcc(a=unit_cell), n=ncopy)
snapshot = system.take_snapshot()

snapshot.particles.diameter[:] = sigma * 2.**(1. / 6.)
# set orientation & MoI
snapshot.particles.moment_inertia[:] = (1, 1, 1)
np.random.seed(seed)
orientation = 2.0 * np.pi * np.random.rand(snapshot.particles.N)
phi = 2.0 * np.pi * np.random.rand(snapshot.particles.N)
theta = 1.0 * np.pi * np.random.rand(snapshot.particles.N)
orient_quat = [(np.cos(orientation[i] / 2),
                np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.cos(phi[i]),
                np.sin(orientation[i] / 2) * np.sin(theta[i]) * np.sin(phi[i]),
                np.sin(orientation[i] / 2) * np.cos(theta[i])) for i in range(snapshot.particles.N)]
snapshot.particles.orientation[:] = orient_quat;
init_position = snapshot.particles.position - np.array([unit_cell] * 3) / 4
#snapshot.particles.position[:] = init_position
system.restore_snapshot(snapshot)
#pos = snapshot.particles.position[:]
#print(np.abs(np.subtract.outer(pos, pos)).shape)
#dist = ((pos[:, None] - pos[:, :, None])**2)
#dist = cdist(pos, pos)
#inds = np.unravel_index(np.array(range(len(dist.flatten())))[dist.flatten() == 0], dist.shape)
#print(len(inds[0][inds[0] != inds[1]]))
#assert False
#else:
#    system = hoomd.init.read_gsd(filename='', restart=gsd_file, frame=-1)

# define group 'all'
group_all = hoomd.group.all()

N = int(len(group_all))
V = system.box.get_volume()

Linitial = system.box.Lz
particle_volume = N * 4 / 3 * np.pi * (2.0**(1 / 6) * sigma / 2)**3
Vfinal = particle_volume / conc
Lfinal = Vfinal / (system.box.Lx * system.box.Ly)
compresstime = np.abs(Linitial - Lfinal) / Ucompress

lj = hoomd.md.pair.lj(r_cut=2.0**(1. / 6.) * sigma, nlist=hoomd.md.nlist.cell())
lj.pair_coeff.set('A', 'A', epsilon=epsilon, sigma=sigma)
print(sigma)
swim_vec = [(swim_force, 0, 0) for i in range(N)]

# set active force
hoomd.md.force.active(group=group_all, seed=seed, f_lst=swim_vec, rotation_diff=0.0, orientation_link=True, orientation_reverse_link=False)

# define integrator (brownian dynamics)
time_step = 5e-5 * tau #timestep based on shortest relaxation time, assumed to be convective swimming
num_steps = total_time / time_step
output = 100.0 * tau / time_step
hoomd.md.integrate.mode_standard(dt=time_step, aniso=True)
bd = hoomd.md.integrate.brownian(group=group_all, kT=1.0, seed=42, noiseless_r=False, noiseless_t=True)
bd.set_gamma('A', zeta)
bd.set_gamma_r('A', zeta_R)
# ap = hoomd.md.compute.active_pressure(group_all, swim_force, tauR)

#hoomd.dump.gsd(filename=gsd_file, period=200000000000, group=group_all)
if not os.path.exists(gsd_file):
    hoomd.dump.gsd(filename=gsd_file, period=int(output), group=group_all)
    hoomd.run(tequil / time_step)
#else:
#    hoomd.dump.gsd(filename=gsd_file, period=int(output), group=group_all)

print()
print("Linitial = " + str(Linitial))
print("Lfinal = " + str(Lfinal))
print()
Lvariant = hoomd.variant.linear_interp([(0, Linitial), (compresstime / time_step, Lfinal)])
hoomd.update.box_resize(Lz=Lvariant, period = 1/time_step, scale_particles=False)

# silence python call_backs to screen
hoomd.util.quiet_status()

# tune neighbor list params
#nl.tune()

# run simulation
hoomd.run(num_steps + compresstime / time_step)
"""
snapshot = system.take_snapshot()
pos = np.asarray(snapshot.particles.position[:])
mins = []
for i in range(len(pos)):
    mins.append(np.sort(np.sqrt((pos[i, 0] - pos[:, 0])**2 + (pos[i, 1] - pos[:, 1])**2 + (pos[i, 2] - pos[:, 2])**2))[1])
#print(np.sort(np.sqrt((pos[15, 0] - pos[:, 0])**2 + (pos[15, 1] - pos[:, 1])**2 + (pos[15, 2] - pos[:, 2])**2)))
#print(np.sort(np.sqrt((pos[14, 0] - pos[:, 0])**2 + (pos[14, 1] - pos[:, 1])**2 + (pos[14, 2] - pos[:, 2])**2)))
#print(mins)
#print(min(mins))
pdata = system.particles.pdata
energies = []
for i in range(len(pos)):
    energies.append(hoomd.data.particle_data_proxy(system.particles.pdata, i).net_energy)
print(energies)
hoomd.dump.gsd(filename=gsd_file, period=1, group=group_all)
hoomd.run(0)
#print(lj.get_net_force(group_all))
for i in range(10):
    print(i)
    hoomd.run(10)
"""
