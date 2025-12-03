import numpy as np
from pyquaternion import Quaternion
import hoomd
import hoomd.md
#import hoomd.deprecated
import argparse
import os
#import scipy.spatial.distance
#from scipy.spatial.distance import cdist


####  SWIM PRESSURE IMPULSE ####

def swim_pressure_imp(timestep):
    global swim_force
    global N

    V = system.box.get_volume()
    a = system.particles

    # obtain net_force/velocity using particle proxy
    tot_force = np.asarray([np.asarray(a[i].net_force) for i in range(N)])
    # compute avg_force for system
    #avg_force = np.mean(tot_force, axis=0)
    # compute net_force by subtracting avg_force
    #net_force = tot_force - avg_force

    # obtain quaternion using particle proxy
    orient_quat = [Quaternion(a[i].orientation) for i in range(N)]
    # from quaternion obtain orientation vector
    orient_vec = np.asarray([np.asarray(orient_quat[i].rotate([1, 0, 0])) for i in range(N)])
    # get swim_vec by multiplying orientation by swim force
    swim_vec = swim_force * orient_vec
    # compute system swim force
    #avg_swim = np.mean(swim_vec, axis=0)
    # subtract avg_swim
    #net_swim = swim_vec - avg_swim

    swim_stress = np.asarray([np.outer(swim_vec[i], tot_force[i] ) for i in range(N)])
    swim_stress_avg = np.sum(swim_stress, axis=0)
    #swim_stress_sym = symmetrize(swim_stress_avg)
    swim_pressure_imp = tauR / 3 / system.box.get_volume() * np.trace(swim_stress_avg) #This is the so-called impusle form, which is defined at any instant in time
    #swim_pressure_imp = tauR/3 * np.trace(swim_stress_sym) / system.box.get_volume()
    np.set_printoptions(suppress=True)

#    print('swim3 \n', swim_stress_sym )

    return swim_pressure_imp

####  INTERACTION  PRESSURE ####

def interaction_pressure(timestep):
    global N
    
    V = system.box.get_volume()
    a = system.particles

    tot_virial = np.asarray([np.asarray(a[i].net_virial) for i in range(N)])

    sum_virial = np.sum(tot_virial, axis=0)
    sum_virial = (sum_virial[0] + sum_virial[3] + sum_virial[5])/3/V 
    #print(sum_virial[0], sum_virial[3], sum_virial[5], V)

    return sum_virial

seed = 123

hoomd.context.initialize("--mode=gpu")
#hoomd.util.quiet_status()

N = 55296

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--runlength")
parser.add_argument("-cf", "--phif")
parser.add_argument("-c0", "--phi0")
parser.add_argument("-p", "--pressure")
args = parser.parse_args()
runlength = np.float(args.runlength)
concf = np.float(args.phif)
conc0 = np.float(args.phi0)
P_int = int(args.pressure)
calc_P = False
if P_int > 0:
    calc_P = True
# without loss of generality, sigma, friction and swim speed are set to unity
U = 1.0 # swim speed
sigma = 1.0 # LJ diameter
zeta = 1.0 # translational drag

LR = runlength * sigma #the runlength provided in the input is in units of sigma
tauR = LR / U #reorientation time
DR = 1 / tauR #rotary diffusion
zeta_R = 1.0 / DR #rotational drag required to ensure correct reoriantion time
swim_force = zeta * U #swim force
#total_time = np.float(time) #time inputted is in units of convection time

tau = sigma / U #convection time
epsilon = 50 * swim_force * sigma #this stiffness ensures effective hard-sphere statistics 

#total_time = tau * total_time
tequil = tau * 10 #time to run simulation before compression
Ucompress = 0.01 * (U) #compression speed is in units of swim speed
#conc0 = 0.5 #initial volume fraction

# define integrator (brownian dynamics)
time_step = 5e-5 * tau #timestep based on shortest relaxation time, assumed to be convective swimming

#num_steps = total_time / time_step
output = 1.0 * tau / time_step

compresstime = tau


scratch_path = f"/home/dan/Research/Data/ABP/q12s/"
if not os.path.exists(scratch_path):
    os.makedirs(scratch_path)

# write output
gsd_file = scratch_path + f"ABP_q12_{str(runlength)}_N_{str(N)}_fluid.gsd"
log_output = scratch_path + f"ABP_q12_{str(runlength)}_N_{str(N)}_fluid_pressure.log"

hoomd.context.initialize("--mode=cpu")

#if not os.path.exists(gsd_file):

unit_cell = np.cbrt(4 * 4.0 / 3.0 * np.pi * (2.0**(1 / 6) * sigma / 2)**3.0 / conc0)
ncopy = np.int(np.cbrt(N / 4.0))

system = hoomd.init.create_lattice(unitcell=hoomd.lattice.fcc(a=unit_cell),
                                   n=ncopy) 

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
init_position = snapshot.particles.position #- np.array([unit_cell] * 3) / 4
snapshot.particles.position[:] = init_position
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

print(N)
print(V)

particle_volume = N * 4 / 3 * np.pi * (2.0**(1 / 6) * sigma / 2)**3

print(particle_volume / V)

Vs = particle_volume / np.arange(concf, conc0, 0.025)
Ls = Vs**(1.0/3.0)
Linitials = Ls[:-1]
Lfinals = Ls[1:]



lj = hoomd.md.pair.lj(r_cut=2.0**(1. / 6.) * sigma, nlist=hoomd.md.nlist.cell())
lj.pair_coeff.set('A', 'A', epsilon=epsilon, sigma=sigma)
swim_vec = [(swim_force, 0, 0) for i in range(N)]

# set active force
hoomd.md.force.active(group=group_all, seed=seed, f_lst=swim_vec, rotation_diff=0.0, orientation_link=True, orientation_reverse_link=False)

hoomd.md.integrate.mode_standard(dt=time_step, aniso=True)
bd = hoomd.md.integrate.brownian(group=group_all, kT=1.0, seed=42, noiseless_r=False, noiseless_t=True)
bd.set_gamma('A', zeta)
bd.set_gamma_r('A', zeta_R)
# ap = hoomd.md.compute.active_pressure(group_all, swim_force, tauR)

#hoomd.dump.gsd(filename=gsd_file, period=200000000000, group=group_all)
if os.path.exists(gsd_file):
    os.remove(gsd_file)
hoomd.dump.gsd(filename=gsd_file, period=int(output), group=group_all)

if calc_P:
    logger = hoomd.analyze.log(filename=log_output,
                               quantities=['interaction_pressure', 'swim_pressure_imp'],
                               period=int(output),
                               header_prefix='#',
                               phase=0,
                               overwrite=True)
    logger.register_callback('swim_pressure_imp', callback=swim_pressure_imp)
    logger.register_callback('interaction_pressure', callback=interaction_pressure)



# hoomd.run(tequil / time_step)
#else:
#    hoomd.dump.gsd(filename=gsd_file, period=int(output), group=group_all)
"""
points = []
current_time = 0
for i in range(len(Linitials)):
    if i == 0:
        points.append((current_time, float(Linitials[i])))
    current_time += i * 2 * tequil / time_step
    points.append((int(current_time), float(Linitials[i])))
    current_time += compresstime / time_step
    points.append((int(current_time), float(Lfinals[i])))
    if i == (len(Linitials) - 1):
        current_time += i * 2 * tequil / time_step
        points.append((int(current_time), float(Lfinals[i])))
"""
#print(points)
#Lvariant = hoomd.variant.linear_interp(points=points)
#hoomd.update.box_resize(L=Lvariant, period = 1/time_step, scale_particles=False)

# silence python call_backs to screen
# hoomd.util.quiet_status()

# tune neighbor list params
#nl.tune()

# run simulation
#hoomd.run(1)

for i in range(len(Linitials)):
    if i == 0:
        hoomd.run(2 * tequil / time_step)
    Lvariant = hoomd.variant.linear_interp([(0, Linitials[i]), (compresstime / time_step, Lfinals[i])])
    updater = hoomd.update.box_resize(L=Lvariant, period = 1/time_step, scale_particles=False)
    hoomd.run(compresstime / time_step)
    updater.disable()
    del Lvariant
    del updater
    hoomd.run(2 * tequil / time_step)

#hoomd.run(current_time - 1)
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
