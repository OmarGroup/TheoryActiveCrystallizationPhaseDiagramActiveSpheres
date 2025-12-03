import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gsd.hoomd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import SmoothBivariateSpline, UnivariateSpline
from scipy.signal import argrelextrema
from scipy.interpolate import Rbf
import pickle

N = 55296
runlength = 0.3
conc0 = 0.74
phi = 0.63
ntau_equilibrate = 200

U = 1.0 # swim speed
sigma = 1.0 # LJ diameter
tau = sigma / U #convection time
Ucompress = 0.01 * (U)
time_step = 5e-5 * tau
nsteps_equilibrate = int(ntau_equilibrate / time_step)
r = (2.0**(1. / 6.) * sigma / 2)
v = (4.0 / 3.0) * np.pi * r**3.0
print(2 * r)

zeta = 1.0 #translational drag

LR = runlength * sigma #the runlength provided in the input is in units of sigma
tauR = LR / U #reorientation time


def get_phi_est(a, c):
	t1 = -(-a**2 + 2 * a * c + 1)/(3 * a**2)
	t2 = (2 * a**6 + 6 * a**5 * c + 6 * a**4 * c**2 - 6 * a**4 + 2 * a**3 * c**3 + 6 * a**3 * c - 15 * a**2 * c**2 + 6 * a**2 + np.sqrt(-108 * a**8 * c**2 - 324 * a**7 * c**3 - 324 * a**6 * c**4 + 216 * a**6 * c**2 - 108 * a**5 * c**5 - 540 * a**5 * c**3 - 27 * a**4 * c**4 - 108 * a**4 * c**2) - 12 * a * c - 2)**(1/3)/(3 * 2**(1/3) * a**2)
	t3_num = -(2**(1/3) * (3 * a**2 * c * (c - 2 * a) - (-a**2 + 2 * a * c + 1)**2))
	t3_denom = (2 * a**6 + 6 * a**5 * c + 6 * a**4 * c**2 - 6 * a**4 + 2 * a**3 * c**3 + 6 * a**3 * c - 15 * a**2 * c**2 + 6 * a**2 + np.sqrt(-108 * a**8 * c**2 - 324 * a**7 * c**3 - 324 * a**6 * c**4 + 216 * a**6 * c**2 - 108 * a**5 * c**5 - 540 * a**5 * c**3 - 27 * a**4 * c**4 - 108 * a**4 * c**2) - 12 * a * c - 2)**(1/3)
	t3_denom *= 3 * a**2
	return t1 + t2 + t3_num / t3_denom


def get_pressure_df(filename, conc):
	Lcompress = (N * 4 / 3 * np.pi * (2.0**(1 / 6) * sigma / 2)**3 / conc)**(1 / 3)
	compresstime = np.abs(np.diff(Lcompress) / Ucompress)

	phi_and_t = [(0, conc[0]), (2 * ntau_equilibrate, conc[0])]
	total_t = 2 * ntau_equilibrate
	for phi, t in zip(conc[1:], compresstime):
		total_t += t
		phi_and_t.append((total_t, phi))
		total_t += ntau_equilibrate
		phi_and_t.append((total_t, phi))

	def get_phi(tau):
		for i in range(len(phi_and_t)):
			tau_i = phi_and_t[i][0]
			if tau < tau_i:
				phi_i = phi_and_t[i][1]
				phi_i_m1 = phi_and_t[i - 1][1]
				tau_i_m1 = phi_and_t[i - 1][0]
				m = (phi_i - phi_i_m1) / (tau_i - tau_i_m1)
				return m * (tau - tau_i_m1) + phi_i_m1

	pressure = np.loadtxt(filename, skiprows=1)

	pressure[:, 0] = (pressure[:, 0] - pressure[0, 0]) * time_step

	phi = np.array([get_phi(t) for t in pressure[:, 0]])

	measured_phis = np.round(phi[np.isin(phi, conc)], 4)

	measured_pressure = np.hstack((pressure[np.isin(phi, conc)], measured_phis.reshape(-1, 1)))

	measured_pressure = measured_pressure[measured_phis <= 0.731]

	df = pd.DataFrame()
	df["Active Pressure"] = measured_pressure[:, 2]
	df["Interaction Pressure"] = measured_pressure[:, 1]
	df["phi"] = measured_pressure[:, 3]

	return df.groupby('phi', as_index=False)[["Active Pressure", "Interaction Pressure"]].mean()


def linear_interp(phi_input, phi_known, pressure):
	output_pressure = []
	for phi in phi_input:
		diffs = phi_known - phi
		if phi >= min(phi_known) and phi <= max(phi_known):
			lower_i = np.argmin(np.abs(diffs[diffs <= 0]))
			upper_i = np.argmin(np.abs(diffs[diffs >= 0]))
			lower_phi = phi_known[diffs <= 0][lower_i]
			upper_phi = phi_known[diffs >= 0][upper_i]
			lower_p = pressure[diffs <= 0][lower_i]
			upper_p = pressure[diffs >= 0][upper_i]
			lower_p_contrib = lower_p * (upper_phi - phi) / (upper_phi - lower_phi)
			upper_p_contrib = upper_p * (phi - lower_phi) / (upper_phi - lower_phi)
			output_pressure.append(lower_p_contrib + upper_p_contrib)
		else:
			slopes = np.diff(pressure) / np.diff(phi_known)
			closest_i = np.argmin(np.abs(diffs))
			output_pressure.append(pressure[closest_i] + (phi - phi_known[closest_i]) * slopes[closest_i])
	return np.asarray(output_pressure)



runlengths_and_phis = [(0.8, 0.64),
					   (0.7, 0.64),
					   (0.6, 0.64),
					   (0.5, 0.64),
					   (0.4, 0.63),
					   (0.3, 0.63),
					   (0.2, 0.62),
					   (0.1, 0.56),
					   (0.08, 0.55),
					   (0.06, 0.55)][::-1]
runlengths = [0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
cutoffs = [0.65, 0.651, 0.687, 0.71, 0.715, 0.715, 0.715, 0.715, 0.71, 0.71, 0.71]

data = {}
for runlength, phi in runlengths_and_phis:
	if runlength == 0.5:
		density_input = f"fluid_and_solid_densities_{str(runlength)}_N_{str(N)}_phi_{str(phi)}_.txt"
	else:
		density_input = f"fluid_and_solid_densities_{str(runlength)}_N_{str(N)}_phi_{str(phi)}_3.txt"
	solid_density = max(np.loadtxt(density_input))
	conc_initial = solid_density
	conc_interval = 0.005
	conc_final = 0.74
	conc = np.arange(conc_initial, conc_final + conc_interval, conc_interval)[::-1]

	log_file = f"ABP_pressure_{str(runlength)}_N_{str(N)}_phi_{str(conc_final)}_.log"
	df = get_pressure_df(log_file, conc)
	phi = df["phi"]
	# print(runlength)
	# print(phi)
	# print()
	P = df["Active Pressure"] + df["Interaction Pressure"]
	data[runlength]= np.hstack((np.asarray(phi).reshape(-1, 1), np.asarray(df["Interaction Pressure"]).reshape(-1, 1), np.asarray(df["Active Pressure"]).reshape(-1, 1)))

data2 = {}
for i in range(len(runlengths)):
	runlength = runlengths[i]
	cutoff = cutoffs[i]
	print(runlength * 0.5 / r)
	P1 = None
	df1 = pd.DataFrame()
	if runlength in [0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
		P1 = np.loadtxt(f'crystal_pressure/crystal_compress_{str(runlength)}_pressure_phi_q122.log')
	else:
		P1 = np.loadtxt(f'crystal_pressure/crystal_compress_{str(runlength)}_pressure_phi_q12.log')
	df1["Active Pressure"] = P1[:, 2]
	df1["Interaction Pressure"] = P1[:, 1]
	df1["phi"] = P1[:, 3]
	df1["q12"] = P1[:, 4]

	group_df1 = df1.groupby('phi', as_index=False)[["Active Pressure", "Interaction Pressure", "q12"]]#.mean()
	# group_df1_mean2 = group_df1.apply(lambda x: np.mean(x[int((len(x) - 1) / 2):]))
	group_df1_mean2 = group_df1.apply(lambda x: x.iloc[len(x) // 2:].mean())
	# assert False

	phi1 = np.asarray(group_df1_mean2['phi'])
	Pc1 = np.asarray(group_df1_mean2['Interaction Pressure'])
	Pa1 = np.asarray(group_df1_mean2['Active Pressure'])
	q121 = np.asarray(group_df1_mean2['q12'])
	psi1 = (q121 - 0.286) / (0.575 - 0.286)
	data2[runlength] = np.hstack((phi1.reshape(-1, 1), Pc1.reshape(-1, 1), Pa1.reshape(-1, 1)))

load_data_crystal = np.loadtxt('q12_vs_phi_crystal2.txt')
load_data_fluid = np.loadtxt('q12_vs_phi_fluid2.txt')
load_data_coex = np.loadtxt('q12_vs_phi_coexistence.txt')
# print(load_data)

runlengths = [0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5]
q12_data_crystal = defaultdict(list)
q12_data_fluid = defaultdict(list)

for runlength in runlengths:
	q12_data_crystal[runlength] = list(map(tuple, load_data_crystal[np.isclose(runlength, load_data_crystal[:, 0])][:, 1:]))
	q12_data_fluid[runlength] = list(map(tuple, load_data_fluid[np.isclose(runlength, load_data_fluid[:, 0])][:, 1:]))
# print(q12_data)

syms = ['o', 'v', 's', 'd', 'P', '^', '<', '>', '*', 'X', 'p']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']

# plt.figure(figsize=(5, 4))
plts = []
firstp = []
plt.figure(figsize=(12, 8))
print()
for i in range(len(runlengths)):
	runlength = runlengths[i]
	if runlength < 0.45 * 2 * r and runlength > 0.05:
		print(runlength * 0.5 / r)
		phi_and_q12_crystal = q12_data_crystal[runlength]
		phi_and_q12_fluid = q12_data_fluid[runlength]
		j = i * 1
		if j > (len(colors) - 1):
			j -= len(colors)
		ps = np.asarray([p for p, _ in phi_and_q12_crystal])
		qs = np.asarray([q for _, q in phi_and_q12_crystal])
		plt.scatter(ps, (qs - 0.285) / 0.315, facecolors='none', edgecolors=colors[j], marker=syms[j], lw=2.5, s=100)
		firstp.append([(min(ps), (qs[np.argmin(ps)] - 0.285) / 0.315), (ps[np.argmin(ps) - 1], (qs[np.argmin(ps) - 1] - 0.285) / 0.315)])
		plt.scatter([p for p, _ in phi_and_q12_fluid], [(q - 0.285) / 0.315 for _, q in phi_and_q12_fluid], facecolors='none', edgecolors=colors[j], marker=syms[j], lw=2.5, s=100)
print()

# runlengths = [500.0, 250.0, 166.5, 122.5, 100.0, 83.5, 71.5, 62.5, 55.5, 50.0, 45.0, 40.0, 37.5, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 7.5, 5.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.05][::-1]
q12_data_fluid = defaultdict(list)

for runlength in runlengths:
	q12_data_fluid[runlength] = list(map(tuple, load_data_fluid[np.isclose(runlength, load_data_fluid[:, 0])][:, 1:]))

lastp = []
# plt.figure()
for i in range(len(runlengths[1:])):
	runlength = runlengths[1:][i]
	# if runlength < 0.29 and runlength > 0.05:

	phi_and_q12_fluid = q12_data_fluid[runlength]
	j = i * 1
	if j > (len(colors) - 1):
		j = i % len(colors)
	phi = np.linspace(0.05, 0.6, 10000)
	P_act_fit = (zeta * U / (np.pi * 4 * r**2)) * phi * (0.5 * runlength / r) / (1 + (1 - np.exp(-2**(7.0/6.0) * 0.5 * runlength / r)) * phi / (1 - phi / 0.645))
	P_int_fit = (zeta * U / (np.pi * 4 * r**2)) * 6 * 2**(-7.0/6.0) * phi**2 / np.sqrt(1 - phi / 0.645)
	P = P_act_fit + P_int_fit
	possible = phi[:-1][np.diff(P) / np.diff(P_int_fit) < 0]
	phi = np.linspace(0.025, 0.74, 1000)
	# y = get_psi(runlength, phi)
	# y[phi > phi_star] = 1 / (0.74 - phi_star) * (phi[phi > phi_star] - phi_star)
	# plt.plot(phi, y, color=colors[j], ls='--')
	ps = np.asarray([p for p, _ in phi_and_q12_fluid])
	qs = np.asarray([q for _, q in phi_and_q12_fluid])
	lastp.append([(max(ps), (qs[np.argmax(ps)] - 0.285) / 0.315), (ps[np.argmax(ps) + 1], (qs[np.argmax(ps) + 1] - 0.285) / 0.315)])
	# plts.append(plt.scatter(ps, (qs - 0.285) / 0.315, c='white', edgecolors=colors[j], marker=syms[j]))

kBT = zeta * U * runlength / 6
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma

A = 3.576 * (1 + np.tanh(0.287 * np.log(ell_div_s) - 1.611))
B = 1.049
C = 0.554

phi = np.linspace(1e-5, 0.74 - 1e-5, 1000)



n = [0, 1, 2, 3, 4, 5, 6, 7, 8]
cn = [1, 1.1649e-1, 2.217e-2, 1.84e-3, 3.373e-5, -1.117e-5, -8.914e-6, -9.469e-7, -4.356e-7]

# First the fluid EOS (from Oscar):
# Pa_fit = (zeta * U / (np.pi * 4 * r**2)) * (ell_div_D) * rho * v / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * rho * v / (1 - rho / rho_max_fit))
Pa = E_div_v * phi * ell_div_D * np.exp(-A * phi**B / (1 - phi / 0.645)**C)
Pc_act = E_div_v * 2**(-7/6) * phi**2 / (1 - phi / 0.645)**0.5
Pc_HS = E_div_v * 4 * kBT / Ea * phi**2 * np.sum([cn[i] * (4 * phi)**n[i] for i in range(len(n))]) / (1 - phi / 0.645)**0.76


# We also have psi^*, rho_max, and phi_ODT:

# m_ODT, c_ODT = (10.559847151050855, 0.08194286572998821)
m_ODT, c_ODT = (3.2608314300965056, 0.3067711951206309)
k_phi_ODT = 0.85 * np.arctanh(0.01 / (0.645 - 0.515)) / (np.log(0.06 * 0.5 / r + 1.1) - np.log(1.1))
def get_phi_ODT(runlength):
	ell_div_D = runlength * 0.5 / r
	ell_div_s = runlength / sigma
	# if runlength < 1e-6:
	# 	return 0.515
	# else:
	# 	return (0.515 + 1 * 0.0635) + 1 * 0.0635 * np.tanh(2 * np.log(m_ODT * ell_div_D + c_ODT))
	# return 0.515 + (0.645 - 0.515) * np.tanh((m_ODT*ell_div_D + c_ODT) * (np.exp(ell_div_D) - 1))
	return 0.515 + (0.645 - 0.515) * np.tanh(k_phi_ODT * np.log(ell_div_D + 1.1) - k_phi_ODT * np.log(1.1))
	# return 0.515 + (0.645 - 0.515) * np.tanh(2 * (np.log(m_ODT * 0.5 * runlength / r + c_ODT) - np.log(c)))


m_psi, c_psi = (18.821730673114374, -13.08211494066777)
def get_psi_star(phinew, runlength):
	ell_div_D = runlength * 0.5 / r
	ell_div_s = runlength / sigma
	cutoff = get_phi_ODT(runlength)
	offset = 0.1
	psinew2 = np.tanh(np.exp(m_psi * phinew + c_psi - 2 * np.log(1e-2 + (ell_div_D)**1.16 / (1 + np.log(1 + (ell_div_D)**2))) * (1 - phinew / 0.74) + 0.05 * phinew / (1 - phinew / 0.74)**0.5))
	psinew = np.asarray([0.0] * len(phinew))
	psinew[phinew > cutoff] = psinew2[phinew > cutoff] * 1
	return psinew


# m_pm, c_pm = (0.49810356147235874, 2.769368150263331)
# def get_phi_max(phinew, runlength):
# 	# p_pm = [0.41668844, -0.56332294, 0.87911393]
# 	p_pm = [12.014714048819592, -38.7720551812438, 46.913644332337306, -25.108621508599636, 5.688573468751057]
# 	# p_pm = [550.9679769388564, -2721.181402502046, 5585.805578656969, -6098.177947539839, 3733.5127016273805, -1215.0109942147294, 164.82187422182696]
# 	deg = 4
# 	ell_div_D = runlength * 0.5 / r
# 	# ell_div_s = runlength / sigma
# 	psinew = get_psi_star(phinew, runlength)
# 	# phi_max = 0.645 + np.tanh(10 * psinew) * (0.74 - 0.645) * np.tanh((5 + np.log(1 + ell_div_D)) * psinew**1)
# 	# return phi_max
# 	s = 0
# 	for i in range(deg + 1):
# 		print(i)
# 		s += p_pm[i] * psinew**(deg - i)
# 	# return s
# 	return 0.645 + np.tanh(10 * psinew) * (0.74 - 0.645) * np.tanh((1.5 + np.log(1 + ell_div_D)) * psinew**0.5)

def get_phi_max(psi, runlength):
    # phi_max = 0.645 * np.ones(len(psi))
    # phi_max[psi > 1e-5] = 0.74
    k = 15.847584758475847 #- (1 - np.exp(-10 * 0.5 * runlength / r))
    phi_max = 0.645 + (0.74 - 0.645) * np.tanh(k * psi) 
    return phi_max

def get_beta(psi, runlength):
    # beta = (0.5 - 1/3) * np.ones(len(psi))
    # beta[psi > 1e-5] = 0.075
    k = 1
    beta_HS = 1.1
    ell_0c = 18.8
    # beta = 0.5 + (0.4 - 0.5) * (0.5 + np.tanh(10 * psi * (0.5 * runlength / r - 18.0)) / 2)
    beta = 0.5 + np.tanh(10 * psi_star) * ( (0.1 - 0.5) * (1 + np.tanh(5 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2 + (0.275 - 0.1) * (1 + np.tanh(0.01 * (0.5 * runlength / r - 50 * 0.5 / r)**1)) / 2 )
    return beta

# def get_beta(phinew, runlength):
# 	ell_div_D = runlength * 0.5 / r
# 	ell_div_s = runlength / sigma
# 	psinew = get_psi_star(phinew, runlength)
# 	beta = np.zeros(len(psinew)) + 0.5
# 	beta[psinew > 0] = 0.5 - (0.1 + 0.3 * (0.5 + 0.5 * np.tanh(10 * ((runlength - 19.75) / r * 0.5)**1))) * np.tanh(5 * psinew[psinew > 0] * 1)
# 	return beta

p = [0.01109925, 0.04995078, 0.16538631, 0.41931119]
def get_frac_act(runlength):
	ell_div_D = runlength * 0.5 / r
	ell_div_s = runlength / sigma
	rl_x = np.log(ell_div_D)
	x = p[3] + p[2] * rl_x + p[1] * rl_x**2 + p[0] * rl_x**3
	x = max([x, 0.0])
	x = min([x, 1.0])
	return x


def get_coex(P, phi, psi, spinodall, spinodalh, E_r):
	sort = np.argsort(E_r)
	P = P[sort]
	phi = phi[sort]
	psi = psi[sort]
	E_r = E_r[sort]
	min_P = P[np.argmin(np.abs(phi - spinodalh))]
	max_P = P[np.argmin(np.abs(phi - spinodall))]
	phi_P_cp = 0.74
	phi1, phi2, P_coex = None, None, None
	if len(phi[((P - min_P) > 0) & (phi < spinodall)]) > 0:
		min_phi = min(phi[((P - min_P) > 0) & (phi < spinodall)])
		if len(phi[((P - max_P) < 0) & (phi > spinodalh)]) == 0:
			spinodall -= 1e-4
			max_P = P[np.argmin(np.abs(phi - spinodall))]
		max_phi = max(phi[((P - max_P) < 0) & (phi > spinodalh)])
		phi_init = phi[(phi > min_phi) & (phi < spinodall)]
		P_init = P[(phi > min_phi) & (phi < spinodall)]
		P_init = P_init[phi_init < phi_P_cp]
		phi_init = phi_init[phi_init < phi_P_cp]
		P_fin = P[(phi < max_phi) & (phi > spinodalh)]
		phi_fin = phi[(phi < max_phi) & (phi > spinodalh)][np.argmin(np.abs(np.subtract.outer(P_fin, P_init)), axis=0)]
		if len(P_fin) > 0 and len(P_init) > 0 and len(phi_fin) > 0:
			P_relevant = P[(phi < max_phi) & (phi > min_phi)]
			phi_relevant = phi[(phi < max_phi) & (phi > min_phi)]
			best = 100000000000000
			for i in range(len(phi_init)):
				if np.abs(phi_init[i] - phi_fin[i]) > 0.001:
					if len(E_r) < len(phi):
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[1:] * np.diff(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])])))
					else:
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[1:] * np.diff(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	return phi1, phi2, P_coex



# First we need to confirm psi_star and fix the phi_ODT fit

phi_stars = []
phi_stars_pred = []
ell_div_Ds = []
m, c = (3.2608314300965056, 0.3067711951206309)
# plt.figure()
count = 0
rls_pred = []
for p in range(len(firstp)):
	runlength = runlengths[1:][p]
	print(runlength * 0.5 / r)
	if runlength > 0.06: 
		count += 1
		ell_div_Ds.append(0.5 * runlength / r)
		# m = (firstp[p][1] - lastp[p][1]) / (firstp[p][0] - lastp[p][0])
		firstm = (firstp[p][1][1] - firstp[p][0][1]) / (firstp[p][1][0] - firstp[p][0][0])
		lastm = (lastp[p][1][1] - lastp[p][0][1]) / (lastp[p][1][0] - lastp[p][0][0])
		x = np.linspace(0.05, 0.74, 1000)
		firsty = firstm * x + (firstp[p][0][1] - firstm * firstp[p][0][0])
		# lasty = lastm * x + (lastp[p][0][1] - lastm * lastp[p][0][0])
		lasty = x * 0
		# phi_star = x[np.argmin(np.abs(y))]
		# print(firstp[p])
		# print(lastp[p])
		# assert False
		phi_star = (firstp[p][0][1] - firstm * firstp[p][0][0]) - (lastp[p][0][1] - lastm * lastp[p][0][0]) / (lastm - firstm)
		phi_star = x[np.argmin(np.abs(firsty - lasty))]
		phi_star_pred = get_phi_ODT(runlength)
		# phi_stars.append(phi_star)
		if runlength < 0.35:
			phi_stars.append(phi_star)
			rls_pred.append(runlength)
		psi_star_pred = get_psi_star(x, runlength)
		# plt.scatter(0.5 * runlength / r, phi_star, color=colors[p])
		# plt.scatter(0.5 * runlength / r, phi_star_pred, c='white', edgecolors=colors[p])
		# print([firstp[p][0], lastp[p][0]], [firstp[p][1], lastp[p][1]])



		# plt.scatter(0.5 * runlength / r, np.sqrt(np.exp(np.arctanh((phi_star - 0.515 - 0.0635) / 0.0635))), color=colors[p])
		# plt.scatter(0.5 * runlength / r, m * 0.5 * runlength / r + c, c='white', edgecolors=colors[p])
		# plt.plot(x, y)
		# print(lastp[p])
		# plt.scatter(firstp[p][0][0], firstp[p][0][1], color=colors[p])
		# plt.scatter(firstp[p][1][0], firstp[p][1][1], color=colors[p])
		# plt.plot(x, firsty, color=colors[p])
		# plt.scatter(lastp[p][0][0], lastp[p][0][1], color=colors[p], marker='s')
		# plt.scatter(lastp[p][1][0], lastp[p][1][1], color=colors[p], marker='s')
		# plt.plot(x, lasty, color=colors[p])
		# # plt.plot(phi_star, firstm * phi_star + (firstp[p][0][1] - firstm * firstp[p][0][0]), color=colors[p], marker='.')
		# plt.plot(phi_star_pred, firstm * phi_star_pred + (firstp[p][0][1] - firstm * firstp[p][0][0]), color='k', marker='.')
		plt.plot(x, psi_star_pred, color=colors[p], ls='--',lw=3.5, zorder=-100)
		# plt.xlim([0.45, 0.74])

plt.rcParams["font.family"] = "Arial"

phi = np.linspace(0.05, 0.74, 1000)
rls = [4 * 2 * r, 40 * 2 * r, 400 * 2 * r]
for i in range(len(rls)):
	runlength = rls[i]
	psi_star = get_psi_star(phi, runlength)
	plt.plot(phi, psi_star, color=colors[count + i], ls='--',lw=3.5, zorder=-100)

plt.xlabel(r'$\phi$', fontsize=14)
plt.ylabel(r'$\psi^*$', fontsize=18, rotation=0, labelpad=15)

ax = plt.gca()
ax.tick_params(width=3, length=10, which='major', labelsize=16)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)
# plt.scatter(phi_stars, [0] * len(phi_stars)

plt.show()
# plt.savefig('psi_vs_phi.svg', format='svg', dpi=1200)
# plt.close()


runlength = np.logspace(-1.5, 1.3, 1000)
phi_ODT = get_phi_ODT(runlength)
# print(phi_ODT)



rls_pred = np.asarray(rls_pred)

print(len(rls_pred))
print(len(phi_stars))

plt.figure(figsize=(5, 4))
plt.scatter(rls_pred * 0.5 / r, phi_stars, s=100, edgecolors='tab:blue', facecolors='none', lw=3, zorder=100)
plt.plot(runlength * 0.5 / r, phi_ODT, ls='-', lw=3.5)

plt.xlabel(r'$\ell_0 / D$', fontsize=14)
plt.ylabel(r'$\phi^{\rm ODT}$', fontsize=18, rotation=0, labelpad=15)
plt.xscale('log')
ax = plt.gca()
ax.tick_params(width=3, length=10, which='major', labelsize=16)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)
plt.yticks([0.52, 0.56, 0.60, 0.64])
# print(ax.get_xticklabels())
ax.set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100, 1000])

plt.show()
# plt.savefig('phiODT_vs_runlength.svg', format='svg', dpi=1200)
# plt.close()


# assert False

phi_pts = np.asarray([0.5500000180936085, 0.5500000180936085, 0.5750000489190953])
psi_pts = np.asarray([0.15863152294799082, 0.08840443883586037, 0.09538000757648735])
runlength_pts = np.log(np.asarray([0.08 * 0.5 / r, 0.1 * 0.5 / r, 0.2 * 0.5 / r]) + 1.1)

coefficients, _, _, _ = np.linalg.lstsq(np.vstack([np.ones(len(phi_pts)), phi_pts, runlength_pts]).T, psi_pts, rcond=None)

# Extract the intercept and coefficients
intercept = coefficients[0]
coef_phi = coefficients[1]
coef_runlength = coefficients[2]

print((intercept, coef_phi, coef_runlength))

d_psi_phi_d_phi = coef_phi * 0.515
# d_psi_phi_d_phi = coef_phi + 2 * d2_psi_phi_d_phi2 * 0.515
psi_ODT_0 = intercept + d_psi_phi_d_phi
# m_2 = (-coef_runlength - m_1 * 0.515 - m_1 / 6) / psi_ODT_0
print((d_psi_phi_d_phi, psi_ODT_0))



# k_phi_ODT = 0.85 * np.arctanh(0.01 / (0.645 - 0.515)) / (np.log(0.06 * 0.5 / r + 1.1) - np.log(1.1))
# rl = np.linspace(0, 1)
# phi_ODT = 0.515 + (0.645 - 0.515) * np.tanh(k_phi_ODT * np.log(rl + 1.1) - k_phi_ODT * np.log(1.1))

# k_psi_ODT = ((0.645 - 0.515) * k_phi_ODT * coef_phi - coef_runlength) / psi_ODT_0
# psi_ODT = psi_ODT_0 * (1 - np.tanh(k_psi_ODT * np.log(rl + 1.1) - k_psi_ODT * np.log(1.1)))


# plt.figure()
# plt.plot(rl, phi_ODT)
# plt.show()

# plt.figure()
# plt.plot(rl, psi_ODT)
# plt.show()
# assert False



m = 1
# psi_ODT = psi_pts - m * (phi_pts - )

# m, c = np.linalg.lstsq(np.vstack([phi_pts, np.ones(len(phi_pts))]).T, psi_pts, rcond=None)[0]
# psi_ODT = 




# assert False

# plt.figure()
# y = np.sqrt(np.exp(np.arctanh((np.asarray(phi_stars) - 0.515 - 0.0635) / 0.0635)))
# x = 0.5 * np.asarray(runlengths2) / r
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]
# print((m, c))
# plt.plot(x, m * x + c, color='k', ls='--')
# min_rl = 1e-2
# ell_div_Ds.insert(0, min_rl)
# phi_stars.insert(0, 0.515)
# phi_stars_pred.insert(0, get_phi_ODT(min_rl))
# phi_stars = np.asarray(phi_stars)
# ell_div_Ds = np.asarray(ell_div_Ds)

# f = np.arctanh((phi_stars - 0.515) / (0.645 - 0.515))
# f_prime = f / (np.exp(ell_div_Ds) - 1)

# best = 1e10
# best_c = 0
# best_A_1 = 0
# best_A_2 = 0
# for A_1 in np.linspace(0.8, 1.6, 100):
# 	for A_2 in np.linspace(0.8, 1.6, 100):
# 		cs = np.arctanh(f_prime / A_1 - A_2 / A_1) - np.log(ell_div_Ds)
# 		# if not np.any(np.isnan(cs)):
# 		# print(f_prime / A_1 - A_2 / A_1)
# 		# assert False
# 		c = np.mean(cs)
# 		f_prime_pred = A_1 + A_2 * np.tanh(np.log(ell_div_Ds) + c)
# 		res = np.sum((f_prime - f_prime_pred)**2)
# 		if res < best:
# 			best = res
# 			best_c = c
# 			best_A_1 = A_1
# 			best_A_2 = A_2
# print(best)
# print((best_A_1, best_A_2, best_c))

# f_prime_pred = best_A_1 + best_A_2 * np.tanh(np.log(ell_div_Ds) + best_c)


# y = [f_prime[i] for i in range(len(f_prime)) if i != 3]
# x = [ell_div_Ds[i] for i in range(len(f_prime)) if i != 3]
# m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
# print((m, c))


# plt.scatter(ell_div_Ds, f_prime)
# plt.plot(np.linspace(0, max(ell_div_Ds), 100), m*np.linspace(0, max(ell_div_Ds), 100) + c)
# plt.show()

# plt.figure()
# plt.scatter(ell_div_Ds, f)
# plt.plot(np.linspace(0, max(ell_div_Ds), 100), (m*np.linspace(0, max(ell_div_Ds), 100) + c) * (np.exp(np.linspace(0, max(ell_div_Ds), 100)) - 1))
# plt.show()

# plt.figure()
# plt.scatter(ell_div_Ds, phi_stars)
# plt.plot(np.linspace(0, max(ell_div_Ds), 100), 0.515 + (0.645 - 0.515) * np.tanh((m*np.linspace(0, max(ell_div_Ds), 100) + c) * (np.exp(np.linspace(0, max(ell_div_Ds), 100)) - 1)))
# plt.show()

# assert False


# plt.figure()
# # ell_div_Ds2 = np.logspace(0.0, 1.0, 10000)
# ell_div_Ds2 = np.linspace(min_rl, 2.0, 10000)
# # ell_div_Ds2 = np.asarray([])
# for rl in ell_div_Ds2:
# 	phi_star_pred = get_phi_ODT(rl * 2 * r)
# 	phi_stars_pred.append(phi_star_pred)
# plt.scatter(np.asarray(ell_div_Ds), phi_stars, s=100, edgecolors='tab:blue', facecolors='none', lw=3, zorder=100)
# sort = np.argsort(np.asarray(ell_div_Ds + list(ell_div_Ds2)))
# plt.plot(np.asarray(ell_div_Ds + list(ell_div_Ds2))[sort], np.asarray(phi_stars_pred)[sort], ls='-', lw=3.5)
# plt.xlabel(r'$\ell_0 / D$', fontsize=14)
# plt.ylabel(r'$\phi^{\rm ODT}$', fontsize=18, rotation=0, labelpad=15)
# plt.xscale('log')
# ax = plt.gca()
# ax.tick_params(width=3, length=10, which='major')
# ax.tick_params(width=3, length=6, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3.5)
# plt.yticks([0.52, 0.56, 0.60, 0.64])
# # plt.xlabel(r'$\phi$', fontsize=14)
# # plt.ylabel(r'$\psi^*$', fontsize=18, rotation=0, labelpad=15)
# # plt.legend(plts, runlengths[1:], title=r'$\frac{l_0}{\sigma}$', title_fontsize=14)
# # plt.scatter(phi_stars, [0] * len(phi_stars))
# plt.show()
# # plt.savefig('phiODT_vs_runlength.svg', format='svg', dpi=1200)

# # Now we need psi star

q12_data_crystal = defaultdict(list)
q12_data_fluid = defaultdict(list)

for runlength in runlengths:
	q12_data_crystal[runlength] = list(map(tuple, load_data_crystal[np.isclose(runlength, load_data_crystal[:, 0])][:, 1:]))
	q12_data_fluid[runlength] = list(map(tuple, load_data_fluid[np.isclose(runlength, load_data_fluid[:, 0])][:, 1:]))
# print(q12_data)


runlengths = [0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5]
q12_data_crystal = defaultdict(list)
q12_data_fluid = defaultdict(list)

for runlength in runlengths:
	q12_data_crystal[runlength] = list(map(tuple, load_data_crystal[np.isclose(runlength, load_data_crystal[:, 0])][:, 1:]))
	q12_data_fluid[runlength] = list(map(tuple, load_data_fluid[np.isclose(runlength, load_data_fluid[:, 0])][:, 1:]))
# print(q12_data)

syms = ['o', 'v', 's', 'd', 'P', '^', '<', '>', '*', 'X', 'p']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:gray', 'k', 'tab:cyan']

# plt.figure(figsize=(12, 8))
plts = []
firstp = []
all_pts_x = []
all_pts_y = []
runlengths2 = []
# p = [1.26302248e04, -3.46968772e03, 9.85952091e01, 4.41156894e01, 3.33529425, 3.40552817e-2]
# p = [248.35849589326912, -29.451926147991713, 4.569294136261736, 0.1575048765643605]
p = [5.4261432979711754, 0.030584777866394817]
# p = [510366735510802.75, -734322181238019.5, 451236182261943.44, -151872287247408.2, 29152064505147.207, -2690472739949.695, -65788668277.614235, 46800411613.2232, -5102158280.007087, 206078159.5902778, 4911142.138339991, -845353.5692760033, 33914.28488245272, -578.8087427835143, 9.629661618191685, 0.019772175151865808]
# deg = 15
# p = [1.1442851658104485e+24, -1.6976166601028082e+24, 9.603575029095748e+23, -2.1531804080802123e+23, -9.809086299920637e+21, 1.1424658819843342e+22, 8.551272422389411e+18, -6.181525684244158e+20, 1.6596059640688339e+19, 3.191816450905243e+19, -1.3829876712836209e+18, -2.305073699900103e+18, 6.246465941550659e+17, -6.466010366720427e+16, -233804685144105.16, 825711902314204.1, -96022171173713.55, 4645824266970.023, 9401145260.813192, -13833980818.788029, 791241334.4031234, -21797794.164463844, 308728.64630566916, -1916.9979699636142, 8.434369580424864, 0.028972152271995833]
deg = 1
for i in range(len(runlengths)):
	runlength = runlengths[i]
	if runlength < 0.45 * 2 * r and runlength > 0.05:
		print(runlength * 0.5 / r)
		cutoff = get_phi_ODT(runlength)
		phi_and_q12_crystal = q12_data_crystal[runlength]
		phi_and_q12_fluid = q12_data_fluid[runlength]
		j = i * 1
		if j > (len(colors) - 1):
			j -= len(colors)
		ps = np.asarray([p for p, _ in phi_and_q12_crystal] + [p for p, q in phi_and_q12_fluid if p > cutoff * 0 and q > 0 * 0.315])
		qs = np.asarray([q for _, q in phi_and_q12_crystal] + [q for p, q in phi_and_q12_fluid if p > cutoff * 0 and q > 0 * 0.315])
		sort = np.argsort(ps)
		ps = ps[sort]
		qs = qs[sort]


		runlengths2.extend([runlength] * len(ps))
		cutoff = (0.515 + 1 * 0.0635) + 1 * 0.0635 * np.tanh(2 * np.log(m * 0.5 * runlength / r + c))
		psis = (qs - 0.285) / 0.315
		phi_ODT = get_phi_ODT(runlength)
		psi_ODT = 1
		# phi_ODT = 0.515 + (0.645 - 0.515) * np.tanh(k_phi_ODT * np.log(runlength * 0.5 / r + 1.1) - k_phi_ODT * np.log(1.1))
		# # phi_ODT = (0.515 + 1 * 0.0635) + 1 * 0.0635 * np.tanh(2 * np.log(m_ODT * ell_div_D + c_ODT))
		# psi_ODT = psi_ODT_0 * (1 - np.tanh(k_psi_ODT * np.log(runlength * 0.5 / r + 1.1) - k_psi_ODT * np.log(1.1)))
		# print((phi_ODT, psi_ODT))
		# y = np.log(np.arctanh(psis)) + 100 * (0.5 * runlength / r)**4 * (1 - ps / 0.74) - 0.05 * ps / (1 - ps / 0.74)**0.5
		# y = np.log(np.arctanh(psis)) + 2 * np.log(1e-2 + (0.5 * runlength / r) / (1 + np.log(1 + (0.5 * runlength / r)**2))) * (1 - ps / 0.74) - 0.05 * ps / (1 - ps / 0.74)**0.5
		# y = get_psi_star(ps, runlength)
		ell_div_D = runlength * 0.5 / r
		ell_div_s = runlength / sigma
		cutoff = phi_ODT
		offset = 0.1
		ps2 = np.linspace(min(ps), max(ps), 1000)
		psinew2 = np.tanh(np.exp(m_psi * ps2 + c_psi - 2 * np.log(1e-2 + (ell_div_D) / (1 + np.log(1 + (ell_div_D)**2))) * (1 - ps2 / 0.74) + 0.05 * ps2 / (1 - ps2 / 0.74)**0.5))
		y = np.asarray([0.0] * len(ps2))
		y[ps2 > cutoff] = psinew2[ps2 > cutoff] * 1
		y3 = get_psi_star(ps2, runlength)
		# print(ps2[y > 0][0])
		# plt.scatter(ps, (qs - 0.285) / 0.315, c=colors[j], marker=syms[j])
		x1 = 2.67 - (ps / phi_ODT + 2.4 * phi_ODT)
		y1 = (1 - (psis - psi_ODT) / (1 - psi_ODT)) * (runlength * 0.5 / r)**-0.2
		x2 = -x1 + 0.25
		y2 = 1 - y1
		# y2 = tanh(k x)
		all_pts_x.extend(list(x2[y2 <= 1]))
		all_pts_y.extend(list(np.arcsinh(np.arcsinh(np.arctanh(y2[y2 <= 1])))))
		# plt.scatter(ps, psis, facecolors='none', edgecolors=colors[j], marker=syms[j], s=100, lw=2.5, zorder=100)
		P = sum([p[i] * x2**(deg - i) for i in range(deg + 1)])
		# plt.plot(x2, psi_ODT + (1 - psi_ODT) * (1 - (runlength * 0.5 / r)**0.2 * (1 - np.tanh(np.sinh(np.sinh(P))))))
		# plt.plot(ps2, y, color=colors[j])
		# plt.plot(ps2, y3, color=colors[j], ls='--')
		# plt.scatter(x2, np.arcsinh(np.arcsinh(np.arctanh(y2))), facecolors='none', edgecolors=colors[j], marker=syms[j], s=100, lw=2.5, zorder=100)
		# plt.scatter(phi_ODT, psi_ODT, marker='.', color=colors[j])
		# firstp.append([(min(ps), (qs[np.argmin(ps)] - 0.285) / 0.315), (ps[np.argmin(ps) - 1], (qs[np.argmin(ps) - 1] - 0.285) / 0.315)])
		# plt.scatter([p for p, _ in phi_and_q12_fluid], [q for _, q in phi_and_q12_fluid], c='white', edgecolors=colors[j], marker=syms[j])
		# plt.scatter(ps, y)
		# plt.scatter(ps, np.exp(ps), color='k')

all_pts_y = np.asarray(all_pts_y)
all_pts_x = np.asarray(all_pts_x)
sort = np.argsort(all_pts_x)

# print(all_pts_y)

# all_pts_x = all_pts_x[all_pts_y >= 0]
# all_pts_y = all_pts_y[all_pts_y >= 0]

# plt.show()

deg = 1
p = np.polyfit(all_pts_x, all_pts_y, deg)
print(list(p))

# pred_y = p[0] * all_pts_x**5 + p[1] * all_pts_x**4 + p[2] * all_pts_x**3 + p[3] * all_pts_x**2 + p[4] * all_pts_x**1 + p[5]
pred_y = sum([p[i] * all_pts_x**(deg - i) for i in range(deg + 1)])

# plt.figure()
# plt.scatter(all_pts_x, all_pts_y)
# plt.plot(all_pts_x[sort], pred_y[sort])
# plt.show()
# assert False


# runlengths = [500.0, 250.0, 166.5, 122.5, 100.0, 83.5, 71.5, 62.5, 55.5, 50.0, 45.0, 40.0, 37.5, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 7.5, 5.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.05][::-1]
# q12_data_fluid = defaultdict(list)

# for runlength in runlengths:
# 	q12_data_fluid[runlength] = list(map(tuple, load_data_fluid[np.isclose(runlength, load_data_fluid[:, 0])][:, 1:]))

# lastp = []
# # m_psi, c_psi = (19.031119290073068, -13.234065182192372)
# # m_psi, c_psi = (18.737454952901963, -13.012403139706487)
# m_psi, c_psi = (18.821730673114374, -13.08211494066777)
# # plt.figure()
# lastj = 0
# for i in range(len(runlengths[1:])):
# 	runlength = runlengths[1:][i]
# 	if runlength < 0.55 and runlength > 0.05:

# 		phi_and_q12_fluid = q12_data_fluid[runlength]
# 		j = i * 1
# 		if j > (len(colors) - 1):
# 			j = i % len(colors)
# 		phi = np.linspace(0.05, 0.6, 10000)
# 		# P_act_fit = (zeta * U / (np.pi * 4 * r**2)) * phi * (0.5 * runlength / r) / (1 + (1 - np.exp(-2**(7.0/6.0) * 0.5 * runlength / r)) * phi / (1 - phi / 0.645))
# 		# P_int_fit = (zeta * U / (np.pi * 4 * r**2)) * 6 * 2**(-7.0/6.0) * phi**2 / np.sqrt(1 - phi / 0.645)
# 		# P = P_act_fit + P_int_fit
# 		# possible = phi[:-1][np.diff(P) / np.diff(P_int_fit) < 0]
# 		phi = np.linspace(0.025, 0.74, 1000)
# 		# psinew2 = np.tanh(np.exp(rbf(phi, [runlength] * len(phi))))
# 		# psinew2 = np.tanh(np.exp(m_psi * phi + c_psi - 2 * np.log(1e-2 + (0.5 * runlength / r) / (1 + np.log(1 + (0.5 * runlength / r)**2))) * (1 - phi / 0.74) + 0.05 * phi / (1 - phi / 0.74)**0.5))
# 		# cutoff = (0.515 + 1 * 0.0635) + 1 * 0.0635 * np.tanh(2 * np.log(m * 0.5 * runlength / r + c))
# 		# print(cutoff)
# 		# # psinew2 = 0.0 + 1.0 * np.tanh(1 * phi / cutoff)
# 		# # psinew2 = 0.5 + 0.5 * np.tanh(10 * (phinew - offset))
# 		# # psinew2 = np.tanh(20 * (phinew - cutoff + offset))
# 		# y = np.asarray([0.0] * len(phi))
# 		# y[phi > cutoff] = psinew2[phi > cutoff]
# 		y = get_psi_star(phi, runlength)
# 		# print(y)
# 		# y = get_psi(runlength, phi)
# 		# y[phi > phi_star] = 1 / (0.74 - phi_star) * (phi[phi > phi_star] - phi_star)
# 		# plts.append(plt.plot(phi, y, color=colors[j], ls='-', lw=3.5)[0])
# 		ps = np.asarray([p for p, _ in phi_and_q12_fluid])
# 		qs = np.asarray([q for _, q in phi_and_q12_fluid])
# 		psis = (qs - 0.285) / 0.315
# 		# lastp.append((max(ps), psis[np.argmax(ps)])
# 		plt.scatter(ps, psis, facecolors='none', edgecolors=colors[j], marker=syms[j], s=100, lw=2.5, zorder=100)
# 		cutoff = get_phi_ODT(runlength)
# 		plt.scatter(cutoff, 0, marker='.', color=colors[j])
# 		lastj = j

# plt.show()
# assert False


# plt.xlabel(r'$\phi$', fontsize=14)
# plt.ylabel(r'$\psi^*$', fontsize=18, rotation=0, labelpad=15)

# fact = 2.0**(1.0/6.0)
# rls = [5.0 * fact, 50.0 * fact, 500.0 * fact]
# # rls = [50.0]
# for i in range(len(rls)):
# 	rl = rls[i]
# 	phi = np.linspace(0.025, 0.74, 1000)
# 	# psinew2 = np.tanh(np.exp(rbf(phi, [runlength] * len(phi))))
# 	# psinew2 = np.tanh(np.exp(m_psi * phi + c_psi - 2 * np.log(1e-2 + (0.5 * rl / r) / (1 + np.log(1 + (0.5 * rl / r)**2))) * (1 - phi / 0.74) + 0.05 * phi / (1 - phi / 0.74)**0.5))
# 	# cutoff = (0.515 + 1 * 0.0635) + 1 * 0.0635 * np.tanh(2 * np.log(m * 0.5 * rl / r + c))
# 	# print(cutoff)
# 	# y = np.asarray([0.0] * len(phi))
# 	# y[phi > cutoff] = psinew2[phi > cutoff]
# 	# print(y)
# 	y = get_psi_star(phi, rl)
# 	plts.append(plt.plot(phi, y, color=colors[lastj + i + 1], ls='-', lw=3.5)[0])
# plt.legend(plts, list(np.asarray(runlengths[1:])[np.asarray(runlengths[1:]) < 0.55] / fact) + list(np.asarray(rls) / fact), title=r'$\ell_0 / D$', title_fontsize=14)

# ax = plt.gca()
# ax.tick_params(width=3, length=10, which='major')
# ax.tick_params(width=3, length=6, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(3.5)
# # plt.scatter(phi_stars, [0] * len(phi_stars))
# plt.show()
# # plt.savefig('psi_vs_phi.svg', format='svg', dpi=1200)
# assert False


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

runlengths_plt = []
plts = []
p = [0.01109925, 0.04995078, 0.16538631, 0.41931119]
for i in range(len(runlengths_and_phis)):
	runlength, _ = runlengths_and_phis[i]
	if runlength < 0.45 * 0.5 / r:
	# if runlength == 0.4:
		runlengths_plt.append(runlength)
		# x = 0.999999999
		cutoff = 0.0
		phi_data = data[runlength][:, 0]
		rho = phi / v
		nondim_factor = zeta * U * 2 * r / v
		Pc = data[runlength][:, 1] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))
		Pa = data[runlength][:, 2] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))

		kBT = zeta * U * runlength / 6
		# kBT = zeta * U * r / 3
		Ea = zeta * U * 2 * r
		E_div_v = Ea / v
		ell_div_D = runlength * 0.5 / r
		ell_div_s = runlength / sigma


		# phi = np.linspace(1e-5, 0.74 - 1e-5, 1000)
		# phi = np.linspace(1e-5, 0.645 - 1e-5, 1000)
		phi = np.linspace(min(phi_data), max(phi_data), 1000)

		n = [0, 1, 2, 3, 4, 5, 6, 7, 8]
		cn = [1, 1.1649e-1, 2.217e-2, 1.84e-3, 3.373e-5, -1.117e-5, -8.914e-6, -9.469e-7, -4.356e-7]
		# cn = np.asarray([1.0, 0.3298, 0.08867, 0.01472, 0.0005396, -0.0003574, -0.0005705, -0.0001212, -0.00011151])
		# print(cn)

		# First the fluid EOS (from Oscar):
		# Pa_fit = (zeta * U / (np.pi * 4 * r**2)) * (ell_div_D) * rho * v / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * rho * v / (1 - rho / rho_max_fit))
		psi_star = get_psi_star(phi, runlength)
		phi_max = get_phi_max(psi_star, runlength)
		psi_star_data = get_psi_star(phi_data, runlength)
		phi_max_data = get_phi_max(psi_star_data, runlength)
		beta = get_beta(psi_star, runlength)
		# beta = 1.1
		# phi_max = 0.645
		A = 3.576 * (1 + np.tanh(0.287 * np.log(ell_div_s) - 1.611)) # * (1 + psi_star * 0.025 / 0.6 * (psi_star - 1.0) / ((0.5 * runlength / r)**2 * np.exp(-0.5 * runlength / r )))
		B = 1.049
		C = 0.554
		ell_0c = 18.8
		rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2
		# C = 0.7
		# Pa_fit = phi * ell_div_D * np.exp(-A * phi**B / (1 - phi / phi_max)**C)
		Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6
		# Pa_fit = np.exp(-A * phi**B / (1 - phi / phi_max)**C + np.log(phi) + np.log(ell_div_D))
		# A_psi_fact = -(0.5 * runlength / r)**2 * np.exp(-0.5 * runlength / r ) * (1 + np.log(Pa / (phi_data * ell_div_D)) * (1 - phi_data / phi_max_data)**C / phi_data**B / 3.576 / (1 + np.tanh(0.287 * np.log(ell_div_s) - 1.611))) / psi_star_data
		# A_psi_fact2 = 0.025 / 0.6 * (psi_star - 1.0)

		# beta_fit = np.asarray([0.0] * len(psi_star))
		# beta_fit[psi_star > 0] = (0.1 + 0.3 * (0.5 + 0.5 * np.tanh(10 * ((runlength - 19.75) / r * 0.5)**1))) * np.tanh(5 * psi_star[psi_star > 0] * 1)
		Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))
		print(kBT)
		# Pc_HS_fit = 24 * kBT / (np.pi * 8 * r**3) * phi**2 * np.sum([cn[i] * (4 * phi)**n[i] for i in range(len(n))]) / (1 - phi / phi_max)**0.76 / E_div_v
		Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor

		# phi_max = phi_data / (1 - (-A * phi_data**B / np.log(Pa / (phi_data * ell_div_D)))**(1/C))
		rl_x = np.log(runlength * 0.5 / r)
		x = p[3] + p[2] * rl_x + p[1] * rl_x**2 + p[0] * rl_x**3
		x = max([x, 0.0])
		x = min([x, 1.0])
		x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
		# x = 1
		Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
		# D0 = (0.4 * 0.5 / r) / np.log(1 / (1 - 0.0003))**(1/2)
		# x = 1 - np.exp(-((runlength * 0.5 / r / D0))**2)
		# x = 0
		# plts.append(plt.scatter(psi_star_data, A_psi_fact))
		# plt.plot([0.4, 1.0], [-0.025, 0.0])
		# plt.plot(psi_star, A_psi_fact2)
		
		ax1.plot(phi, Pc_fit, ls='--', lw=3.5, color=colors[i])
		ax1.scatter(phi_data, Pc, facecolors='none', edgecolors=colors[i], lw=2.5, marker=syms[i], s=100)
		ax2.plot(phi, Pa_fit, ls='--', lw=3.5, color=colors[i])
		ax2.scatter(phi_data, Pa, facecolors='none', edgecolors=colors[i], lw=2.5, marker=syms[i], s=100)
		ax3.plot(phi, Pa_fit + Pc_fit, ls='--', lw=3.5, color=colors[i])
		ax3.scatter(phi_data, Pa + Pc, facecolors='none', edgecolors=colors[i], lw=2.5, marker=syms[i], s=100)

		# plts.append(plt.scatter(psi_star, phi_max))
		# if runlength == 0.3:
		# 	deg2 = 6
		# 	coeffs = np.polyfit(psi_star, phi_max, deg2)
		# 	print(list(coeffs))
		# 	phi_max_fit = coeffs[0] * psi_star**2 + coeffs[1] * psi_star + coeffs[2]
		# # 	# print(phi_max_fit)
		# # 	# print(phi_data)
		# 	# plt.plot(psi_star, sum([coeffs[i] * psi_star**(deg2 - i) for i in range(deg2 + 1)]))
		# # 	# plt.plot(psi_star, get_phi_max(phi_data, runlength))
		# # # plt.scatter(psi_star, get_phi_max(phi_data, runlength))
ax1.set_xlabel(r'$\phi$', fontsize=18)
ax2.set_xlabel(r'$\phi$', fontsize=18)
ax3.set_xlabel(r'$\phi$', fontsize=18)
# plt.ylabel(r'$\phi_{\rm max}$', fontsize=14, rotation=0)
ax1.set_ylabel(r'$\overline{p}_C$', fontsize=18, rotation=0, labelpad=10)
ax2.set_ylabel(r'$\overline{p}_{\rm act}$', fontsize=18, rotation=0, labelpad=10)
ax3.set_ylabel(r'$\overline{\mathcal{P}}^{\rm bulk}$', fontsize=18, rotation=0, labelpad=10)
# plt.legend(plts, runlengths_plt, title=r'$\ell_0 / \sigma$')
for ax in [ax1, ax2, ax3]:
	ax.tick_params(width=3, length=10, which='major', labelsize=16)
	ax.tick_params(width=3, length=6, which='minor')
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(3.5)
	ax.set_xticks([0.6, 0.64, 0.68, 0.72])
ax2.set_yticks([0.002, 0.006, 0.01, 0.014])
# ax3.set_yticks([0, 1, 2])
# ax1.set_yticks([0, 1, 2])
plt.tight_layout()
plt.show()
# plt.savefig('p_vs_phi_SI.svg', format='svg', dpi=1200)
# plt.close()

# assert False

cf_act = np.loadtxt('cf_act.txt')
rl_act = cf_act[:, 0]
phifs_act = cf_act[:, 1]
phiss_act = cf_act[:, 2]

lg_act = np.loadtxt('lg_act.txt')
rl_mips_act = lg_act[:, 0]
phigs_act = lg_act[:, 1]
phils_act = lg_act[:, 2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

f = f'ABP_fluid_pressure/q12sABP_q12_0.3_N_55296_fluid_pressure_and_phi2.txt'
P1 = np.loadtxt(f) # timestep, interaction pressure, swim pressure, phi
df1 = pd.DataFrame()
df1["Active Pressure"] = P1[:, 2]
df1["Interaction Pressure"] = P1[:, 1]
df1["phi"] = P1[:, 3]

group_df1 = df1.groupby('phi', as_index=False)[["Active Pressure", "Interaction Pressure"]]#.mean()
group_df1_mean2 = group_df1.apply(lambda x: x.iloc[len(x) // 2:].mean())

# print(group_df1)


ell_0c = 18.8

runlengths_plt = []
plts = []
p = [0.01109925, 0.04995078, 0.16538631, 0.41931119]
runlength = 0.3
runlengths_plt.append(runlength)
# x = 0.999999999
cutoff = 0.0
phi_data = data[runlength][:, 0]
rho = phi / v
nondim_factor = zeta * U * 2 * r / v
Pc = data[runlength][:, 1] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))
Pa = data[runlength][:, 2] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))

phif_data = np.asarray(group_df1_mean2['phi']) #/ 8
Pcf_data = np.asarray(group_df1_mean2['Interaction Pressure']) / nondim_factor
Paf_data = np.asarray(group_df1_mean2['Active Pressure']) / nondim_factor

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


phi = np.linspace(0.45, 0.7, 1000)

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
psi_star_data = get_psi_star(phi_data, runlength)
phi_max_data = get_phi_max(psi_star_data, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax1.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax1.plot(phi, Pc_fit, color='k', lw=2.5, ls='--') # ls='-', lw=3.5, color='k')
ax1.plot(phi, Pa_fit, color='k', lw=2.5, ls=':') # ls='-', lw=3.5, color='k')
ax1.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# ax1.scatter(phi_data, Pa + Pc, facecolors='none', edgecolors='k', lw=2)
# ax1.scatter(phif_data[phif_data < phi[psi_star > 0][0]][::3], (Paf_data + Pcf_data)[phif_data < phi[psi_star > 0][0]][::3], facecolors='none', edgecolors='k', lw=2)
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)

# plt.savefig('p_vs_phi.svg', format='svg', dpi=1200)
# plt.show()




runlength = 19.
# x = 0.999999999
cutoff = 0.0
rho = phi / v
nondim_factor = zeta * U * 2 * r / v

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


# phi = np.linspace(0.3, 0.74, 1000)
phi = np.asarray(list(np.linspace(0.3, 0.6, 100)) + list(np.linspace(0.601, 0.67, 19)) + list(np.linspace(0.671, 0.74 - 1e-4, 1000)))

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
x[np.isnan(x)] = 1

Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax2.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax2.plot(phi, Pc_fit, color='k', lw=2.5, ls='--') # ls='-', lw=3.5, color='k')
ax2.plot(phi, Pa_fit, color='k', lw=2.5, ls=':') # ls='-', lw=3.5, color='k')
ax2.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)


# plt.legend(plts, runlengths_plt, title=r'$\ell_0 / \sigma$')



runlength = 25
# x = 0.999999999
cutoff = 0.0
rho = phi / v
nondim_factor = zeta * U * 2 * r / v

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


phi = np.linspace(0.15, 0.74, 500)

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))

Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax3.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax3.plot(phi, Pc_fit, color='k', lw=2.5, ls='--') # ls='-', lw=3.5, color='k')
ax3.plot(phi, Pa_fit, color='k', lw=2.5, ls=':') # ls='-', lw=3.5, color='k')
ax3.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)

# plt.legend(plts, runlengths_plt, title=r'$\ell_0 / \sigma$')
for ax in [ax1, ax2, ax3]:
	ax.set_xlabel(r'$\phi$', fontsize=18)
	ax.tick_params(width=3, length=10, which='major', labelsize=16)
	ax.tick_params(width=3, length=6, which='minor')
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

ax1.set_yticks([0.0, 0.2, 0.4])
ax1.set_xticks([0.5, 0.6, 0.7])
ax3.set_yticks([0, 2, 4])

plt.tight_layout()
plt.show()
# plt.savefig('p_vs_phi_SI2.svg', format='svg', dpi=1200)
# plt.close()



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

f = f'ABP_fluid_pressure/q12sABP_q12_0.3_N_55296_fluid_pressure_and_phi2.txt'
P1 = np.loadtxt(f) # timestep, interaction pressure, swim pressure, phi
df1 = pd.DataFrame()
df1["Active Pressure"] = P1[:, 2]
df1["Interaction Pressure"] = P1[:, 1]
df1["phi"] = P1[:, 3]

group_df1 = df1.groupby('phi', as_index=False)[["Active Pressure", "Interaction Pressure"]]#.mean()
group_df1_mean2 = group_df1.apply(lambda x: x.iloc[len(x) // 2:].mean())

# print(group_df1)


ell_0c = 18.8

runlengths_plt = []
plts = []
p = [0.01109925, 0.04995078, 0.16538631, 0.41931119]
runlength = 0.3
runlengths_plt.append(runlength)
# x = 0.999999999
cutoff = 0.0
phi_data = data[runlength][:, 0]
rho = phi / v
nondim_factor = zeta * U * 2 * r / v
Pc = data[runlength][:, 1] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))
Pa = data[runlength][:, 2] / nondim_factor #/ (zeta * U / (np.pi * 4 * r**2))

phif_data = np.asarray(group_df1_mean2['phi']) #/ 8
Pcf_data = np.asarray(group_df1_mean2['Interaction Pressure']) / nondim_factor
Paf_data = np.asarray(group_df1_mean2['Active Pressure']) / nondim_factor

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


phi = np.linspace(0.45, 0.7, 1000)

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
psi_star_data = get_psi_star(phi_data, runlength)
phi_max_data = get_phi_max(psi_star_data, runlength)
beta = get_beta(psi_star, runlength)

print(get_phi_ODT(runlength))

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
# print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax1.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax1.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
ax1.scatter(phi_data[phi_data <= max(phi)], (Pa + Pc)[phi_data <= max(phi)], facecolors='none', edgecolors='k', lw=2, marker='s', s=100)
ax1.scatter(phif_data[(phif_data < phi[psi_star > 0][0]) & (phif_data >= min(phi))][::3], (Paf_data + Pcf_data)[(phif_data < phi[psi_star > 0][0]) & (phif_data >= min(phi))][::3], facecolors='none', edgecolors='k', lw=2, s=100)
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)

# plt.savefig('p_vs_phi.svg', format='svg', dpi=1200)
# plt.show()




runlength = 19.
# x = 0.999999999
cutoff = 0.0
rho = phi / v
nondim_factor = zeta * U * 2 * r / v

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


# phi = np.linspace(0.3, 0.74, 1000)
phi = np.asarray(list(np.linspace(0.3, 0.6, 100)) + list(np.linspace(0.601, 0.67, 19)) + list(np.linspace(0.671, 0.74 - 1e-4, 1000)))

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
x[np.isnan(x)] = 1
print(get_phi_ODT(runlength))
Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
# print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax2.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax2.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
ax2.plot([phi[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], phi[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]])
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)

# ax2.set_yscale("functionlog", functions=[lambda x: x-0.9999 * min(Pa_fit + Pc_fit), lambda x: x+0.9999 * min(Pa_fit + Pc_fit)])
ax2.set_yscale('log')
# plt.legend(plts, runlengths_plt, title=r'$\ell_0 / \sigma$')



runlength = 25
# x = 0.999999999
cutoff = 0.0
rho = phi / v
nondim_factor = zeta * U * 2 * r / v

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


phi = np.linspace(0.15, 0.74, 500)

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
print(get_phi_ODT(runlength))
Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
# print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52
ax3.plot(phi, Pa_fit + Pc_fit, color='k', lw=2.5) # ls='-', lw=3.5, color='k')
ax3.plot([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
ax3.plot([phi[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], phi[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]])
# plt.scatter(phif_data, (Paf_data + Pcf_data), facecolors='none', edgecolors='k', lw=2)
# ax3.set_yscale("functionlog", functions=[lambda x: x-0.9999 * min(Pa_fit + Pc_fit), lambda x: x+0.9999 * min(Pa_fit + Pc_fit)])
ax3.set_yscale('log')
# plt.legend(plts, runlengths_plt, title=r'$\ell_0 / \sigma$')
for ax in [ax1, ax2, ax3]:
	ax.set_xlabel(r'$\phi$', fontsize=18)
	ax.tick_params(width=3, length=10, which='major', labelsize=16)
	ax.tick_params(width=3, length=6, which='minor', labelsize=16)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

ax1.set_xticks([0.5, 0.6, 0.7])
ax1.set_yticks([0.1, 0.3, 0.5])
ax2.set_yticks([0.6, 0.7, 0.8, 0.9, 2, 3], minor=True)
ax2.set_yticks([1])
ax2.set_yticklabels(['', '', '', '', 2, 3], minor=True)
ax2.set_yticklabels([1])
ax3.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6], minor=True)
ax3.set_yticks([1])
ax3.set_yticklabels(['', '', '', '', '', '', 3, '', 5, ''], minor=True)
ax3.set_yticklabels([1])
# ax3.set_yticks([1, 3, 5])
# set_xticklabels

ax1.set_ylabel(r'$\overline{\mathcal{P}}^{\rm bulk}$', fontsize=18)



plt.tight_layout()
plt.show()
# plt.savefig('p_vs_phi_main.svg', format='svg', dpi=1200)
# plt.close()



plt.figure(figsize=(5 * 0.5, 4 * 0.5))

runlength = 19.
# x = 0.999999999
cutoff = 0.0
rho = phi / v
nondim_factor = zeta * U * 2 * r / v

kBT = zeta * U * runlength / 6
# kBT = zeta * U * r / 3
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma


# phi = np.linspace(0.3, 0.74, 1000)
phi = np.asarray(list(np.linspace(0.415, 0.543, 1000)))

psi_star = get_psi_star(phi, runlength)
phi_max = get_phi_max(psi_star, runlength)
beta = get_beta(psi_star, runlength)

rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

Pa_fit = (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6


Pc_act_fit = 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2))

Pc_HS_fit = kBT * (phi / v) / (1 - (phi / phi_max))**1 / nondim_factor


x = (1 - np.tanh(10 * psi_star)) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) + np.tanh(10 * psi_star) * np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1))
x[np.isnan(x)] = 1
print(get_phi_ODT(runlength))
Pc_fit = x * Pc_act_fit + (1 - x) * Pc_HS_fit
# print([phi[np.argmin(np.abs(phi - phifs_act[np.argmin(np.abs(runlength - rl_act))]))], phi[np.argmin(np.abs(phi - phiss_act[np.argmin(np.abs(runlength - rl_act))]))]])
# phif_data /= 6.52

plt.plot(phi, Pa_fit + Pc_fit, lw=2.5)
plt.plot([phi[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], phi[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]], [(Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phigs_act[np.argmin(np.abs(runlength - rl_mips_act))]))], (Pa_fit + Pc_fit)[np.argmin(np.abs(phi - phils_act[np.argmin(np.abs(runlength - rl_mips_act))]))]])

ax = plt.gca()
ax.set_xlabel(r'$\phi$', fontsize=18)
ax.tick_params(width=3, length=10, which='major', labelsize=16)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.5)

ax.set_xticks([0.4, 0.5])
ax.set_yticks([0.6725, 0.6775])

ax.set_ylabel(r'$\overline{\mathcal{P}}^{\rm bulk}$', fontsize=18)

# plt.show()
plt.savefig('p_vs_phi_inset.svg', format='svg', dpi=1200)
plt.close()

