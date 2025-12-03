import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline



N = 55296
runlength = 0.3
conc0 = 0.74
phif = 0.63
ntau_equilibrate = 200

U = 1.0 # swim speed
sigma = 1.0 # LJ diameter
tau = sigma / U #convection time
Ucompress = 0.01 * (U)
time_step = 5e-5 * tau
nsteps_equilibrate = int(ntau_equilibrate / time_step)
r = (2.0**(1. / 6.) * sigma / 2)
v = (4.0 / 3.0) * np.pi * r**3.0

zeta = 1.0 #translational drag

LR = runlength * sigma #the runlength provided in the input is in units of sigma
tauR = LR / U #reorientation time

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
					# if len(E_r) < len(phi):
					# 	integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])])))
					# else:
					# 	integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					if len(E_r) < len(phi):
						integral = np.abs(np.sum(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					else:
						integral = integral = np.abs(np.sum(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	return phi1, phi2, P_coex

def get_coex3(P, phi, psi, spinodall, spinodalh, E_r):
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
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])])))
					else:
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					# if len(E_r) < len(phi):
					# 	integral = np.abs(np.sum(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					# else:
					# 	integral = integral = np.abs(np.sum(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	return phi1, phi2, P_coex

def get_coex4(P, phi, psi, spinodall1, spinodall2, spinodalh1, spinodalh2, E_r):
	sort = np.argsort(E_r)
	P = P[sort]
	phi = phi[sort]
	psi = psi[sort]
	E_r = E_r[sort]
	min_P = P[np.argmin(np.abs(phi - spinodalh2))]
	max_P = P[np.argmin(np.abs(phi - spinodall2))]
	phi_P_cp = 0.74
	phi1, phi2, P_coex = None, None, None
	if len(phi[((P - min_P) > 0) & (phi < spinodall2) & (phi > spinodalh1)]) > 0:
		min_phi = min(phi[((P - min_P) > 0) & (phi < spinodall2) & (phi > spinodalh1)])
		if len(phi[((P - max_P) < 0) & (phi > spinodalh2)]) == 0:
			spinodall -= 1e-4
			max_P = P[np.argmin(np.abs(phi - spinodall))]
		max_phi = max(phi[((P - max_P) < 0) & (phi > spinodalh2)])
		phi_init = phi[(phi > min_phi) & (phi < spinodall2) & (phi > spinodalh1)]
		P_init = P[(phi > min_phi) & (phi < spinodall2) & (phi > spinodalh1)]
		P_init = P_init[phi_init < phi_P_cp]
		phi_init = phi_init[phi_init < phi_P_cp]
		P_fin = P[(phi < max_phi) & (phi > spinodalh2)]
		phi_fin = phi[(phi < max_phi) & (phi > spinodalh2)][np.argmin(np.abs(np.subtract.outer(P_fin, P_init)), axis=0)]
		if len(P_fin) > 0 and len(P_init) > 0 and len(phi_fin) > 0:
			P_relevant = P[(phi < max_phi) & (phi > min_phi)]
			phi_relevant = phi[(phi < max_phi) & (phi > min_phi)]
			best = 100000000000000
			for i in range(len(phi_init)):
				if np.abs(phi_init[i] - phi_fin[i]) > 0.001:
					if len(E_r) < len(phi):
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])])))
					else:
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * np.diff(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					# if len(E_r) < len(phi):
					# 	integral = np.abs(np.sum(E_r[(phi[-1] > phi_init[i]) & (phi[-1] < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					# else:
					# 	integral = integral = np.abs(np.sum(E_r[(phi > phi_init[i]) & (phi < phi_fin[i])][:-1] * np.diff(P[(phi > phi_init[i]) & (phi < phi_fin[i])])))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	return phi1, phi2, P_coex

def get_coex2(P, phi, psi, spinodall, spinodalh, E_rr, E_rp):
	# print(np.count_nonzero(np.isnan(P)))
	# print(np.count_nonzero(np.isnan(phi)))
	# print(np.count_nonzero(np.isnan(psi)))
	# print(np.count_nonzero(np.isnan(E_rr)))
	# print(np.count_nonzero(np.isnan(E_rp)))
	# if len(phi) == len(E_rr):
	# 	print(phi[np.isnan(E_rr)])
	# else:
	# 	print(phi[1:][np.isnan(E_rr)])
	# # assert False
	# print()
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
			best = 1e20
			# print(len(E_rr))
			# print(len(E_rp))
			# print(len(P))
			# print(len(phi))
			# print(len(psi))
			for i in range(len(phi_init)):
				if np.abs(phi_init[i] - phi_fin[i]) > 0.001:
					if len(E_rr) < len(phi):
						if len(E_rr[(phi[1:] > phi_init[i]) & (phi[1:] < phi_fin[i])]) < len(phi[(phi[:] > phi_init[i]) & (phi[:] < phi_fin[i])]):
							integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * (E_rr[(phi[1:] > phi_init[i]) & (phi[1:] < phi_fin[i])] * np.diff(phi[(phi > phi_init[i]) & (phi < phi_fin[i])])[:] + E_rp[(phi[1:] > phi_init[i]) & (phi[1:] < phi_fin[i])] * np.diff((psi)[(phi > phi_init[i]) & (phi < phi_fin[i])])[:])))
						else:
							integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * (E_rr[(phi[1:] > phi_init[i]) & (phi[1:] < phi_fin[i])][1:] * np.diff(phi[(phi > phi_init[i]) & (phi < phi_fin[i])])[:] + E_rp[(phi[1:] > phi_init[i]) & (phi[1:] < phi_fin[i])][1:] * np.diff((psi)[(phi > phi_init[i]) & (phi < phi_fin[i])])[:])))
					else:
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[:-1] * (E_rr[(phi > phi_init[i]) & (phi < phi_fin[i])][1:] * np.diff(phi[(phi > phi_init[i]) & (phi < phi_fin[i])])[:] + E_rp[(phi > phi_init[i]) & (phi < phi_fin[i])][1:] * np.diff((psi)[(phi > phi_init[i]) & (phi < phi_fin[i])])[:])))
					# print(np.count_nonzero(np.isnan((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])]))))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	if phi1 is not None and phi2 is not None:
		return min([phi1, phi2]), max([phi1, phi2]), P_coex
	else:
		return phi1, phi2, P_coex


m_ODT, c_ODT = (10.559847151050855, 0.08194286572998821)
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
    phi_max = 0.645 * np.ones(len(psi))
    phi_max[psi > 1e-5] = 0.74
    k = 15.847584758475847 #- 10 * (1 + np.tanh(10 * (0.5 * runlength / r - 20.)**1)) / 2 - 3.1 * (1 + np.tanh(10 * (0.5 * runlength / r - 20.0)**3)) / 2 #- 0.5 * (1 + np.tanh(10 * (0.5 * runlength / r - 22.0)**3)) / 2
    # k = 5 + np.exp(0.5 * runlength / r)
    # k = 2
    x = np.exp(-(0.5 * runlength / r / 22)**10)
    x = 1
    # b = (1 + (1 + np.tanh(10 * (0.5 * runlength / r - 19.)**1)) / 2)
    b = 3
    phi_max = 0.645 + (0.74 - 0.645) * (np.tanh(k * psi) * x + (1 - x) * np.log(1 + 1 * psi / 3 + 2 * psi**0.5 / 3) / np.log(2))
    # phi_max = 0.645 + (0.74 - 0.645) * (np.tanh(k * (psi**b + 2 * psi)/3) * x + (1 - x) * np.log(1 + 1 * psi / 3 + 2 * psi**0.5 / 3) / np.log(2))
    # phi_max = 0.645 + (0.74 - 0.645) * (np.tanh(k * np.log(psi**b + psi + 1) / np.log(3) * x + (1 - x) * np.log(1 + 1 * psi / 3 + 2 * psi**0.5 / 3) / np.log(2)))
    return phi_max

delta_sigma2 = 1e-4

def delta(x):
	return np.exp(-0.5 * x**2 / delta_sigma2) / (np.sqrt(2 * np.pi * delta_sigma2))

def ddelta_dx(x):
	return delta(x) * x / delta_sigma2

def d2delta_dx2(x):
	return (x**2 / delta_sigma2 - 1) * delta(x) / delta_sigma2

def get_kappa(rho, R, kBT):

	R0 = 1
	R1 = R
	R2 = 4 * np.pi * R**2
	R3 = 4 * np.pi / 3 * R**3
	xi0 = rho * R0
	xi1 = rho * R1
	xi2 = rho * R2
	xi3 = rho * R3
	chi0 = 1 / (1 - xi3)
	chi1 = xi2 / (1 - xi3)**2
	chi2 = xi1 / (1 - xi3)**2 + (xi2**2) / (4 * np.pi) / (1 - xi3)**3
	chi3 = xi0 / (1 - xi3)**2 + (2 * xi1 * xi2) / (1 - xi3)**3 + (xi2**3) / (4 * np.pi) / (1 - xi3)**4

	rmax = 10 * R
	rmin = 0.01 * R
	r = R * np.logspace(-2, 0.5, 1000)

	delta_V = 4 * np.pi * (R - r)**3 / 3
	delta_V[(R - r) < 0] = 0
	delta_S = 8 * np.pi * (R - r)**3 * delta(r - R) / 3
	delta_R = (R - r)**3 / 3 * ddelta_dx(r - R) - (r - R)**2 * delta(r - R)
	delta_Theta = (-(R - r)**2 - 4 * (R - r)**3 / 3 / r) * ddelta_dx(R - r) + (R - r)**3 / 3 * d2delta_dx2(R - r)

	c_r = 1 * (np.outer(chi3, delta_V) + np.outer(chi2, delta_S) + np.outer(chi1, delta_R) + np.outer(chi0, delta_Theta))
	kappa = 4 * np.pi * kBT * np.sum(c_r[:, :-1] * r[:-1]**4 * np.diff(r), axis=1) / 6 # + kBT * psi**rho

	return kappa - min(kappa) + 0.01



p_x = [0.01109925, 0.04995078, 0.16538631, 0.41931119]
m_diff, c_diff = (1.7467557835164393, 2.8901008821134653)
def get_Ps(phi, psi_star, phi_max, runlength):
	ell_0c = 18.8
	# ell_0c = 20.25
	kBT = zeta * U * runlength / 6
	Ea = zeta * U * 2 * r
	E_div_v = Ea / v
	ell_div_D = runlength * 0.5 / r
	ell_div_s = runlength / sigma
	nondim_factor = zeta * U * 2 * r / v
	rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2

	Pa = nondim_factor * (1 + np.tanh(10 * psi_star) * (1 - 29 * rl_func + 30 * (1 + np.tanh(1 * (0.5 * runlength / r - 3 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi_star) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi_star) * rl_func)) / 6
	# Pa = nondim_factor * (1 + np.tanh(10 * psi_star) * (1 - (1 + np.tanh(1 * (0.5 * runlength / r - 20 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi / (1 - phi / phi_max)**(1)) / 6


	beta = 0.5 + np.tanh(10 * psi_star) * ( (0.1 - 0.5) * (1 + np.tanh(5 * (0.5 * runlength / r - ell_0c * 0.5 / r)**1)) / 2 + (0.275 - 0.1) * (1 + np.tanh(0.01 * (0.5 * runlength / r - 50 * 0.5 / r)**1)) / 2 ) # + np.tanh(psi) * (1e-4 - 0.1) * (1 + np.tanh(2 * (0.5 * runlength / r - 19.0 * 0.5 / r))) / 2 # + (0.05 - 0.1) * (0.5 + np.tanh(10 * psi) * np.tanh(10 * (0.5 * runlength / r - 21.0)))
	# beta = 0.5 #+ (0.3 - 0.5) * np.tanh(psi * 0.5 * runlength / r)
	Pc_act = nondim_factor * 2**(-7/6) * phi**(2 + 0 * np.tanh(10 * psi_star) * rl_func) / (1 - 1 * (phi / phi_max)**1 - 0. * (np.sinh(phi / phi_max - 0.5) + np.sinh(0.5)) / np.sinh(0.5))**beta * (1 - 5/6 * np.tanh(10 * psi_star) * (1 - 0 * (1 + np.tanh(100 * (0.5 * runlength / r - 14 * 0.5 / r)**1)) / 2 - 0.9 * rl_func))

	# Pc_HS_fit = 24 * kBT / (np.pi * 8 * r**3) * phi**2 * np.sum([cn[i] * (4 * phi)**n[i] for i in range(len(n))]) / (1 - phi / phi_max)**0.76 / E_div_v
	Pc_HS = (kBT / v) * phi / (1 - (phi / phi_max))**1. # / nondim_factor

	# x = np.tanh(1 * np.log(runlength * 0.5 / r + 1))
	x = (1 - np.tanh(10 * psi_star)) * ( np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * (1 + np.tanh(10 * (0.5 * runlength / r - 18.8 * 0.5 / r)**1)) / 2**1 * (np.exp(1 * runlength**1 / (18.8 + 0)**1) - 1)) ) + np.tanh(10 * psi_star) * ( np.tanh(1 * np.log(runlength * 0.5 / r + 1) + 1 * rl_func**10 * (np.exp(runlength**10 / (ell_0c + 0)**10) - 1)) )
	# x = np.tanh(1 * np.log(runlength * 0.5 / r + 1) + runlength**2 / ell_0c**2)
	Pc = x * Pc_act + (1 - x) * Pc_HS

	return Pc, Pa


cf_data = np.asarray([(0.52226665, 0.05, 0.57945, 0.05),
					  # (None, 0.06, None, 0.06),
					  # (None, 0.07, None, 0.07),
					  # (None, 0.08, None, 0.08),
					  # (None, 0.09, None, 0.09),
					  # (None, 0.1, None, 0.1),
					  # (None, 0.2, None, 0.2),
					  # (None, 0.3, None, 0.3),
					  # (None, 0.4, None, 0.4),
					  (0.57770184, 0.50, 0.70215, 0.50),
					  # (None, 0.6, None, 0.6),
					  # (None, 0.7, None, 0.7),
					  # (None, 0.8, None, 0.8),
					  # (None, 0.9, None, 0.9),
					  # (None, 1.0, None, 1.0),
					  # (None, 2.0, None, 2.0),
					  # (None, 3.0, None, 3.0),
					  # (None, 4.0, None, 4.0),
                      (0.59602888, 5.00, 0.73946, 5.00),
                      (0.59596600, 7.50, 0.74050, 7.50),
                      (0.59521300, 10.00, 0.74050, 10.00),
                      (0.59399964, 12.50, 0.73998, 12.50),
                      (0.59199193, 15.00, 0.73998, 15.00),
                      (0.57135176, 20.00, 0.73998, 20.00),
                      (0.56181091, 20.50, 0.73998, 20.50),
                      (0.55463744, 21.00, 0.73998, 21.00),
                      (0.31947480, 21.50, 0.73998, 21.50),
                      (0.30727338, 22.00, 0.73998, 22.00),
                      (0.27570402, 23.00, 0.73998, 23.00),
                      (0.25784975, 24.00, 0.73998, 24.00),
                      (0.23883106, 25.00, 0.73998, 25.00),
                      (0.18378581, 30.00, 0.73998, 30.00),
                      (0.14963249, 35.00, 0.73998, 35.00),
                      (0.13868171, 37.50, 0.73998, 37.50),
                      (0.12839358, 40.00, 0.73998, 40.00),
                      (0.11287778, 45.00, 0.73998, 45.00),
                      (0.10126001, 50.00, 0.73998, 50.00),
                      (0.09228456, 55.50, 0.73998, 55.50),
                      (0.08368932, 62.50, 0.74050, 62.50),
                      (0.07381443, 71.50, 0.73998, 71.50),
                      (0.06475552, 83.50, 0.74050, 83.50),
                      (0.05721007, 100.0, 0.74050, 100.00),
                      (0.04821929, 125.0, 0.73998, 125.00),
                      (0.04082957, 166.5, 0.73998, 166.50),
                      (0.03248872, 250.0, 0.74050, 250.00),
                      (0.02444256, 500.0, 0.73998, 500.00)])

rl_cf_data = cf_data[:, 1]

cf_data = list(cf_data)

cf_data.append((None, 0.11, None, 0.11))
cf_data.append((None, 0.12, None, 0.12))
cf_data.append((None, 0.13, None, 0.13))
cf_data.append((None, 0.14, None, 0.14))
cf_data.append((None, 0.15, None, 0.15))
cf_data.append((None, 0.16, None, 0.16))
cf_data.append((None, 0.17, None, 0.17))
cf_data.append((None, 0.18, None, 0.18))
cf_data.append((None, 0.19, None, 0.19))
# cf_data.append((None, 0.2, None, 0.2))
cf_data.append((None, 0.21, None, 0.21))
cf_data.append((None, 0.22, None, 0.22))
cf_data.append((None, 0.24, None, 0.24))
cf_data.append((None, 0.25, None, 0.25))
cf_data.append((None, 0.26, None, 0.26))
cf_data.append((None, 0.27, None, 0.27))
cf_data.append((None, 0.28, None, 0.28))
cf_data.append((None, 0.29, None, 0.29))

cf_data.append((None, 0.31, None, 0.31))
cf_data.append((None, 0.32, None, 0.32))
cf_data.append((None, 0.34, None, 0.34))
cf_data.append((None, 0.35, None, 0.35))
cf_data.append((None, 0.36, None, 0.36))
cf_data.append((None, 0.37, None, 0.37))
cf_data.append((None, 0.38, None, 0.38))
cf_data.append((None, 0.39, None, 0.39))

cf_data.append((None, 0.41, None, 0.41))
cf_data.append((None, 0.42, None, 0.42))
cf_data.append((None, 0.44, None, 0.44))
cf_data.append((None, 0.45, None, 0.45))
cf_data.append((None, 0.46, None, 0.46))
cf_data.append((None, 0.47, None, 0.47))
cf_data.append((None, 0.48, None, 0.48))
cf_data.append((None, 0.49, None, 0.49))

cf_data.append((None, 0.51, None, 0.51))
cf_data.append((None, 0.52, None, 0.52))
cf_data.append((None, 0.54, None, 0.54))
cf_data.append((None, 0.55, None, 0.55))
cf_data.append((None, 0.56, None, 0.56))
cf_data.append((None, 0.57, None, 0.57))
cf_data.append((None, 0.58, None, 0.58))
cf_data.append((None, 0.59, None, 0.59))

cf_data.append((None, 0.61, None, 0.61))
cf_data.append((None, 0.62, None, 0.62))
cf_data.append((None, 0.64, None, 0.64))
cf_data.append((None, 0.65, None, 0.65))
cf_data.append((None, 0.66, None, 0.66))
cf_data.append((None, 0.67, None, 0.67))
cf_data.append((None, 0.68, None, 0.68))
cf_data.append((None, 0.69, None, 0.69))

cf_data.append((None, 0.71, None, 0.71))
cf_data.append((None, 0.72, None, 0.72))
cf_data.append((None, 0.74, None, 0.74))
cf_data.append((None, 0.75, None, 0.75))
cf_data.append((None, 0.76, None, 0.76))
cf_data.append((None, 0.77, None, 0.77))
cf_data.append((None, 0.78, None, 0.78))
cf_data.append((None, 0.79, None, 0.79))

cf_data.append((None, 0.81, None, 0.81))
cf_data.append((None, 0.82, None, 0.82))
cf_data.append((None, 0.84, None, 0.84))
cf_data.append((None, 0.85, None, 0.85))
cf_data.append((None, 0.86, None, 0.86))
cf_data.append((None, 0.87, None, 0.87))
cf_data.append((None, 0.88, None, 0.88))
cf_data.append((None, 0.89, None, 0.89))

cf_data.append((None, 0.91, None, 0.91))
cf_data.append((None, 0.92, None, 0.92))
cf_data.append((None, 0.94, None, 0.94))
cf_data.append((None, 0.95, None, 0.95))
cf_data.append((None, 0.96, None, 0.96))
cf_data.append((None, 0.97, None, 0.97))
cf_data.append((None, 0.98, None, 0.98))
cf_data.append((None, 0.99, None, 0.99))

cf_data.append((None, 0.6, None, 0.6))
cf_data.append((None, 0.7, None, 0.7))
cf_data.append((None, 0.8, None, 0.8))
cf_data.append((None, 0.9, None, 0.9))
cf_data.append((None, 1.0, None, 1.0))
cf_data.append((None, 1.1, None, 1.1))
cf_data.append((None, 1.2, None, 1.2))
cf_data.append((None, 1.3, None, 1.3))
cf_data.append((None, 1.4, None, 1.4))
cf_data.append((None, 1.5, None, 1.5))
cf_data.append((None, 1.6, None, 1.6))
cf_data.append((None, 1.7, None, 1.7))
cf_data.append((None, 1.8, None, 1.8))
cf_data.append((None, 1.9, None, 1.9))
cf_data.append((None, 2.1, None, 2.1))
cf_data.append((None, 2.2, None, 2.2))
cf_data.append((None, 2.3, None, 2.3))
cf_data.append((None, 2.4, None, 2.4))
cf_data.append((None, 2.5, None, 2.5))
cf_data.append((None, 2.6, None, 2.6))
cf_data.append((None, 2.7, None, 2.7))
# cf_data.append((None, 2.8, None, 2.8))
# cf_data.append((None, 2.9, None, 2.9))
cf_data.append((None, 3.1, None, 3.1))
cf_data.append((None, 3.2, None, 3.2))
cf_data.append((None, 3.3, None, 3.3))
cf_data.append((None, 3.4, None, 3.4))
cf_data.append((None, 3.5, None, 3.5))
cf_data.append((None, 3.6, None, 3.6))
cf_data.append((None, 3.7, None, 3.7))
cf_data.append((None, 3.8, None, 3.8))
cf_data.append((None, 3.9, None, 3.9))

cf_data.append((None, 18.0, None, 18.0))
cf_data.append((None, 18.25, None, 18.25))
cf_data.append((None, 18.5, None, 18.5))
cf_data.append((None, 18.75, None, 18.75))
# cf_data.append((None, 19.0, None, 19.0))
# cf_data.append((None, 19.25, None, 19.25))

min_l = 0.05
cf_ref = np.loadtxt('cf_ref_new3.txt')
cf_ref = cf_ref[(cf_ref[:, 1] > 0) & (cf_ref[:, 2] > 0) & (cf_ref[:, 0] >= min_l)]

for i in range(len(cf_ref)):
	rl = cf_ref[i, 0]
	if rl < 0.50001 or min(np.abs(rl - rl_cf_data)) > 1e-5:
		cf_data.append((cf_ref[i, 1], rl, cf_ref[i, 2], rl))



# phi = np.asarray(list(np.linspace(1e-3, 0.73999, 10000)) + [0.644998])
# phi = phi[np.argsort(phi)]
phi = np.asarray(list(np.linspace(1e-3, 0.64, 5000)) + list(np.linspace(0.6401, 0.645, 1000)) + list(np.linspace(0.64501, 0.72, 4000)) + list(np.linspace(0.7201, 0.7399999, 2000)))
phi = phi[np.argsort(phi)]
# phi = np.linspace(1e-3, 0.73999, 10000)
rl = 21.5
runlength = 21.5
kBT = zeta * U * rl / 6
psi = get_psi_star(phi, rl)
phi_max = get_phi_max(psi, rl)
Pc, Pa = get_Ps(phi, psi, phi_max, rl)





P = Pc + Pa

print(phi[-1])
print(P[-1])
print(max(P))
print(phi[np.argmax(P)])

# print(Pc[phi > 0.515])
# print(Pa[phi > 0.515])

# print(max(phi))

# plt.figure()
# plt.plot(phi, psi)
# plt.show()

# plt.figure()
# plt.plot(phi, Pa)
# plt.plot(phi, phi * kBT / v)
# plt.show()

# plt.figure()
# plt.plot(phi, phi_max)
# plt.show()

possible_min = argrelextrema(P, np.less)
possible_max = argrelextrema(P, np.greater)
# if Pc[-1] < max(Pc + Pa) - 1e-4:
# 	phi = np.append(phi, np.linspace(max(phi), 0.74))
# 	phi_max = np.append(phi_max, np.asarray([0.74 / v] * 50))
# 	psi = np.append(psi, np.asarray([1.0] * 50))
# 	Pa = np.append(Pa, np.asarray([0.0] * 50))
# 	Pc = np.append(Pc, np.linspace(Pc[-1], max(Pc) * 1.1))


P = Pc + Pa
E_r_eqm = 1 / phi
E_r_act = Pc
E_rr_eqm = 1 / phi**2
E_rp_eqm = np.zeros(len(phi))
E_rr_act = np.diff(Pc) / np.diff(phi)
E_rp_act = np.diff(Pc) / np.diff(phi * psi)
E_rp_act[psi[1:] < 1e-12] = 0

if len(possible_min[0]) == 1:
	spinodall1 = min(phi[possible_max])
	spinodalh1 = min(phi[possible_min])

	# phif_eqm, phis_eqm, P_coex_eqm = get_coex(P, phi, psi, spinodall1, spinodalh1, E_r_eqm)
	phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh1, E_rr_eqm, E_rp_eqm)
else:
	spinodall1 = min(phi[possible_max])
	spinodalh2 = max(phi[possible_min])

	# phif_eqm, phis_eqm, P_coex_eqm = get_coex(P, phi, psi, spinodall1, spinodalh1, E_r_eqm)
	phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh2, E_rr_eqm, E_rp_eqm)

# print((phif_eqm, phis_eqm))

kBT = zeta * U * runlength / 6
Ea = zeta * U * 2 * r
E_div_v = Ea / v
ell_div_D = runlength * 0.5 / r
ell_div_s = runlength / sigma
nondim_factor = zeta * U * 2 * r / v
rl_func = (1 + np.tanh(100 * (0.5 * runlength / r - 18.3 * 0.5 / r)**1)) / 2

# Pa = nondim_factor * (1 + np.tanh(10 * psi_star) * (1 + 0 * rl_func)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi**(1 - 0. * np.tanh(10 * psi) * rl_func) / (1 - phi / phi_max)**(1 - 0.0 * np.tanh(10 * psi) * rl_func)) / 6
# Pa = nondim_factor * (1 + np.tanh(10 * psi_star) * (1 - (1 + np.tanh(1 * (0.5 * runlength / r - 20 * 0.5 / r)**1)) / 2)) * phi * ell_div_D / (1 + (1 - np.exp(-2**(7.0/6.0) * ell_div_D)) * phi / (1 - phi / phi_max)**(1)) / 6


# beta = 0.5 + np.tanh(10 * psi) * (0.05 - 0.5) * rl_func # + np.tanh(psi) * (1e-4 - 0.1) * (1 + np.tanh(2 * (0.5 * runlength / r - 19.0 * 0.5 / r))) / 2 # + (0.05 - 0.1) * (0.5 + np.tanh(10 * psi) * np.tanh(10 * (0.5 * runlength / r - 21.0)))
# # beta = 0.5 #+ (0.3 - 0.5) * np.tanh(psi * 0.5 * runlength / r)
# Pc_act = nondim_factor * 2**(-7/6) * phi**(2 + 0. * np.tanh(10 * psi) * rl_func) / (1 - (phi / phi_max))**beta * (1 - 5/6 * np.tanh(10 * psi))

# Pc_HS = (kBT / v) * phi / (1 - (phi / phi_max))**1. # / nondim_factor

# x = np.tanh(1 * np.log(runlength * 0.5 / r + 1) + np.exp(runlength**2 / 1e2))
# print(x)
# Pc2 = x * Pc_act + (1 - x) * Pc_HS

# plt.figure()
# # plt.plot(phi, P)
# plt.plot(phi, Pc)
# plt.plot(phi, Pc_act)
# plt.plot(phi, Pc2)
# # plt.plot(phi, Pa)
# # plt.plot(phi, Pc2 / rl)
# plt.show()

# # plt.figure()
# # plt.plot(phi[:-1], Pc[:-1] * np.diff(P))
# # # plt.plot(phi, Pc2 / rl)
# # plt.show()

# plt.figure()
# mu = np.cumsum(Pc[:-1] * np.diff(P))
# mu -= min(mu)
# plt.plot(phi[:-1], mu / max(np.abs(mu)))
# plt.plot(phi, P / max(np.abs(P)))
# # plt.plot(phi, Pc2 / rl)
# plt.show()

# # plt.figure()
# # plt.plot(phi, Pa)
# # # plt.plot(phi, Pc2 / rl)
# # plt.show()

# assert False


phifs_sim = []
phiss_sim = []
rl_sim = []
phifs_eqm = []
phiss_eqm = []
rl_eqm = []
phifs_eqm2 = []
phiss_eqm2 = []
rl_eqm2 = []
phils_eqm = []
phigs_eqm = []
rl_mips_eqm = []
phifs_act = []
phiss_act = []
phils_act = []
phigs_act = []
rl_act = []
rl_mips_act = []
for phif_sim, rl, phis_sim, _ in cf_data:
	# phi = np.asarray(list(np.linspace(1e-3, 0.739999, 10000)) + [0.64499999])
	# phi = phi[np.argsort(phi)]
	# phi = np.linspace(1e-3, 0.739999, 10000)
	phi = np.asarray(list(np.linspace(1e-3, 0.64, 5000)) + list(np.linspace(0.6401, 0.645, 2000)) + list(np.linspace(0.64501, 0.73999, 6000)))
	phi = phi[np.argsort(phi)]
	print()
	print(rl)
	print((phif_sim, phis_sim))
	if phif_sim is not None:
		phifs_sim.append(phif_sim)
		phiss_sim.append(phis_sim)
		rl_sim.append(rl)
	psi = get_psi_star(phi, rl)
	phi_max = get_phi_max(psi, rl)
	Pc, Pa = get_Ps(phi, psi, phi_max, rl)
	P = Pc + Pa
	# print(np.count_nonzero(np.isnan(Pc)))
	# plt.figure()
	# plt.plot(phi, P)
	# plt.show()
	# # plt.figure()
	# # plt.plot(phi, P * Pc)
	# # plt.show()
	# assert False

	possible_min = argrelextrema(P, np.less)
	possible_max = argrelextrema(P, np.greater)
	# if Pc[-1] < max(Pc + Pa) - 1e-4:
	# 	phi = np.append(phi, np.linspace(max(phi), 0.74))
	# 	phi_max = np.append(phi_max, np.asarray([0.74 / v] * 50))
	# 	psi = np.append(psi, np.asarray([1.0] * 50))
	# 	Pa = np.append(Pa, np.asarray([0.0] * 50))
	# 	Pc = np.append(Pc, np.linspace(Pc[-1], max(Pc) * 1.1))


	P = Pc + Pa
	sort1 = np.argsort(Pc)
	sort2 = np.argsort(Pc[1:])
	E_r_eqm = 1 / phi
	E_r_act = Pc
	E_rr_eqm = 1 / phi**2
	E_rp_eqm = np.zeros(len(phi))
	E_rr_act = np.diff(Pc) / np.diff(phi)
	E_rp_act = np.diff(Pc) / np.diff(psi)
	E_rp_act[psi[1:] < 1e-12] = 0
	# print(min(np.diff(phi)))
	# # print(np.count_nonzero(np.isnan(np.diff(phi))))
	# print(np.count_nonzero(np.isnan(E_rp_act)))
	# plt.figure()
	# plt.plot(phi, Pc)
	# plt.show()

	if len(possible_min[0]) == 1:
		spinodall1 = min(phi[possible_max])
		spinodalh1 = min(phi[possible_min])
		# print((spinodall1, spinodalh1))

		phif_eqm, phis_eqm, P_coex_eqm = get_coex3(P, phi, psi, spinodall1, spinodalh1, E_r_eqm)
		# phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh1, E_rr_eqm, E_rp_eqm)
		phifs_eqm.append(phif_eqm)
		phiss_eqm.append(phis_eqm)
		rl_eqm.append(rl)
		print((phif_eqm, phis_eqm))
		phif_act, phis_act, P_coex_act = get_coex(P, phi, psi, spinodall1, spinodalh1, E_r_act)
		# phif_act, phis_act, P_coex_act = get_coex2(P[sort1], phi[sort1], psi[sort1], spinodall1, spinodalh1, E_rr_act[sort2], E_rp_act[sort2])
		phifs_act.append(phif_act)
		phiss_act.append(phis_act)
		rl_act.append(rl)
		print((phif_act, phis_act))


	if len(possible_min[0]) > 1:
		spinodall1 = min(phi[possible_max])
		spinodall2 = max(phi[possible_max])
		spinodalh1 = min(phi[possible_min])
		spinodalh2 = max(phi[possible_min])
		# print((spinodall2, spinodalh2))
		# assert False
		if runlength > 18.8:
			phif_eqm, phis_eqm, P_coex_eqm = get_coex4(P, phi, psi, spinodall1, spinodall2, spinodalh1, spinodalh2, E_r_eqm)
			# phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh2, E_rr_eqm, E_rp_eqm)
			phifs_eqm.append(phif_eqm)
			phiss_eqm.append(phis_eqm)
			phif_eqm2, phis_eqm2, P_coex_eqm2 = get_coex3(P, phi, psi, spinodall1, spinodalh2, E_r_eqm)
			phifs_eqm2.append(phif_eqm2)
			phiss_eqm2.append(phis_eqm2)
			rl_eqm.append(rl)
			rl_eqm2.append(rl)
			print((phif_eqm, phis_eqm))
			print((phif_eqm2, phis_eqm2))
		else:
			phif_eqm, phis_eqm, P_coex_eqm = get_coex3(P, phi, psi, spinodall1, spinodalh2, E_r_eqm)
			# phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh2, E_rr_eqm, E_rp_eqm)
			phifs_eqm.append(phif_eqm)
			phiss_eqm.append(phis_eqm)
			rl_eqm.append(rl)
			print((phif_eqm, phis_eqm))
		phif_act, phis_act, P_coex_act = get_coex(P, phi, psi, spinodall1, spinodalh2, E_r_act)
		# phif_act, phis_act, P_coex_act = get_coex2(P[sort1], phi[sort1], psi[sort1], spinodall1, spinodalh2, E_rr_act[sort2], E_rp_act[sort2])
		phifs_act.append(phif_act)
		phiss_act.append(phis_act)
		rl_act.append(rl)
		print((phif_act, phis_act))

		phig_eqm, phil_eqm, P_coex_eqm = get_coex3(P[phi < 0.645], phi[phi < 0.645], np.zeros(len(phi))[phi < 0.645], spinodall1, spinodalh1, E_r_eqm[phi < 0.645])
		# phif_eqm, phis_eqm, P_coex_eqm = get_coex2(P, phi, psi, spinodall1, spinodalh2, E_rr_eqm, E_rp_eqm)
		phigs_eqm.append(phig_eqm)
		phils_eqm.append(phil_eqm)
		rl_mips_eqm.append(rl)

		phig_act, phil_act, P_coex_act = get_coex3(P[phi < 0.645], phi[phi < 0.645], np.zeros(len(phi))[phi < 0.645], spinodall1, spinodalh1, E_r_act[phi < 0.645])
		# phif_act, phis_act, P_coex_act = get_coex2(P[sort1], phi[sort1], psi[sort1], spinodall1, spinodalh2, E_rr_act[sort2], E_rp_act[sort2])
		phigs_act.append(phig_act)
		phils_act.append(phil_act)
		rl_mips_act.append(rl)

phifs_sim = np.asarray(phifs_sim)
phiss_sim = np.asarray(phiss_sim)
rl_sim = np.asarray(rl_sim)
phifs_eqm = np.asarray(phifs_eqm)
phiss_eqm = np.asarray(phiss_eqm)
rl_eqm = np.asarray(rl_eqm)
phifs_eqm2 = np.asarray(phifs_eqm2)
phiss_eqm2 = np.asarray(phiss_eqm2)
rl_eqm2 = np.asarray(rl_eqm2)
phils_eqm = np.asarray(phils_eqm)
phigs_eqm = np.asarray(phigs_eqm)
rl_mips_eqm = np.asarray(rl_mips_eqm)
phifs_act = np.asarray(phifs_act)
phiss_act = np.asarray(phiss_act)
phils_act = np.asarray(phils_act)
phigs_act = np.asarray(phigs_act)
rl_act = np.asarray(rl_act)
rl_mips_act = np.asarray(rl_mips_act)

sort_sim = np.argsort(rl_sim)
sort_eqm = np.argsort(rl_eqm)
sort_eqm2 = np.argsort(rl_eqm2)
sort_act = np.argsort(rl_act)

rl_sim = rl_sim[sort_sim]
phifs_sim = phifs_sim[sort_sim]
phiss_sim = phiss_sim[sort_sim]
rl_eqm = rl_eqm[sort_eqm]
phifs_eqm = phifs_eqm[sort_eqm]
phiss_eqm = phiss_eqm[sort_eqm]
rl_eqm2 = rl_eqm2[sort_eqm2]
phifs_eqm2 = phifs_eqm2[sort_eqm2]
phiss_eqm2 = phiss_eqm2[sort_eqm2]
rl_act = rl_act[sort_act]
phifs_act = phifs_act[sort_act]
phiss_act = phiss_act[sort_act]


np.savetxt('cf_ref.txt', np.hstack((rl_sim.reshape(-1, 1), phifs_sim.reshape(-1, 1), phiss_sim.reshape(-1, 1))))
np.savetxt('cf_eqm.txt', np.hstack((rl_eqm.reshape(-1, 1), phifs_eqm.reshape(-1, 1), phiss_eqm.reshape(-1, 1))))
np.savetxt('cf_eqm2.txt', np.hstack((rl_eqm2.reshape(-1, 1), phifs_eqm2.reshape(-1, 1), phiss_eqm2.reshape(-1, 1))))
np.savetxt('cf_act.txt', np.hstack((rl_act.reshape(-1, 1), phifs_act.reshape(-1, 1), phiss_act.reshape(-1, 1))))
np.savetxt('lg_eqm.txt', np.hstack((rl_mips_eqm.reshape(-1, 1), phigs_eqm.reshape(-1, 1), phils_eqm.reshape(-1, 1))))
np.savetxt('lg_act.txt', np.hstack((rl_mips_act.reshape(-1, 1), phigs_act.reshape(-1, 1), phils_act.reshape(-1, 1))))



cf_ref = np.loadtxt('cf_ref.txt')
rl_sim = cf_ref[:, 0]
phifs_sim = cf_ref[:, 1]
phiss_sim = cf_ref[:, 2]

cf_eqm = np.loadtxt('cf_eqm.txt')
rl_eqm = cf_eqm[:, 0]
phifs_eqm = cf_eqm[:, 1]
phiss_eqm = cf_eqm[:, 2]

cf_eqm2 = np.loadtxt('cf_eqm2.txt')
rl_eqm2 = cf_eqm2[:, 0]
phifs_eqm2 = cf_eqm2[:, 1]
phiss_eqm2 = cf_eqm2[:, 2]

lg_eqm = np.loadtxt('lg_eqm.txt')
rl_mips_eqm = lg_eqm[:, 0]
phigs_eqm = lg_eqm[:, 1]
phils_eqm = lg_eqm[:, 2]

cf_act = np.loadtxt('cf_act.txt')
rl_act = cf_act[:, 0]
phifs_act = cf_act[:, 1]
phiss_act = cf_act[:, 2]

lg_act = np.loadtxt('lg_act.txt')
rl_mips_act = lg_act[:, 0]
phigs_act = lg_act[:, 1]
phils_act = lg_act[:, 2]



mips_data = np.array([(0.406218168, 19.5, 0.533850168, 19.5),
                      (0.397283616, 20, 0.548228996, 20),
                      (0.380613973, 20.5, 0.5595614, 20.5),
                      (0.36978804, 21, 0.556025217, 21),
                      (0.363632244, 21.5, 0.56796914, 21.5),
                      (0.348635727, 22, 0.569170432, 22),
                      (0.338018063, 22.5, 0.57426917, 22.5),
                      (0.335518514, 23, 0.575719187, 23),
                      (0.326482144, 23.5, 0.577346316, 23.5),
                      (0.319431387, 24, 0.578127871, 24),
                      (0.313312217, 24.5, 0.581335736, 24.5),
                      (0.306997822, 25, 0.58288359, 25),
                      (0.299835679, 25.5, 0.585240767, 25.5),
                      (0.293199725, 26, 0.586861688, 26),
                      (0.290179514, 26.5, 0.588030577, 26.5),
                      (0.283371661, 27, 0.589779678, 27),
                      (0.277773439, 27.5, 0.589860562, 27.5),
                      (0.259544015, 30, 0.594685462, 30),
                      (0.240362631, 32.5, 0.598936469, 32.5),
                      (0.237686105, 33, 0.599556358, 33),
                      (0.234674274, 33.5, 0.600577235, 33.5),
                      (0.231648033, 34, 0.601144259, 34),
                      (0.228077798, 34.5, 0.60152121, 34.5),
                      (0.226427024, 35, 0.602346904, 35),
                      (0.213611867, 37.5, 0.604760677, 37.5),
                      (0.201916574, 40, 0.60659055, 40),
                      (0.193057529, 42.5, 0.608618074, 42.5),
                      (0.183848247, 45, 0.610232608, 45),
                      (0.175979661, 47.5, 0.612095685, 47.5),
                      (0.16946723,  50, 0.612776074, 50),
                      (0.156368889, 55.5, 0.615210923, 55.5),
                      (0.142490013, 62.5, 0.617876632, 62.5),
                      (0.128936052, 71.5, 0.620167209, 71.5),
                      (0.115150735, 83.5, 0.622566029, 83.5),
                      (0.101750283, 100, 0.625106941, 100),
                      (0.08770798,  125, 0.627691313, 125),
                      (0.073638509, 166.5, 0.630415813, 166.5),
                      (0.059727943, 250, 0.633694721, 250),
                      (0.044613938, 500, 0.637660437, 500)])
                      # (0.035257728, 1000, 0.640648501, 1000),
                      # (0.032438162, 1500, 0.642358393, 1500),
                      # (0.029952805, 2000, 0.64341217, 2000),
                      # (0.028314247, 2500, 0.644073649, 2500),
                      # (0.026166017, 3000, 0.644435475, 3000),
                      # (0.034814421, 3500, 0.640101002, 3500)])





plt.figure(figsize=(9, 7))
# plt.scatter(phifs_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
# plt.scatter(phiss_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
plt.scatter(mips_data[:, 0], mips_data[:, 1] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
plt.scatter(mips_data[:, 2], mips_data[:, 1] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
plt.plot(phifs_act, rl_act * 0.5 / r, lw=2.5, color='k')
plt.plot(phiss_act, rl_act * 0.5 / r, lw=2.5, color='k')
# plt.plot(phifs_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
# plt.plot(phiss_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
# plt.plot(phifs_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
# plt.plot(phiss_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
plt.plot(phigs_act, rl_mips_act * 0.5 / r, lw=2.5, color='r')
plt.plot(phils_act, rl_mips_act * 0.5 / r, lw=2.5, color='r')
# plt.plot(phigs_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')
# plt.plot(phils_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')

plt.scatter(phifs_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
plt.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)


plt.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)

plt.yscale('log')

ax = plt.gca()
ax.set_xlabel(r'$\phi$', fontsize=18)
ax.tick_params(width=3, length=10, which='major', labelsize=16)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.5)

# ax.set_xticks([0.4, 0.5])
ticks = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70 , 80, 90, 200, 300, 400, 500]
ax.set_yticks(ticks, minor=True)
ax.set_yticks([0.1, 1.0, 10, 100])
labs = ['' for _ in ticks]
ax.set_yticklabels(labs, minor=True)
ax.set_yticklabels([0.1, 1.0, 10, 100])
ax.set_ylabel(r'$\ell_0 / D$', fontsize=18)

# plt.show()
plt.savefig('ABP_pd_main.svg', format='svg', dpi=1200)
plt.close()


"""

extra_cf = [(None, 21.25, None, 21.25),
			(None, 21.75, None, 21.75),
			(None, 22.25, None, 22.25),
			(None, 22.75, None, 22.75),
			(None, 23.25, None, 23.25),
			(None, 23.75, None, 23.75),
			(None, 26, None, 26),
			(None, 27, None, 27),
			(None, 28, None, 28),
			(None, 29, None, 29),
			(None, 31, None, 31),
			(None, 32, None, 32),
			(None, 33, None, 33),
			(None, 34, None, 34),
			(None, 36, None, 36)]


extra_phifs_act = []
extra_phiss_act = []
extra_rl_act = []
for phif_sim, rl, phis_sim, _ in extra_cf:
	# phi = np.asarray(list(np.linspace(1e-3, 0.739999, 10000)) + [0.64499999])
	# phi = phi[np.argsort(phi)]
	# phi = np.linspace(1e-3, 0.739999, 10000)
	phi = np.asarray(list(np.linspace(1e-3, 0.64, 5000)) + list(np.linspace(0.6401, 0.645, 2000)) + list(np.linspace(0.64501, 0.73999, 6000)))
	phi = phi[np.argsort(phi)]
	print()
	print(rl)
	print((phif_sim, phis_sim))
	if phif_sim is not None:
		phifs_sim.append(phif_sim)
		phiss_sim.append(phis_sim)
		rl_sim.append(rl)
	psi = get_psi_star(phi, rl)
	phi_max = get_phi_max(psi, rl)
	Pc, Pa = get_Ps(phi, psi, phi_max, rl)
	P = Pc + Pa
	# print(np.count_nonzero(np.isnan(Pc)))
	# plt.figure()
	# plt.plot(phi, P)
	# plt.show()
	# # plt.figure()
	# # plt.plot(phi, P * Pc)
	# # plt.show()
	# assert False

	possible_min = argrelextrema(P, np.less)
	possible_max = argrelextrema(P, np.greater)
	# if Pc[-1] < max(Pc + Pa) - 1e-4:
	# 	phi = np.append(phi, np.linspace(max(phi), 0.74))
	# 	phi_max = np.append(phi_max, np.asarray([0.74 / v] * 50))
	# 	psi = np.append(psi, np.asarray([1.0] * 50))
	# 	Pa = np.append(Pa, np.asarray([0.0] * 50))
	# 	Pc = np.append(Pc, np.linspace(Pc[-1], max(Pc) * 1.1))


	P = Pc + Pa
	sort1 = np.argsort(Pc)
	sort2 = np.argsort(Pc[1:])
	E_r_eqm = 1 / phi
	E_r_act = Pc
	E_rr_eqm = 1 / phi**2
	E_rp_eqm = np.zeros(len(phi))
	E_rr_act = np.diff(Pc) / np.diff(phi)
	E_rp_act = np.diff(Pc) / np.diff(psi)
	E_rp_act[psi[1:] < 1e-12] = 0
	# print(min(np.diff(phi)))
	# # print(np.count_nonzero(np.isnan(np.diff(phi))))
	# print(np.count_nonzero(np.isnan(E_rp_act)))
	# plt.figure()
	# plt.plot(phi, Pc)
	# plt.show()

	if len(possible_min[0]) == 1:
		spinodall1 = min(phi[possible_max])
		spinodalh1 = min(phi[possible_min])

		phif_act, phis_act, P_coex_act = get_coex(P, phi, psi, spinodall1, spinodalh1, E_r_act)
		# phif_act, phis_act, P_coex_act = get_coex2(P[sort1], phi[sort1], psi[sort1], spinodall1, spinodalh1, E_rr_act[sort2], E_rp_act[sort2])
		extra_phifs_act.append(phif_act)
		extra_phiss_act.append(phis_act)
		extra_rl_act.append(rl)
		print((phif_act, phis_act))


	if len(possible_min[0]) > 1:
		spinodall1 = min(phi[possible_max])
		spinodall2 = max(phi[possible_max])
		spinodalh1 = min(phi[possible_min])
		spinodalh2 = max(phi[possible_min])
		phif_act, phis_act, P_coex_act = get_coex(P, phi, psi, spinodall1, spinodalh2, E_r_act)
		# phif_act, phis_act, P_coex_act = get_coex2(P[sort1], phi[sort1], psi[sort1], spinodall1, spinodalh2, E_rr_act[sort2], E_rp_act[sort2])
		extra_phifs_act.append(phif_act)
		extra_phiss_act.append(phis_act)
		extra_rl_act.append(rl)
		print((phif_act, phis_act))


for i in range(len(extra_phiss_act)):
	phifs_act = np.append(phifs_act, extra_phifs_act[i])
	phiss_act = np.append(phiss_act, extra_phiss_act[i])
	rl_act = np.append(rl_act, extra_rl_act[i])

sort = np.argsort(rl_act)
phifs_act = phifs_act[sort]
phiss_act = phiss_act[sort]
rl_act = rl_act[sort]


plt.figure(figsize=(4, 3))
# plt.scatter(phifs_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
# plt.scatter(phiss_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
plt.scatter(mips_data[:, 0][(mips_data[:, 1] > 9) & (mips_data[:, 1] < 40)], mips_data[:, 1][(mips_data[:, 1] > 9) & (mips_data[:, 1] < 40)] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
plt.scatter(mips_data[:, 2][(mips_data[:, 1] > 9) & (mips_data[:, 1] < 40)], mips_data[:, 1][(mips_data[:, 1] > 9) & (mips_data[:, 1] < 40)] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
plt.plot(phifs_act[(rl_act > 9) & (rl_act < 40)], rl_act[(rl_act > 9) & (rl_act < 40)] * 0.5 / r, lw=2.5, color='k')
plt.plot(phiss_act[(rl_act > 9) & (rl_act < 40)], rl_act[(rl_act > 9) & (rl_act < 40)] * 0.5 / r, lw=2.5, color='k')
# plt.plot(extra_phifs_act, extra_rl_act * 0.5 / r, lw=2.5, color='k')
# plt.plot(extra_phiss_act, extra_rl_act * 0.5 / r, lw=2.5, color='k')
# plt.plot(phifs_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
# plt.plot(phiss_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
# plt.plot(phifs_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
# plt.plot(phiss_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
plt.plot(phigs_act[(rl_mips_act > 9) & (rl_mips_act < 40)], rl_mips_act[(rl_mips_act > 9) & (rl_mips_act < 40)] * 0.5 / r, lw=2.5, color='r')
plt.plot(phils_act[(rl_mips_act > 9) & (rl_mips_act < 40)], rl_mips_act[(rl_mips_act > 9) & (rl_mips_act < 40)] * 0.5 / r, lw=2.5, color='r')
# plt.plot(phigs_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')
# plt.plot(phils_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')

plt.scatter(phifs_sim[(rl_sim > 9) & (rl_sim < 40)], rl_sim[(rl_sim > 9) & (rl_sim < 40)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
plt.scatter(phiss_sim[(rl_sim > 9) & (rl_sim < 40)], rl_sim[(rl_sim > 9) & (rl_sim < 40)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)

# plt.yscale('log')

ax = plt.gca()
# ax.set_xlabel(r'$\phi$', fontsize=18))
ax.tick_params(width=3, length=10, which='major', labelsize=14)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.5)

# ax.set_xticks([0.4, 0.5])
# ax.set_yticks([0.6725, 0.6775])

# ax.set_ylabel(r'$\ell_0 / D$', fontsize=18)

# plt.show()
plt.savefig('ABP_pd_main_inset.svg', format='svg', dpi=1200)
plt.close()
"""



# fig = plt.figure(figsize=(16, 5))
fig = plt.figure(figsize=(5.38, 12))
# gs = fig.add_gridspec(12, 1, hspace=0)
gs = fig.add_gridspec(3, 1)
gs_sub = gs[1].subgridspec(4, 1, hspace=0)
# axs = gs.subplots(sharex=True)
# print(axs)
# ax1 = fig.add_subplot(gs[:, :3])
# ax2 = fig.add_subplot(gs[0, 3:5])
# ax3 = fig.add_subplot(gs[1, 3:5], sharex=ax2)
# ax4 = fig.add_subplot(gs[2, 3:5], sharex=ax2)
# ax5 = fig.add_subplot(gs[3, 3:5], sharex=ax2)
# ax6 = fig.add_subplot(gs[:, 5:])
# ax1 = fig.add_subplot(gs[:, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
# ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)
# ax5 = fig.add_subplot(gs[3, 1], sharex=ax2)
# ax6 = fig.add_subplot(gs[:, 2])
# ax1 = fig.add_subplot(gs[:4, 0])
# ax2 = fig.add_subplot(gs[4, 0])
# ax3 = fig.add_subplot(gs[5, 0], sharex=ax2)
# ax4 = fig.add_subplot(gs[6, 0], sharex=ax2)
# ax5 = fig.add_subplot(gs[7, 0], sharex=ax2)
# ax6 = fig.add_subplot(gs[8:, 0])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs_sub[0])
ax3 = fig.add_subplot(gs_sub[1], sharex=ax2)
ax4 = fig.add_subplot(gs_sub[2], sharex=ax2)
ax5 = fig.add_subplot(gs_sub[3], sharex=ax2)
ax6 = fig.add_subplot(gs[2])
# plt.scatter(phifs_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
# plt.scatter(phiss_sim, rl_sim, facecolors='none', edgecolors='k', lw=2, s=100)
# plt.scatter(mips_data[:, 0], mips_data[:, 1] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
# plt.scatter(mips_data[:, 2], mips_data[:, 1] * 0.5 / r, facecolors='none', edgecolors='r', lw=2, s=100, marker='s')
ax1.plot(phifs_act, rl_act * 0.5 / r, lw=2.5, color='k')
ax1.plot(phiss_act, rl_act * 0.5 / r, lw=2.5, color='k')
ax1.plot(phifs_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
ax1.plot(phiss_eqm, rl_eqm * 0.5 / r, lw=2.5, color='k', ls='--')
ax1.plot(phifs_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
ax1.plot(phiss_eqm2, rl_eqm2 * 0.5 / r, lw=2.5, color='k', ls='-.')
# plt.plot(phigs_act, rl_mips_act * 0.5 / r, lw=2.5, color='r')
# plt.plot(phils_act, rl_mips_act * 0.5 / r, lw=2.5, color='r')
# plt.plot(phigs_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')
# plt.plot(phils_eqm, rl_mips_eqm * 0.5 / r, lw=2.5, color='r', ls='--')

ax1.scatter(phifs_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax1.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)


ax1.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax1.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)

ax1.set_yscale('log')

ax1.set_xlabel(r'$\phi$', fontsize=24)

xlim = ax1.get_xlim()
ax1.set_xlim([-1e-3, 0.74 + 20e-3])
ylim = ax1.get_ylim()
ax1.set_ylim([0.05 * 0.5 / r - 12e-3, 500 * 0.5 / r + 220])

ticks = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70 , 80, 90, 200, 300, 400, 500]
ax1.set_yticks(ticks, minor=True)
ax1.set_yticks([0.1, 1.0, 10, 100])
labs = ['' for _ in ticks]
ax1.set_yticklabels(labs, minor=True)
ax1.set_yticklabels([0.1, 1.0, 10, 100])
ax1.set_ylabel(r'$\ell_0 / D$', fontsize=24)



ax5.plot(phifs_act[rl_act * 0.5 / r < 1.0], rl_act[rl_act * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k')
ax5.plot(phiss_act[rl_act * 0.5 / r < 1.0], rl_act[rl_act * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k')
ax5.plot(phifs_eqm[rl_eqm * 0.5 / r < 1.0], rl_eqm[rl_eqm * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k', ls='--')
ax5.plot(phiss_eqm[rl_eqm * 0.5 / r < 1.0], rl_eqm[rl_eqm * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k', ls='--')
ax5.plot(phifs_eqm2[rl_eqm2 * 0.5 / r < 1.0], rl_eqm2[rl_eqm2 * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax5.plot(phiss_eqm2[rl_eqm2 * 0.5 / r < 1.0], rl_eqm2[rl_eqm2 * 0.5 / r < 1.0] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax5.scatter(phifs_sim[(rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[(rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phifs_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)], rl_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)], rl_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phifs_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)], rl_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phiss_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)], rl_sim[(rl_sim * 0.5 / r < 1.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
ax5.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax5.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)






# ax1.scatter(phiss_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)], rl_sim[(rl_sim >= 5.0) | (rl_sim == 0.05) | (rl_sim == 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phiss_sim[(rl_sim < 0.5) & (rl_sim > 0.05)], rl_sim[(rl_sim < 0.5) & (rl_sim > 0.05)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phiss_sim[rl_sim < 0.05], rl_sim[rl_sim < 0.05] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim > 0.5)], rl_sim[(rl_sim < 5.0) & (rl_sim > 0.5)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=200, s=150)
# ax1.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)


# ax1.scatter(phiss_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim == 0.5][0], rl_sim[rl_sim == 0.5][0] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phiss_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
# ax1.scatter(phifs_sim[rl_sim == 0.05][1], rl_sim[rl_sim == 0.05][1] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)





ax4.plot(phifs_act[(rl_act * 0.5 / r >= 1.0) & (rl_act * 0.5 / r < 10.0)], rl_act[(rl_act * 0.5 / r >= 1.0) & (rl_act * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k')
ax4.plot(phiss_act[(rl_act * 0.5 / r >= 1.0) & (rl_act * 0.5 / r < 10.0)], rl_act[(rl_act * 0.5 / r >= 1.0) & (rl_act * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k')
ax4.plot(phifs_eqm[(rl_eqm * 0.5 / r >= 1.0) & (rl_eqm * 0.5 / r < 10.0)], rl_eqm[(rl_eqm * 0.5 / r >= 1.0) & (rl_eqm * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax4.plot(phiss_eqm[(rl_eqm * 0.5 / r >= 1.0) & (rl_eqm * 0.5 / r < 10.0)], rl_eqm[(rl_eqm * 0.5 / r >= 1.0) & (rl_eqm * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax4.plot(phifs_eqm2[(rl_eqm2 * 0.5 / r >= 1.0) & (rl_eqm2 * 0.5 / r < 10.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 1.0) & (rl_eqm2 * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax4.plot(phiss_eqm2[(rl_eqm2 * 0.5 / r >= 1.0) & (rl_eqm2 * 0.5 / r < 10.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 1.0) & (rl_eqm2 * 0.5 / r < 10.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax4.scatter(phifs_sim[((rl_sim * 0.5 / r >= 1.0) & (rl_sim * 0.5 / r < 10.0))], rl_sim[((rl_sim * 0.5 / r >= 1.0) & (rl_sim * 0.5 / r < 10.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax4.scatter(phiss_sim[((rl_sim * 0.5 / r >= 1.0) & (rl_sim * 0.5 / r < 10.0))], rl_sim[((rl_sim * 0.5 / r >= 1.0) & (rl_sim * 0.5 / r < 10.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax4.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)], rl_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax4.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)], rl_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)] * 0.5 / r, facecolors='gray', alpha=0.5, edgecolors='w', lw=2, marker='o', zorder=100, s=150)
ax4.scatter(phifs_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)], rl_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax4.scatter(phiss_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)], rl_sim[(rl_sim < 5.0) & (rl_sim * 0.5 / r >= 1.0)] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)

ax3.plot(phifs_act[(rl_act * 0.5 / r >= 10.0) & (rl_act * 0.5 / r < 100.0)], rl_act[(rl_act * 0.5 / r >= 10.0) & (rl_act * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k')
ax3.plot(phiss_act[(rl_act * 0.5 / r >= 10.0) & (rl_act * 0.5 / r < 100.0)], rl_act[(rl_act * 0.5 / r >= 10.0) & (rl_act * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k')
ax3.plot(phifs_eqm[(rl_eqm * 0.5 / r >= 10.0) & (rl_eqm * 0.5 / r < 100.0)], rl_eqm[(rl_eqm * 0.5 / r >= 10.0) & (rl_eqm * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax3.plot(phiss_eqm[(rl_eqm * 0.5 / r >= 10.0) & (rl_eqm * 0.5 / r < 100.0)], rl_eqm[(rl_eqm * 0.5 / r >= 10.0) & (rl_eqm * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax3.plot(phifs_eqm2[(rl_eqm2 * 0.5 / r >= 10.0) & (rl_eqm2 * 0.5 / r < 100.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 10.0) & (rl_eqm2 * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax3.plot(phiss_eqm2[(rl_eqm2 * 0.5 / r >= 10.0) & (rl_eqm2 * 0.5 / r < 100.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 10.0) & (rl_eqm2 * 0.5 / r < 100.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax3.scatter(phifs_sim[((rl_sim * 0.5 / r >= 10.0) & (rl_sim * 0.5 / r < 100.0))], rl_sim[((rl_sim * 0.5 / r >= 10.0) & (rl_sim * 0.5 / r < 100.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax3.scatter(phiss_sim[((rl_sim * 0.5 / r >= 10.0) & (rl_sim * 0.5 / r < 100.0))], rl_sim[((rl_sim * 0.5 / r >= 10.0) & (rl_sim * 0.5 / r < 100.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)

ax2.plot(phifs_act[(rl_act * 0.5 / r >= 100.0) & (rl_act * 0.5 / r < 1000.0)], rl_act[(rl_act * 0.5 / r >= 100.0) & (rl_act * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k')
ax2.plot(phiss_act[(rl_act * 0.5 / r >= 100.0) & (rl_act * 0.5 / r < 1000.0)], rl_act[(rl_act * 0.5 / r >= 100.0) & (rl_act * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k')
ax2.plot(phifs_eqm[(rl_eqm * 0.5 / r >= 100.0) & (rl_eqm * 0.5 / r < 1000.0)], rl_eqm[(rl_eqm * 0.5 / r >= 100.0) & (rl_eqm * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax2.plot(phiss_eqm[(rl_eqm * 0.5 / r >= 100.0) & (rl_eqm * 0.5 / r < 1000.0)], rl_eqm[(rl_eqm * 0.5 / r >= 100.0) & (rl_eqm * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k', ls='--')
ax2.plot(phifs_eqm2[(rl_eqm2 * 0.5 / r >= 100.0) & (rl_eqm2 * 0.5 / r < 1000.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 100.0) & (rl_eqm2 * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax2.plot(phiss_eqm2[(rl_eqm2 * 0.5 / r >= 100.0) & (rl_eqm2 * 0.5 / r < 1000.0)], rl_eqm2[(rl_eqm2 * 0.5 / r >= 100.0) & (rl_eqm2 * 0.5 / r < 1000.0)] * 0.5 / r, lw=2.5, color='k', ls='-.')
ax2.scatter(phifs_sim[((rl_sim * 0.5 / r >= 100.0) & (rl_sim * 0.5 / r < 1000.0))], rl_sim[((rl_sim * 0.5 / r >= 100.0) & (rl_sim * 0.5 / r < 1000.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)
ax2.scatter(phiss_sim[((rl_sim * 0.5 / r >= 100.0) & (rl_sim * 0.5 / r < 1000.0))], rl_sim[((rl_sim * 0.5 / r >= 100.0) & (rl_sim * 0.5 / r < 1000.0))] * 0.5 / r, facecolors='none', edgecolors='k', lw=2, marker='o', zorder=100, s=150)


for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
	ax.tick_params(width=3, length=10, which='major', labelsize=16)
	ax.tick_params(width=3, length=6, which='minor')
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

ax2.set_xlim([0.0, 0.74 + 2e-2])
ax2.set_ylim([ax2.get_ylim()[0], 500])
ax5.set_ylim([0, ax5.get_ylim()[1]])

ax2.set_yticks([300])

ax3.set_yticks([50, 100])

ax4.set_yticks([5, 10])
ax5.set_yticks([0, 0.5, 1.0])
ax5.set_yticklabels([0, 0.5, '1'])

# ax5.set_xticklabels(['' for _ in ax2.get_xticklabels()])
# ax3.set_xticklabels(['' for _ in ax3.get_xticklabels()])
# ax4.set_xticklabels(['' for _ in ax4.get_xticklabels()])




ax5.set_xlabel(r'$\phi$', fontsize=24)
ax2.set_ylabel(r'$\ell_0 / D$', fontsize=24)

print()
print(phifs_sim[-1])
print((phifs_eqm[-1] - phifs_sim[-1]) / phifs_sim[-1])
print()
rl = rl_sim[-1]
phif = phifs_sim[-1]
phis = phiss_sim[-1]
print(np.abs(phif - phifs_act[np.argmin(rl - rl_act)]) / phif)
print(np.abs(phis - phiss_act[np.argmin(rl - rl_act)]) / phis)
print(np.abs(phif - phifs_eqm[np.argmin(rl - rl_eqm)]) / phif)
print(np.abs(phis - phiss_eqm[np.argmin(rl - rl_eqm)]) / phis)
print()

error_rl = []
errorf_act = []
errors_act = []
errorf_eqm = []
errors_eqm = []
for i in range(len(rl_sim)):
	rl = rl_sim[i]
	phif = phifs_sim[i]
	phis = phiss_sim[i]
	errorf_act.append(100 * np.abs(phif - phifs_act[np.argmin(np.abs(rl - rl_act))]) / phif)
	errors_act.append(100 * np.abs(phis - phiss_act[np.argmin(np.abs(rl - rl_act))]) / phis)
	errorf_eqm.append(100 * np.abs(phif - phifs_eqm[np.argmin(np.abs(rl - rl_eqm))]) / phif)
	errors_eqm.append(100 * np.abs(phis - phiss_eqm[np.argmin(np.abs(rl - rl_eqm))]) / phis)
	error_rl.append(rl * 0.5 / r)


ax6.scatter(error_rl, errorf_act, lw=2, facecolors='none', edgecolors='k', s=100)
ax6.scatter(error_rl, errors_act, lw=2, facecolors='none', edgecolors='k', marker='s', s=100)
ax6.scatter(error_rl, errorf_eqm, lw=2, facecolors='none', edgecolors='r', s=100)
ax6.scatter(error_rl, errors_eqm, lw=2, facecolors='none', edgecolors='r', marker='s', s=100)

ax6.set_xlabel(r'$\ell_0 / D$', fontsize=24)
ax6.set_ylabel(r'$|\Delta \phi|(\%)$', fontsize=24)
ax6.set_yscale('log')
ax6.set_xscale('log')

# ticks = [4e-6, 6e-6, 8e-6, 2e-5, 4e-5, 6e-5, 8e-5, 2e-4, 4e-4, 6e-4, 8e-4, 2e-3, 4e-3, 6e-3, 8e-3, 2e-2, 4e-2, 6e-2, 8e-2, 2e-1, 4e-1, 6e-1, 8e-1, 2, 4, 6, 8, 2e1, 30, 40]
ax6.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax6.minorticks_off()
ax6.set_yticklabels([0.001, 0.01, 0.1, 1, 10, 100, 1000])
# labs = ['' for _ in ticks]
# ax6.set_yticklabels(, minor=True)
# ax6.set_yticklabels([r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', 1, 10])
# ax6.set_ylim([4.1e-6, 45])
ax6.set_xlim([0.032, 600])

# ticks = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400]
# ax6.set_xticks(ticks, minor=True)
ax6.set_xticks([0.1, 1, 10, 100])
labs = ['' for _ in ticks]
# ax6.set_xticklabels(labs, minor=True)
ax6.set_xticklabels([0.1, 1, 10, 100])

plt.tight_layout()

plt.show()
# plt.savefig('ABP_pd_SI.svg', format='svg', dpi=1200)
# plt.close()



work_f_to_s_list = []
for i in range(len(rl_sim)):
	rl = rl_sim[i]
	phif = phifs_sim[i]
	phis = phiss_sim[i]
	phi = np.linspace(phif, phis, 10000)
	psi_star = get_psi_star(phi, rl)
	phi_max = get_phi_max(psi_star, rl)

	Pc, Pa = get_Ps(phi, psi_star, phi_max, rl)
	P = Pa + Pc
	P_coex = 0.5 * (P[0] + P[-1])

	work_f_to_s = np.sum((P - P_coex)[:-1] * np.diff(v / phi))
	work_f_to_s_list.append(work_f_to_s)


work_g_to_l_list = []
for i in range(len(mips_data[:, 1])):
	rl = mips_data[i, 1]
	phig = mips_data[i, 0]
	phil = mips_data[i, 2]

	phi = np.linspace(phig, phil, 10000)
	psi_star = np.zeros(len(phi))
	phi_max = 0.645 * np.ones(len(phi))

	Pc, Pa = get_Ps(phi, psi_star, phi_max, rl)
	P = Pa + Pc
	P_coex = 0.5 * (P[0] + P[-1])

	work_g_to_l = np.sum((P - P_coex)[:-1] * np.diff(v / phi))
	work_g_to_l_list.append(work_g_to_l)

print(work_g_to_l_list)
# print(work_f_to_s_list)

# E_unit = zeta * U * rl_sim / 6
# E_unitm = zeta * U * mips_data[:, 1] / 6
E_unit = zeta * U * 2 * r
E_unitm = zeta * U * 2 * r

tau = rl_sim / 18.8 - 1 * 0
taum = mips_data[:, 1] / 18.8 - 1 * 0
# plt.figure()
plt.figure(figsize=(6.5,8))
plt.scatter(tau, np.asarray(work_f_to_s_list) / E_unit, s=200, marker='o', lw=3, edgecolors='tab:blue', facecolors='none')
# plt.scatter(runlengths_plt, work_f_to_s2_list)
plt.scatter(taum, np.asarray(work_g_to_l_list) / E_unitm, s=200, marker='s', lw=3, edgecolors='tab:orange', facecolors='none')
plt.xlabel(r'$\ell_0 / \ell_0^c$', fontsize=24)
plt.ylabel(r'$\bar{W}$', fontsize=24)
plt.xscale('log')
# plt.legend(['Fluid --> Solid', 'Gas --> Liquid'])
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

plt.plot(xlim, [0, 0], lw=3, ls='--', color='k', zorder=-100)
plt.xlim(xlim)

# plt.ylim([ylim[0], 122])

ax.tick_params(width=3, length=10, which='major', labelsize=18)
ax.tick_params(width=3, length=6, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)

ticks = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40]
ax.set_xticks(ticks, minor=True)
ax.set_xticks([1e-2, 0.1, 1, 10])
labs = ['' for _ in ticks]
ax.set_xticklabels(labs, minor=True)
ax.set_xticklabels([0.01, 0.1, 1, 10])

# ax.axes.xaxis.set_ticklabels([-2, -1, 0, 1])
# plt.yticks(([-2, -1, 0, 1]))
# ax.axes.yaxis.set_ticklabels([])
plt.savefig('ABP_work_figure.svg', format='svg', dpi=1200)
# plt.show()
plt.close()
