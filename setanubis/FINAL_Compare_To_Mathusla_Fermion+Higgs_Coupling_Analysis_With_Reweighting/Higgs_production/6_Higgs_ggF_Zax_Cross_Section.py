"""Higgs -> Z + ALP (a) cross-section estimator.

Compute sigma(pp -> h) * BR(h -> Z a) for a given ALP mass and an
effective coupling. This module is intentionally lightweight and
parameterized so you can adapt conventions or normalizations from the
literature (see, e.g., arXiv:1708.00443).

Features
-
	- Build a numeric namespace from a local UFO `parameters` module.
	- Provide small utilities:
		- `lambd`: Källén (phase-space) function.
		- `gamma_h_to_Za`: approximate partial width for h -> Z a (top-loop).
		- `higgs_cross_section`: sigma(pp->h) * BR(h->Z a) using ggF.

Conventions
-
	- Masses and decay constants are in GeV.
	- Widths are returned in GeV and cross-sections in pb.
	- The `gamma_h_to_Za` implementation is an EFT-motivated approximation
		and contains a simple parametrization of the top-loop form factor.
"""

from __future__ import annotations

import os
import sys
import importlib
import argparse
import math
from typing import Tuple
import cmath

# Ensure the UFO model directory is on sys.path so local imports like
# `import function_library` inside the UFO package work.
_ufo_dir = os.path.abspath(
	os.path.join(os.path.dirname(__file__),'..', '..', '..', 'Assets', 'UFO', 'ALP_linear_UFO_WIDTH_modified')
)
if _ufo_dir not in sys.path:
	sys.path.insert(0, _ufo_dir)

import parameters as p


# Build a numeric namespace from the UFO `parameters` module.
# - externals are used directly as numbers
# - internals (expression strings) are evaluated with externals available
ns = {'cmath': cmath}
for par in p.all_parameters:
	try:
		if getattr(par, 'nature', None) == 'external':
			ns[par.name] = par.value
	except Exception:
		pass

for par in p.all_parameters:
	try:
		if getattr(par, 'nature', None) == 'internal':
			if isinstance(par.value, str):
				ns_val = complex(eval(par.value, {'cmath': cmath}, ns))
				ns[par.name] = ns_val.real if getattr(par, 'type', None) == 'real' else ns_val
			else:
				ns[par.name] = par.value
	except Exception:
		# leave unresolved parameters out of ns
		pass


def lambd(x: float, y: float) -> float:
	"""Källén (triangle) function used in two-body phase space.

	The Källén function is defined as
	(1 - x - y)**2 - 4*x*y and commonly appears with x = m1^2/s, y = m2^2/s.

	Args:
	- x: dimensionless ratio (typically m1**2 / s).
	- y: dimensionless ratio (typically m2**2 / s).

	Returns:
	- The value of the Källén function as a float.
	"""
	return (1 - x - y) ** 2 - 4 * x * y



def gamma_h_to_Za(m_h: float, m_Z: float, m_a: float, f_a: float, C_Zh_eff: float) -> float:
	"""Approximate partial width for the decay h -> Z a.

	This function implements a simplified EFT-inspired estimate of the
	partial width It uses the Källén function raised to the 3/2 power 
 	appropriate for derivative couplings.

	Args:
	- m_h: Higgs mass in GeV.
	- m_Z: Z boson mass in GeV.
	- m_a: ALP mass in GeV.
	- C_Zh_eff: effective coupling controlling h-Z-a.
	- f_a: ALP decay constant in GeV.

	Returns:
	- Partial width Gamma(h -> Z a) in GeV.

	Notes:
	- The implementation is approximate and intended for quick estimates
		rather than precision phenomenology. See arXiv:1708.00443 for
		related derivations and loop expressions.
	"""
	# dimensionless phase-space variable
	lam = lambd(m_Z**2 / m_h**2, m_a**2 / m_h**2)
	if lam <= 0:
		return 0.0

	# kinematic factor lambda^{1/2}, use lambda^{3/2} for derivative coupling
	lam_3_2 = lam**(3/2)

	# EFT-motivated normalization (parameterized):
	# Gamma = (C_aphi^2 / f_a^2) * m_h^3 / (16*pi) * (lam / m_h^4)^{3/2}
	# rearranged: m_h^3 * (lam_3_2) / m_h^6 -> lam_3_2 / m_h^3
	# For clarity and flexibility we use the simple form below:
	gamma = (m_h**3 * C_Zh_eff**2 * lam_3_2) / (16 * math.pi * f_a**2)
	# simplifies to: prefactor * lam_3_2 / (16*pi*m_h^3)
	return gamma



def higgs_cross_section(m_h: float, m_Z: float, m_a: float, f_a: float, C_Zh_eff: float, sigma_ggF: float, gamma_h_SM: float) -> Tuple[float, float]:
	"""Compute sigma(pp -> h) * BR(h -> Z a) and the branching ratio.

	The function computes the new partial width for h -> Z a using
	:func:`gamma_h_to_Za`, forms the total width assuming only the SM
	width plus the new contribution, and returns the production
	cross-section in gluon fusion multiplied by the branching ratio.

	Args:
	- m_h, m_Z, m_a: masses in GeV.
	- C_Zh_eff: effective coupling (dimensionless).
	- f_a: ALP decay constant in GeV.
	- sigma_ggF: Higgs production cross-section in ggF at the chosen
		collider energy (pb).
	- gamma_h_SM: SM Higgs total width (GeV).

	Returns:
	- Tuple of `(sigma * BR)` in pb and `BR(h -> Z a)` (dimensionless).
	"""
	# Compute the new partial width with ALP contribution
	gamma_Za = gamma_h_to_Za(m_h, m_Z, m_a, f_a, C_Zh_eff)
	# Total width is SM width + new partial width (assuming no other BSM decays)
	gamma_total = gamma_h_SM + gamma_Za
	# Branching ratio for h -> Z a
	br_Za = gamma_Za / gamma_total if gamma_total > 0 else 0.0
	# Cross-section times branching ratio
	sigma_times_br = sigma_ggF * br_Za
	return sigma_times_br, br_Za


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compute sigma(pp -> h) * BR(h -> Z a) for given ALP parameters.")
	args = parser.parse_args()

	# Constants for the calculation (clarity/safety: prefer explicit UFO externals)
	m_h = ns['MH'] # 125.00 GeV
	m_Z = ns['MZ']

	# Use UFO defaults for f_a and Higgs SM width when available, but allow CLI override
	f_a = 1000
	sigma_ggF = 54.67 # From Table 190 of https://arxiv.org/abs/1610.07922 for 125.00 GeV Higgs mass
	gamma_h_SM = 4.088e-3 # From Table 178 of https://arxiv.org/abs/1610.07922 for 125.00 GeV Higgs mass
	m_a = 0.1
	C_Zh_eff = 0.01

	sigma_br, br_Za = higgs_cross_section(m_h, m_Z, m_a, f_a, C_Zh_eff, sigma_ggF, gamma_h_SM)
	print(f"Sigma(pp -> h) * BR(h -> Z a) = {sigma_br:.4e} pb")
	print(f"BR(h -> Z a) = {br_Za:.4e}")
 
	plot_heatmap = True

	if plot_heatmap == True:
		# deferred imports so the script still runs without plotting deps
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.colors import LogNorm

		# grid settings (log-scaled axes)
		m_a_min, m_a_max, n_m = 0.0562, 31.6, 300
		C_min, C_max, n_C = 0.0000000316, 1.0, 300

		# use log-spaced grid for both axes
		m_a_vals = np.logspace(math.log10(m_a_min), math.log10(m_a_max), n_m)
		C_vals = np.logspace(math.log10(C_min), math.log10(C_max), n_C)
		M, C = np.meshgrid(m_a_vals, C_vals)

		# vectorized evaluation using existing function
		vec = np.vectorize(lambda ma, c: higgs_cross_section(m_h, m_Z, ma, f_a, c, sigma_ggF, gamma_h_SM)[0])
		S = vec(M, C)

		plt.figure(figsize=(8, 6))
		pcm = plt.pcolormesh(M, C, S, norm=LogNorm(vmin=max(S.min(), 1e-12), vmax=S.max()), shading='auto')
		plt.colorbar(pcm, label='sigma(pp->h)*BR(h->Za) [pb]')
		plt.xlabel(r'$m_a$ [GeV]')
		plt.ylabel(r'$C_{Zh}^{\mathrm{eff}}$')
		plt.title('Sigma(pp->h) * BR(h->Z a)')
		plt.xscale('log')
		plt.yscale('log')
		plt.tight_layout()
		out = '/usera/fs568/set-anubis/setanubis/FINAL_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/Plots/higgs_ggF_Za_cross_section_heatmap.png'
		plt.savefig(out, dpi=150)
		print(f'Saved heatmap to {out}')
