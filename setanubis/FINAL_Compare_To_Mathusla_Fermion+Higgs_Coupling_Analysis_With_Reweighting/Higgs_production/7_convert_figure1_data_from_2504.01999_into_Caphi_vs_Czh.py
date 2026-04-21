from __future__ import annotations

import pandas as pd
import numpy as np

import os
import sys
import cmath
import argparse

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

def calculate_couplings(input_csv, output_csv, BR_muon, Lambda_scale=1000.0, fa_scale=1000.0):
    # ==========================================
    # Constants (in GeV unless stated otherwise)
    # ==========================================
    m_h = ns['MH']      # Higgs mass (from UFO parameters)
    m_Z = ns['MZ']      # Z boson mass (from UFO parameters)
    m_a = 15.0          # ALP mass (from MATHUSLA plot)
    m_mu = 0.10566          # Muon mass (mass of b-quark from UFO)
    
    # Standard Model Higgs total width (approx 4.07 MeV)
    Gamma_SM_H = 4.07e-3
    
    # Conversion factor: hbar * c in GeV * m
    hbar_c = 1.973269804e-16
    
    # Scaling factor derived for BR(h -> Za) based on multiplicity and kinematics
    BR_scaling_factor = 2 * ( (m_h**2 - m_Z**2 + m_a**2)/(2*m_h) ) * ( m_h / 2)**(-1)
    
    # ==========================================
    # Load Data
    # ==========================================
    # Assuming the input CSV has columns 'ctau_m' and 'br_hxx'
    df = pd.read_csv(input_csv)

    # Normalize column names (strip stray spaces) and verify required columns
    df.columns = df.columns.str.strip()
    required = ['ctau_m', 'br_hxx']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}. Available columns: {list(df.columns)}")

    # Ensure data is numeric
    ctau = pd.to_numeric(df['ctau_m'], errors='coerce')
    br_hxx = pd.to_numeric(df['br_hxx'], errors='coerce')
    
    # ==========================================
    # 1. Calculate C_zh (ALP-Higgs coupling)
    # ==========================================
    # Scale the branching ratio for h -> Za
    br_hZa = br_hxx * BR_scaling_factor / BR_muon
    
    # Calculate partial width Gamma(h -> Za)
    # BR = Gamma_Za / (Gamma_SM_H + Gamma_Za)  =>  Gamma_Za = (BR * Gamma_SM_H) / (1 - BR)
    Gamma_hZa = (br_hZa * Gamma_SM_H) / (1.0 - br_hZa)
    
    # Källén function for phase space: lambda(x, y) = (1-x-y)^2 - 4xy
    x = (m_Z / m_h)**2
    y = (m_a / m_h)**2
    kallen_lambda = (1 - x - y)**2 - 4 * x * y
    
    # Isolate |C_zh^eff| / Lambda from the Gamma(h -> Za) equation
    # Gamma(h -> Za) = (m_h^3 / 16*pi) * (|C_zh|/Lambda)^2 * lambda^(3/2)
    C_zh_over_Lambda_squared = (16 * np.pi * Gamma_hZa) / ( (m_h**3) * (kallen_lambda**(1.5)) )
    C_zh_over_Lambda = np.sqrt(C_zh_over_Lambda_squared)
    
    # ==========================================
    # 2. Calculate C_aphi (ALP-fermion coupling)
    # ==========================================
    # Convert c*tau (meters) to decay width Gamma(a -> bb) (GeV)
    Gamma_total = hbar_c / ctau
    Gamma_mu = BR_muon * Gamma_total
    
    # Phase space factor for a -> mu+ mu-
    phase_space_a = np.sqrt(1.0 - (4.0 * m_mu**2 / m_a**2))
    
    # Isolate C_aphi / f_a from the Gamma(a -> mu+ mu-) equation
    # Gamma(a -> mu+ mu-) = (m_a * m_mu^2 / 8*pi) * (C_aphi / f_a)^2 * sqrt(1 - 4m_mu^2/m_a^2)
    C_aphi_over_fa_squared = (8 * np.pi * Gamma_mu) / (m_a * (m_mu**2) * phase_space_a)
    C_aphi_over_fa = np.sqrt(C_aphi_over_fa_squared)
    
    # ==========================================
    # Export dimensionless couplings
    # ==========================================
    # Multiply by the energy scales to get dimensionless coefficients
    C_zh = C_zh_over_Lambda * Lambda_scale
    C_aphi = C_aphi_over_fa * fa_scale

    # Create output dataframe with only the computed coupling columns
    out_df = pd.DataFrame({'x': C_zh, 'y': C_aphi})

    # Save to a new CSV
    out_df.to_csv(output_csv, index=False)
    print(f"Data successfully processed and saved to {output_csv}")
    print(f"Assumed Lambda = {Lambda_scale} GeV, f_a = {fa_scale} GeV")

# Example usage:
def main():
    parser = argparse.ArgumentParser(description="Convert Figure1 CSV into C_aphi vs C_zh values")
    default_input = os.path.join(os.path.dirname(__file__), 'Figure1_from_2504.01999_data.csv')
    default_output = os.path.join(os.path.dirname(__file__), 'MATHUSLA40.csv')
    parser.add_argument('-i', '--input', default=default_input, help='Path to input CSV file')
    parser.add_argument('-o', '--output', default=default_output, help='Path to output CSV file')
    parser.add_argument('--Lambda_scale', type=float, default=1000.0, help='Lambda scale in GeV')
    parser.add_argument('--fa_scale', type=float, default=1000.0, help='f_a scale in GeV')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Input file not found: {args.input}\nPlace the file at this path or specify a different path with --input.")

    # Produce CSVs for three muon branching ratios: 1.0, 0.1, 0.001
    ratios = [
        (1.0, '1'),
        (0.1, '0.1'),
        (0.001, '0.001'),
    ]

    for br_val, label in ratios:
        out_file = os.path.join(os.path.dirname(__file__), f"MATHUSLA40_BR_{label}.csv")
        print(f"Processing BR_muon={br_val} -> writing {out_file}")
        calculate_couplings(args.input, out_file, br_val, Lambda_scale=args.Lambda_scale, fa_scale=args.fa_scale)


if __name__ == "__main__":
    main()