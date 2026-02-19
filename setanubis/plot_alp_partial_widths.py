#!/usr/bin/env python3
"""Plot ALP partial widths and proper lifetime vs mass using UFO formulas.

This script evaluates the analytic ALP partial-width expressions provided
by the project's UFO model (Assets/UFO/ALP_linear_UFO_WIDTH) through the
`Width_Calculator.py` adapter and the `SetAnubis` interfaces. It produces
two plots saved to the `setanubis/` directory:

- `alp_partial_widths_vs_mass_symlog.pdf`: partial widths for each decay
    channel (symlog Y axis, log X axis) with kinematic thresholds marked.
- `alp_ctau_vs_mass.pdf`: ALP proper decay length `c·tau` (meters) vs mass
    (log-log) with example detector radii overlayed.

Default model settings in the script:
- Couplings: `CaPhi=0.001`, `CGtil=0.0`, `CWtil=0.0`, `CBtil=0.0`.
- Decay constant: `fa=1000.0` GeV.
- Mass grid: logspace from 0.1 to 1000 GeV (500 points).

Quick usage:
    python3 setanubis/plot_alp_partial_widths.py

To change couplings or other parameters, edit the `sa.set_leaf_param(...)`
calls in the `main()` function. Thresholds are read from the UFO parameters
and drawn at `2*m_i` for relevant SM masses.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from SetAnubis.core.BranchingRatio.adapters.input.DecayInterface import DecayInterface, CalculationDecayStrategy
from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UFO_PATH = os.path.join(CURRENT_DIR, '..', 'Assets', 'UFO', 'ALP_linear_UFO_WIDTH')
PY_SCRIPT_PATH = os.path.join(CURRENT_DIR, 'Width_Calculator.py')


def main():
    out_dir = os.path.join(CURRENT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    sa = SetAnubisInterface(UFO_PATH)
    sa.set_leaf_param('ZERO', 0)

    # -- user-requested coupling choices --
    # Set CaPhi = 0.001 and zero the SM effective couplings
    # Keep fa at 1000 GeV (UFO default) unless overridden
    sa.set_leaf_param('CaPhi', 0.001)
    sa.set_leaf_param('CGtil', 0.0)
    sa.set_leaf_param('CWtil', 0.0)
    sa.set_leaf_param('CBtil', 0.0)
    sa.set_leaf_param('fa', 1000.0)

    br = DecayInterface(sa)

    # register decay calculators using the provided Width_Calculator
    alp_decay_list = [
        {"mother": 9000005, "daughters": [22, 22]},     # γγ
        {"mother": 9000005, "daughters": [22, 23]},     # γZ
        {"mother": 9000005, "daughters": [5, -5]},      # bb
        {"mother": 9000005, "daughters": [4, -4]},      # cc
        {"mother": 9000005, "daughters": [1, -1]},      # dd
        {"mother": 9000005, "daughters": [11, -11]},    # ee
        {"mother": 9000005, "daughters": [21, 21]},     # gg
        {"mother": 9000005, "daughters": [13, -13]},    # mumu
        {"mother": 9000005, "daughters": [3, -3]},      # ss
        {"mother": 9000005, "daughters": [6, -6]},      # tt
        {"mother": 9000005, "daughters": [15, -15]},    # tautau
        {"mother": 9000005, "daughters": [2, -2]},      # uu
        {"mother": 9000005, "daughters": [-24, 24]},    # WW
        {"mother": 9000005, "daughters": [23, 23]},     # ZZ
    ]

    br.add_decays(alp_decay_list, CalculationDecayStrategy.PYTHON, config={"script_path": PY_SCRIPT_PATH})

    channels = [
        ("gamma_gamma", [22, 22]),
        ("gamma_Z", [22, 23]),
        ("bb", [5, -5]),
        ("cc", [4, -4]),
        ("dd", [1, -1]),
        ("ee", [11, -11]),
        ("gg", [21, 21]),
        ("mumu", [13, -13]),
        ("ss", [3, -3]),
        ("tt", [6, -6]),
        ("tautau", [15, -15]),
        ("uu", [2, -2]),
        ("WW", [-24, 24]),
        ("ZZ", [23, 23]),
    ]

    # mass grid (GeV) - from 0.1 GeV to 1000 GeV on a log grid
    # reduced density to speed up runs
    masses = np.logspace(-1, 3, 500)

    results = {name: [] for (name, _) in channels}

    for m in masses:
        # set ALP mass parameter 'Ma' in the model
        sa.set_leaf_param('Ma', float(m))

        for name, daughters in channels:
            try:
                val = br.get_decay(9000005, daughters).real
            except Exception:
                val = 0.0
            # ensure non-negative float
            try:
                results[name].append(float(val))
            except Exception:
                results[name].append(0.0)

    # compute total width (sum of partials)
    total = np.zeros_like(masses)
    for name in results:
        total += np.array(results[name])

    # --- compute kinematic thresholds from UFO mass parameters ---
    # We'll draw vertical lines at 2*m for relevant SM masses (include light quarks)
    threshold_params = ['Me', 'MMU', 'MTA', 'MU', 'MD', 'MC', 'MS', 'MB', 'MW', 'MZ', 'MT']
    thresholds = {}
    for p in threshold_params:
        try:
            pv = sa.get_parameter_value(p)
            # pv is often a complex-like number, take real part
            val = float(pv.real) if hasattr(pv, 'real') else float(pv)
            thresholds[p] = 2.0 * val
        except Exception:
            # ignore missing parameters
            pass

    # --- Partial widths plot (symlog Y, log X) ---
    plt.figure(figsize=(10, 6))
    param_label = {'Me':'e', 'MMU':'mu', 'MTA':'tau', 'MU':'u', 'MD':'d', 'MC':'c', 'MS':'s', 'MB':'b', 'MW':'W', 'MZ':'Z', 'MT':'t'}
    for name in results:
        arr = np.array(results[name])
        if np.all(arr <= 0):
            continue
        plt.plot(masses, arr, label=name, alpha=0.7)

    # log x, symlog y to show linear behavior near zero and log behavior elsewhere
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-23)

    # total width (plotted on same axes)
    plt.plot(masses, total, label='total', color='k', lw=2.5)

    plt.xlabel('ALP mass $m_a$ [GeV]')
    plt.ylabel('Partial width [GeV] (symlog y)')
    plt.title('ALP partial widths vs mass (symlog y, log x)', pad=14)
    # place legend inside the plot near the top-left but shifted right
    # so it does not overlap the low-mass thresholds (muon/strange)
    plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(0.15, 0.98))
    plt.grid(which='both', linestyle='--', alpha=0.25)

    # thresholds on symlog plot
    ax = plt.gca()
    y_top = ax.get_ylim()[1]
    for pname, thr in thresholds.items():
        if thr >= masses[0] and thr <= masses[-1]:
            ax.axvline(thr, color='gray', linestyle='--', alpha=0.6)
            lab = param_label.get(pname, pname)
            ax.text(thr * 1.001, y_top * 0.9, f"2*{lab}", rotation=90, color='gray', fontsize=8, va='top', ha='left')

    out_pdf_pw_symlog = os.path.join(out_dir, 'alp_partial_widths_vs_mass_symlog.pdf')
    plt.tight_layout()
    plt.savefig(out_pdf_pw_symlog, bbox_inches='tight')
    print(f"Wrote: {out_pdf_pw_symlog}")

    # --- Proper lifetime c*tau (meters) plot ---
    # hbar*c in GeV*m: hbar (GeV*s) * c (m/s)
    hbar_GeV_s = 6.582119569e-25
    c_ms = 299792458.0
    hbar_c_GeV_m = hbar_GeV_s * c_ms

    # avoid zero or negative total widths
    total_pos = np.where(total <= 0, np.nan, total)
    ctau_m = hbar_c_GeV_m / total_pos

    plt.figure(figsize=(10, 6))
    plt.loglog(masses, np.maximum(ctau_m, 1e-30), label='c·tau (m)', color='C0')

    # overlay detector radii (example: inner detector/cavern)
    detector_radii_m = {'InnerDetector': 1.2, 'MS': 9.5, 'Cavern': 20.0}
    for name, r in detector_radii_m.items():
        plt.axhline(r, color='gray', linestyle='--', alpha=0.6)
        plt.text(masses[3], r * 1.1, name, color='gray')

    plt.xlabel('ALP mass $m_a$ [GeV]')
    plt.ylabel('c·tau [m]')
    plt.title('ALP proper decay length c·tau vs mass', pad=14)
    plt.grid(which='both', linestyle='--', alpha=0.25)
    plt.legend()

    # vertical lines at thresholds on c*tau plot
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    for pname, thr in thresholds.items():
        if thr >= masses[0] and thr <= masses[-1]:
            ax.axvline(thr, color='gray', linestyle='--', alpha=0.5)
            lab = param_label.get(pname, pname)
            ax.text(thr * 1.001, ymax * 0.9, f"2*{lab}", rotation=90, color='gray', fontsize=8, va='top', ha='left')

    out_pdf_ctau = os.path.join(out_dir, 'alp_ctau_vs_mass.pdf')
    plt.tight_layout()
    plt.savefig(out_pdf_ctau)
    print(f"Wrote: {out_pdf_ctau}")

    # compute masses where c*tau crosses detector radii
    print('\nMass where c·tau <= detector radius (first mass):')
    hdr = f"{'detector':15} {'radius[m]':>10} {'mass[GeV]':>12}"
    print(hdr)
    print('-' * len(hdr))
    for name, r in detector_radii_m.items():
        idx = np.where(ctau_m <= r)[0]
        mass_cross = float(masses[idx[0]]) if idx.size else float('nan')
        print(f"{name:15} {r:10.3g} {mass_cross:12.4g}")


if __name__ == '__main__':
    main()
