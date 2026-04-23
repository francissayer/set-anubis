"""
Plot layered sensitivity contours for multiple C_Zh^eff couplings.

This script computes expected signal events dynamically for specified
values of the effective coupling C_Zh^eff (1.0, 0.1, 0.015) and plots
their target sensitivity contours (e.g., 4 events) on a single plot
with different linestyles. It bypasses writing intermediate data to CSV.

Outputs: layered_sensitivity_contours.png and .pdf
"""

import os
import sys
import glob
import argparse
import math
import cmath
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import importlib.util

try:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull
except ImportError:
    LinearNDInterpolator = None
    gaussian_filter = None
    ConvexHull = None

import mplhep as hep
plt.style.use(hep.style.ATLAS)


# ====================================================================
# Event Calculation Helpers (From Script 5)
# ====================================================================

def load_module_from_path(name: str, path: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ufo_namespace(ufo_dir: str):
    """Builds a dictionary of UFO parameters by evaluating internal dependencies."""
    if ufo_dir not in sys.path:
        sys.path.insert(0, ufo_dir)
    import parameters as p

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
            pass
    return ns


def compute_alp_brs_for_point(ns_template: dict, channels):
    """Calculates ALP branching ratios for a given mass/coupling point."""
    channel_map = {
        'bb': ('b', 3), 'cc': ('c', 3), 'dd': ('d', 3),
        'ee': ('e', 1), 'mumu': ('mu', 1), 'ss': ('s', 3),
        'tata': ('tau', 1), 'uu': ('u', 3),
    }

    Ma = float(ns_template['Ma'])
    Ca = float(ns_template.get('CaPhi', ns_template.get('Ca')))
    fa = float(ns_template.get('fa', ns_template.get('FA', 1000.0)))

    vgev = 246.22056907348585
    sm_m = {
        'd': 0.00504, 'u': 0.00255, 's': 0.101, 'c': 1.27,
        'b': 4.7, 't': 172.0, 'e': 0.000511, 'mu': 0.10566, 'tau': 1.777,
    }

    gammas = {}
    total = 0.0
    for ch in channels:
        mapping = channel_map.get(ch)
        if mapping is None:
            gammas[ch] = 0.0
            continue
        pname, color = mapping
        m_f = sm_m.get(pname, None)
        if m_f is None or Ma <= 2.0 * m_f:
            gammas[ch] = 0.0
            continue

        y_f = math.sqrt(2.0) * m_f / vgev
        sqrt_term = math.sqrt(max(0.0, Ma * Ma - 4.0 * m_f * m_f))
        prefactor = ((3.0 if color == 3 else 1.0) * (Ca * Ca) * (vgev * vgev) * (y_f * y_f))
        gamma_f = prefactor * sqrt_term / (16.0 * math.pi * (fa * fa))
        gammas[ch] = float(max(gamma_f, 0.0))
        total += gammas[ch]

    brs = {ch: (gammas.get(ch, 0.0) / total if total > 0 else 0.0) for ch in channels}
    return brs, gammas, total


def load_channel_acceptances(csv_path: str):
    """Extracts acceptance (N_final/N_orig) from cutflow CSVs."""
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()

    if 'nLLP_original' in df.columns and 'nLLP_Final' in df.columns:
        orig_col, final_col = 'nLLP_original', 'nLLP_Final'
    elif 'nLLP_original' in df.columns and 'n_surviving_llps' in df.columns:
        orig_col, final_col = 'nLLP_original', 'n_surviving_llps'
    else:
        return {}

    grouped = df.groupby(['mass', 'CaPhi'])[[orig_col, final_col]].sum().reset_index()
    acc_map = {}
    for _, row in grouped.iterrows():
        orig, final = float(row[orig_col]), float(row[final_col])
        acc_map[(float(row['mass']), float(row['CaPhi']))] = (final / orig) if orig > 0 else 0.0

    return acc_map

# ====================================================================
# Interpolation Helper (From Script 6)
# ====================================================================

def interpolate_grid(mass_vals, caphi_vals, Z, smooth_sigma=0.0):
    """Performs log-linear interpolation over the mass-coupling grid."""
    if LinearNDInterpolator is None:
        raise RuntimeError('scipy.interpolate.LinearNDInterpolator is required.')

    x = np.asarray(mass_vals, dtype=float)
    y = np.asarray(caphi_vals, dtype=float)
    xlog = np.log10(x)
    ylog = np.log10(y)

    Z2 = np.array(Z, dtype=float)
    positive_mask = Z2 > 0
    
    if np.any(positive_mask):
        min_pos = float(np.min(Z2[positive_mask]))
        max_pos = float(np.max(Z2[positive_mask]))
        eps = min_pos * 1e-3
    else:
        eps, max_pos = 1e-15, 1e-15
        
    Z_safe = np.where(Z2 > 0, Z2, eps)
    zlog = np.log10(Z_safe)

    XX, YY = np.meshgrid(xlog, ylog)
    pts = np.column_stack((XX.ravel(), YY.ravel()))
    vals = zlog.ravel()

    finite_mask = np.isfinite(vals)
    pts_f = pts[finite_mask]
    vals_f = vals[finite_mask]

    interp = LinearNDInterpolator(pts_f, vals_f)

    nx_grid, ny_grid = 1000, 1000 # Higher resolution for contours
    LOGX = np.linspace(xlog.min(), xlog.max(), nx_grid)
    LOGY = np.linspace(ylog.min(), ylog.max(), ny_grid)
    LOGGX, LOGGY = np.meshgrid(LOGX, LOGY)
    eval_pts = np.column_stack((LOGGX.ravel(), LOGGY.ravel()))
    GZ_log = interp(eval_pts).reshape(LOGGX.shape)

    if ConvexHull is not None and pts_f.shape[0] >= 3:
        try:
            hull = ConvexHull(pts_f)
            hull_path = MplPath(pts_f[hull.vertices])
            inside = hull_path.contains_points(eval_pts)
            GZ_log_flat = GZ_log.ravel()
            GZ_log_flat[~inside] = np.nan
            GZ_log = GZ_log_flat.reshape(GZ_log.shape)
        except Exception:
            pass

    if gaussian_filter is not None and smooth_sigma > 0:
        med = np.nanmedian(GZ_log)
        filled = np.where(np.isfinite(GZ_log), GZ_log, med if np.isfinite(med) else 0.0)
        smoothed = gaussian_filter(filled, sigma=smooth_sigma, mode='nearest')
        smoothed[~np.isfinite(GZ_log)] = np.nan
        GZ_log = smoothed

    GZ = np.where(np.isfinite(GZ_log), 10.0 ** GZ_log, np.nan)
    GX, GY = np.meshgrid(10.0 ** LOGX, 10.0 ** LOGY)

    return GX, GY, GZ


# ====================================================================
# Main Script
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot layered sensitivity contours for varied C_Zh_eff')
    parser.add_argument('--higgs-dir', type=str,
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production',
                        help='Base Higgs_production directory')
    parser.add_argument('--ufo-dir', type=str,
                        default='/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified',
                        help='UFO directory providing decays/parameters')
    parser.add_argument('--cross-section-script', type=str,
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/4_Higgs_ggF_Zax_Cross_Section.py',
                        help='Path to Higgs cross-section helper')
    parser.add_argument('--output-dir', type=str, 
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/Plots', 
                        help='Where to save the plot')
    parser.add_argument('--luminosity', type=float, default=3000.0)
    parser.add_argument('--selection-eff', type=float, default=0.5)
    parser.add_argument('--br-z-visible', type=float, default=0.8)
    parser.add_argument('--sigma-ggf', type=float, default=54.61)
    parser.add_argument('--gamma-h-sm', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=15.0, help='Gaussian smoothing sigma')
    parser.add_argument('--target-events', type=float, default=4.0, help='Contour level (default: 4.0 events)')
    args = parser.parse_args()

    print('Building UFO parameter namespace...')
    ns = build_ufo_namespace(args.ufo_dir)

    print('Loading Higgs cross-section helper...')
    higgs_mod = load_module_from_path('higgs_xsec', args.cross_section_script)

    base = Path(args.higgs_dir)
    chans = [p.name.replace('_Decay_Channel', '') for p in sorted(base.iterdir()) 
             if p.is_dir() and p.name.endswith('_Decay_Channel')]
    
    print('Loading acceptances for channels:', chans)
    channel_acceptances = {}
    keys = set()
    for ch in chans:
        folder = base / f"{ch}_Decay_Channel"
        files = glob.glob(str(folder / 'selection_cutflow_*_decay_channel.csv')) or glob.glob(str(folder / 'selection_cutflow_*.csv'))
        if files:
            acc = load_channel_acceptances(files[0])
            channel_acceptances[ch] = acc
            keys.update(acc.keys())

    if not keys:
        raise RuntimeError('No grid points found. Ensure selection CSVs exist.')
    
    keys = sorted(keys)
    mass_vals = np.sort(np.unique([k[0] for k in keys]))
    caphi_vals = np.sort(np.unique([k[1] for k in keys]))
    
    target_couplings = [1.0, 0.1, 0.015]
    linestyles = {1.0: 'solid', 0.1: 'dashed', 0.015: 'dotted'}
    
    gamma_h_sm = float(args.gamma_h_sm) if args.gamma_h_sm else float(ns.get('WH', 0.00407))
    lumi_pb = args.luminosity * 1000.0
    
    grids = {czh: np.full((len(caphi_vals), len(mass_vals)), np.nan) for czh in target_couplings}
    
    print('Calculating expected events for dynamically generated grids...')
    for (m_a, caphi) in keys:
        ns_point = dict(ns)
        ns_point['Ma'] = m_a
        ns_point['CaPhi'] = caphi
        brs, _, _ = compute_alp_brs_for_point(ns_point, chans)
        
        acceptance = 0.0
        for ch in chans:
            acceptance += brs.get(ch, 0.0) * channel_acceptances.get(ch, {}).get((m_a, caphi), 0.0)
            
        m_h, m_Z = float(ns_point['MH']), float(ns_point['MZ'])
        f_a = float(ns_point.get('fa', ns_point.get('FA', 1000.0)))
        
        i, j = np.searchsorted(caphi_vals, caphi), np.searchsorted(mass_vals, m_a)

        for czh in target_couplings:
            sigma_pb, br_h_za = higgs_mod.higgs_cross_section(m_h, m_Z, m_a, f_a, czh, args.sigma_ggf, gamma_h_sm)
            N_signal = lumi_pb * sigma_pb * acceptance * args.selection_eff * args.br_z_visible
            grids[czh][i, j] = N_signal

    print('Generating layered contours plot...')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot base grid points
    mass_arr, caphi_arr = np.meshgrid(mass_vals, caphi_vals)
    ax.scatter(mass_arr.flatten(), caphi_arr.flatten(), facecolors='none', edgecolors='lightgrey', 
               s=120, linewidths=0.8, zorder=2)

    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                             markeredgecolor='lightgrey', markersize=10, label='Grid Points')]
    
    for czh in target_couplings:
        try:
            GX, GY, GZ = interpolate_grid(mass_vals, caphi_vals, grids[czh], args.sigma)
            cs = ax.contour(GX, GY, GZ, levels=[args.target_events], colors='red', 
                            linestyles=linestyles[czh], linewidths=3.0, zorder=70)
            
            # Matplotlib 3.8+ compatibility for path effects
            pe = [path_effects.withStroke(linewidth=6, foreground='white')]
            if hasattr(cs, 'collections'):
                for coll in cs.collections:
                    coll.set_path_effects(pe)
            else:
                # Newer Matplotlib versions
                for line in cs.legend_elements()[0]:
                    line.set_path_effects(pe)
            
            ax.contourf(GX, GY, GZ, levels=[args.target_events, np.inf], colors=['red'], alpha=0.08, zorder=50)
            legend_handles.append(Line2D([0], [0], color='red', lw=3.0, linestyle=linestyles[czh], 
                                         label=f'$C_{{Zh}}^{{eff}} = {czh}$'))
                                         
        except Exception as e:
            print(f'Warning: failed to draw contour for C_Zh^eff = {czh}: {e}')

    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower left', 
                  title=f'Sensitivity ({args.target_events} events)', framealpha=0.9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass [GeV]', fontsize=14)
    ax.set_ylabel(r'Coupling $C_{a\Phi}$', fontsize=14)
    ax.set_title(f'Layered Sensitivity: $pp\\to H \\to Z a$ ({args.target_events} Events)', fontsize=16)
    plt.tight_layout()

    out_path = Path(args.output_dir) / 'layered_sensitivity_contours.png'
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(out_path.with_suffix('.pdf')), bbox_inches='tight')
    plt.close()
    print(f'Saved layered plots to {out_path} and .pdf')

if __name__ == '__main__':
    main()