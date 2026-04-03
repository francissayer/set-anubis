"""
pp -> Z a production: plot heat map of expected signal events vs ALP mass and coupling CaPhi.

This script computes N_signal for pp -> Z a, where the ALP (a)
may decay into several channels. The acceptance for each channel is read
from that channel's selection cutflow CSV; the overall geometric/kinematic
acceptance is computed as a BR-weighted average over channels:

  acceptance_overall = sum_i BR(a->i) * (N_{i,sel} / N_{i,gen})

Cross-section σ(pp→Z a) values are read from precomputed scan files
(`scan_run_*.txt`). ALP partial widths / branching ratios are computed
using the analytic fermionic partial-width formula used in the per-channel
extraction scripts (see `*_Decay_Channel/2_extract_selection_data.py`).

Outputs: three PNG heatmaps and an optional extended CSV with per-point numbers.
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
import importlib.util
import re
from collections import defaultdict


def load_module_from_path(name: str, path: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ufo_namespace(ufo_dir: str):
    """Construct numeric namespace from UFO `parameters` module.

    Returns a dict mapping parameter names -> numeric values (floats/complex).
    """
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
                    # evaluate expression with existing ns available
                    ns_val = complex(eval(par.value, {'cmath': cmath}, ns))
                    ns[par.name] = ns_val.real if getattr(par, 'type', None) == 'real' else ns_val
                else:
                    ns[par.name] = par.value
        except Exception:
            # leave unresolved parameters out of ns
            pass

    return ns


def parse_scan_cross_section_files(scan_dir: str, pattern: str = 'scan_run_*.txt'):
    """Parse scan files in `scan_dir` matching `pattern`.

    Returns a mapping keyed by (rounded_mass, rounded_CaPhi) -> dict with
    key 'sigma_pb' (float).
    """
    files = sorted(glob.glob(os.path.join(scan_dir, pattern)))
    if not files:
        raise RuntimeError(f"No scan cross-section files found at {os.path.join(scan_dir, pattern)}")

    temp = defaultdict(lambda: {'sigma': []})

    for fname in files:
        try:
            with open(fname, 'r') as fh:
                lines = [l.strip() for l in fh]

            # Extract header to dynamically map columns
            mass_idx, caphi_idx, sigma_idx = -1, -1, -1
            
            for line in lines[:10]: # Check top lines for headers
                if line.startswith('#') and ('mass' in line.lower() or 'alppar' in line.lower() or 'cross' in line.lower()):
                    clean_header = re.sub(r'^#\s*', '', line).strip()
                    header_tokens = re.split(r'\s+|,|\t', clean_header)
                    for idx, h in enumerate(header_tokens):
                        hlow = h.lower()
                        if 'mass#9000005' in hlow or hlow == 'mass' or hlow.startswith('mass#'):
                            mass_idx = idx
                        elif re.search(r'alppars[#\(\[]?5', hlow):
                            caphi_idx = idx
                        elif 'cross' in hlow or 'sigma' in hlow:
                            sigma_idx = idx
                    if mass_idx != -1 and caphi_idx != -1 and sigma_idx != -1:
                        break
            
            for i, line in enumerate(lines):
                if not line or line.startswith('#'):
                    continue

                # 1. Parse table-style 'run_XX' rows
                if line.startswith('run_'):
                    parts = line.split()
                    try:
                        if mass_idx != -1 and caphi_idx != -1 and sigma_idx != -1:
                            m_m = float(parts[mass_idx])
                            m_c = float(parts[caphi_idx])
                            m_s = float(parts[sigma_idx])
                        else:
                            # Fallback to standard MG5 order: run_name alppars mass ... cross error
                            m_c = float(parts[1])
                            m_m = float(parts[2])
                            m_s = float(parts[-2]) # Second to last is usually cross-section
                        
                        key = (round(m_m, 8), round(m_c, 8))
                        temp[key]['sigma'].append(m_s)
                        continue
                    except (IndexError, ValueError) as e:
                        raise RuntimeError(f"Unrecognized table format in {fname} at line {i+1}: '{line}'. Error: {e}")

                # 2. Try inline key=value parsing
                kv_pairs = dict(re.findall(r'(\S+?)\s*[:=]\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)', line))
                if kv_pairs:
                    m_m, m_c, m_s = None, None, None
                    for keyname in ('mass#9000005', 'mass'):
                        if keyname in kv_pairs:
                            m_m = float(kv_pairs[keyname])
                            break
                    for keyname in kv_pairs.keys():
                        if re.search(r'alppars[#\(\[]?5', keyname, re.I):
                            m_c = float(kv_pairs[keyname])
                            break
                    for keyname in ('cross', 'sigma', 'xsec', 'cross_section'):
                        if keyname in kv_pairs:
                            m_s = float(kv_pairs[keyname])
                            break

                    if m_m is not None and m_c is not None and m_s is not None:
                        key = (round(m_m, 8), round(m_c, 8))
                        temp[key]['sigma'].append(m_s)
                        continue

                # 3. Explicit Named Field Fallback
                mass_field_re = re.search(r"(?:mass#9000005|mass)\s*[:=]\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", line, re.I)
                alppars_field_re = re.search(r"(?:alppars[#\(\[]?5[\)\]]?)\s*[:=]\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", line, re.I)
                sigma_re = re.search(r"\b(?:cross|sigma|xsec|xsec_pb|cross_section|cross-section)\b\s*[:=]?\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", line, re.I)
                
                if mass_field_re and alppars_field_re and sigma_re:
                    mval = float(mass_field_re.group(1))
                    cval = float(alppars_field_re.group(1))
                    sval = float(sigma_re.group(1))
                    key = (round(mval, 8), round(cval, 8))
                    temp[key]['sigma'].append(sval)
                    continue

                # Final guard to report genuinely unrecognized lines
                float_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
                floats = float_re.findall(line)
                if len(floats) >= 3:
                    raise RuntimeError(f"Unrecognized data line in {fname} at line {i+1}: '{line}'. Expected header with columns or key=value pairs.")

        except Exception as e:
            raise RuntimeError(f"Error reading {fname}: {e}")

    # Average collected values
    mapping = {}
    for k, d in temp.items():
        if d['sigma']:
            mapping[k] = {
                'sigma_pb': float(np.mean(d['sigma']))
            }

    if not mapping:
        raise RuntimeError(f'No valid (mass, CaPhi, sigma) entries parsed from files in {scan_dir}')

    return mapping


def lookup_scan_sigma_br(scan_map: dict, mass: float, caphi: float):
    """Lookup sigma from scan_map, with nearest-key fallback.

    Returns sigma_pb (float) for the exact key; raises KeyError otherwise.
    """
    if not scan_map:
        raise RuntimeError('Scan map is empty; no cross-section data available. Check scan files and path provided to --scan-cross-section-dir')

    key = (round(float(mass), 8), round(float(caphi), 8))
    if key in scan_map:
        v = scan_map[key]
        return v.get('sigma_pb')

    # Do not fallback silently — raise an informative error
    available_keys = list(scan_map.keys())
    # find closest key just for diagnostic message
    best_key = min(available_keys, key=lambda k: abs(k[0] - mass) + abs(k[1] - caphi))
    best_dist = abs(best_key[0] - mass) + abs(best_key[1] - caphi)
    unique_masses = sorted(set(k[0] for k in available_keys))
    unique_caphis = sorted(set(k[1] for k in available_keys))
    raise KeyError(f"Exact scan key {(mass, caphi)} not found in scan data. Closest available key {best_key} (dist={best_dist}).\nAvailable mass points: {unique_masses[:10]}{'...' if len(unique_masses)>10 else ''}\nAvailable coupling points: {unique_caphis[:10]}{'...' if len(unique_caphis)>10 else ''}\nPlease ensure your scan files contain the exact (mass, alppars#5) grid points used by the selection CSVs.")


def compute_alp_brs_for_point(ns_template: dict, channels):
    """Compute ALP partial widths and BRs using the analytic fermionic formula

    Implements the same formula used in `*_Decay_Channel/2_extract_selection_data.py`:

        gamma_f = color_factor * Ca^2 * v^2 * y_f^2 * sqrt(Ma^2 - 4 m_f^2) / (16*pi*fa^2)

    where y_f = sqrt(2) * m_f / v.

    Returns (brs_dict, gammas_dict, total_width).
    """
    # Map our channel-folder names to (param_key, color_factor)
    channel_map = {
        'bb': ('b', 3),
        'cc': ('c', 3),
        'dd': ('d', 3),
        'ee': ('e', 1),
        'mumu': ('mu', 1),
        'ss': ('s', 3),
        'tata': ('tau', 1),
        'uu': ('u', 3),
    }

    # Require essential parameters to avoid silent fallbacks
    if 'Ma' not in ns_template:
        raise RuntimeError("Missing required parameter 'Ma' for ALP mass in namespace")
    if 'CaPhi' not in ns_template and 'Ca' not in ns_template:
        raise RuntimeError("Missing required coupling 'CaPhi' (or 'Ca') in namespace")
    # Expect explicit axion decay constant present; do not assume default
    if 'fa' in ns_template:
        fa = float(ns_template['fa'])
    elif 'FA' in ns_template:
        fa = float(ns_template['FA'])
    else:
        raise RuntimeError("Missing required parameter 'fa' (axion decay constant) in namespace; do not fallback to arbitrary values")

    # Values provided explicitly
    Ma = float(ns_template['Ma'])
    Ca = float(ns_template.get('CaPhi', ns_template.get('Ca')))

    # Use explicit SM constants (do not fallback to potentially inconsistent UFO values)
    vgev = 246.22056907348585
    sm_m = {
        'd': 0.00504,
        'u': 0.00255,
        's': 0.101,
        'c': 1.27,
        'b': 4.7,
        't': 172.0,
        'e': 0.000511,
        'mu': 0.10566,
        'tau': 1.777,
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
        if m_f is None:
            gammas[ch] = 0.0
            continue

        # Kinematic check: require open phase space
        if Ma <= 2.0 * m_f:
            gammas[ch] = 0.0
            continue

        # Yukawa relation
        y_f = math.sqrt(2.0) * m_f / vgev
        sqrt_term = math.sqrt(max(0.0, Ma * Ma - 4.0 * m_f * m_f))
        prefactor = ((3.0 if color == 3 else 1.0) * (Ca * Ca) * (vgev * vgev) * (y_f * y_f))
        gamma_f = prefactor * sqrt_term / (16.0 * math.pi * (fa * fa))
        gammas[ch] = float(max(gamma_f, 0.0))
        total += gammas[ch]

    brs = {ch: (gammas.get(ch, 0.0) / total if total > 0 else 0.0) for ch in channels}
    return brs, gammas, total


def load_channel_acceptances(csv_path: str):
    """Return a dict mapping (mass, CaPhi) -> acceptance for that channel.

    Preference: use weighted columns if present (`nLLP_original_weighted`),
    otherwise fall back to unweighted counts (`nLLP_original`, `nLLP_Final`).
    """
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()

    # Require raw/unweighted counts to match per-channel scripts: sum across
    # independently-generated groups and compute final_sum / orig_sum.
    if 'nLLP_original' in df.columns and 'nLLP_Final' in df.columns:
        orig_col = 'nLLP_original'
        final_col = 'nLLP_Final'
    elif 'nLLP_original' in df.columns and 'n_surviving_llps' in df.columns:
        orig_col = 'nLLP_original'
        final_col = 'n_surviving_llps'
    else:
        raise RuntimeError(f"No suitable unweighted LLP count columns found in {csv_path} (expected 'nLLP_original' + 'nLLP_Final' or 'n_surviving_llps')")

    grouped = df.groupby(['mass', 'CaPhi'])[[orig_col, final_col]].sum().reset_index()
    acc_map = {}
    for _, row in grouped.iterrows():
        m = float(row['mass'])
        c = float(row['CaPhi'])
        orig = float(row[orig_col])
        final = float(row[final_col])
        acc = (final / orig) if orig > 0 else 0.0
        acc_map[(m, c)] = acc

    return acc_map


def prepare_heatmap_data_from_grid(grid_df: pd.DataFrame, value_column='N_signal'):
    """Prepare arrays for heatmap plotting from a DataFrame of points.
    Expects `mass`, `CaPhi`, and `value_column` columns.
    Returns mass_vals (sorted), caphi_vals (sorted), heatmap 2D (CaPhi x Mass).
    """
    df = grid_df.copy()
    mass_vals = np.sort(df['mass'].unique())
    caphi_vals = np.sort(df['CaPhi'].unique())
    heat = np.full((len(caphi_vals), len(mass_vals)), np.nan)
    # fill
    for _, row in df.iterrows():
        i = int(np.searchsorted(caphi_vals, row['CaPhi']))
        j = int(np.searchsorted(mass_vals, row['mass']))
        heat[i, j] = float(row[value_column]) if not np.isnan(row[value_column]) else np.nan
    return mass_vals, caphi_vals, heat


def plot_heatmap(mass_vals, caphi_vals, heatmap_data, output_path,
                 title="Expected Signal Events", use_log_scale=True,
                 colorbar_label="N_signal", vmin_override=None, vmax_override=None,
                 show_percentage=None, sigfig=3):
    """Scatter/box-style heatmap used throughout the Higgs plots."""
    fig, ax = plt.subplots(figsize=(12, 9))

    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    positive = valid_data[valid_data > 0]
    if len(positive) > 0:
        vmin = vmin_override if vmin_override is not None else np.min(positive)
        vmax = vmax_override if vmax_override is not None else np.max(positive)
    else:
        vmin = vmin_override if vmin_override is not None else 1e-12
        vmax = vmax_override if vmax_override is not None else 1.0

    if use_log_scale and vmax > 0:
        # Allow very small positive values as the lower bound instead of forcing 1e-12.
        # Use the actual smallest positive datum when available; fall back to the smallest
        # safe positive float for LogNorm to avoid vmin <= 0.
        if len(positive) > 0:
            vmin_val = vmin_override if vmin_override is not None else float(np.min(positive))
            vmax_val = vmax_override if vmax_override is not None else float(np.max(positive))
        else:
            vmin_val = vmin_override if vmin_override is not None else np.finfo(float).tiny
            vmax_val = vmax_override if vmax_override is not None else 1.0
        vmin_safe = max(vmin_val, np.finfo(float).tiny)
        norm = colors.LogNorm(vmin=vmin_safe, vmax=vmax_val)
    else:
        norm = None

    # gather all non-nan points
    points = []
    for i, caphi in enumerate(caphi_vals):
        for j, mass in enumerate(mass_vals):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                points.append((mass, caphi, val))

    if len(points) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        return

    arr = np.array(points, dtype=object)
    mass_arr = arr[:, 0].astype(float)
    caphi_arr = arr[:, 1].astype(float)
    vals_arr = arr[:, 2].astype(float)

    # Cluster points per CaPhi and plot large squares for outer points and small
    # dots for intermediate points
    idx_large = []
    idx_small = []
    try:
        log_mass_arr = np.log10(mass_arr)
        for c in np.unique(caphi_arr):
            mask = (caphi_arr == c)
            masses = mass_arr[mask]
            log_masses = log_mass_arr[mask]
            vals = vals_arr[mask]
            idxs = np.where(mask)[0]
            if len(masses) == 1:
                idx_large.append(idxs[0])
            else:
                sorted_idx = np.argsort(masses)
                sorted_masses = masses[sorted_idx]
                sorted_log_masses = log_masses[sorted_idx]
                sorted_idxs = idxs[sorted_idx]
                cluster = [sorted_idxs[0]]
                for ii in range(1, len(sorted_masses)):
                    if abs(sorted_log_masses[ii] - sorted_log_masses[ii-1]) < 0.08:
                        cluster.append(sorted_idxs[ii])
                    else:
                        if len(cluster) == 1:
                            idx_large.append(cluster[0])
                        else:
                            idx_large.append(cluster[0])
                            idx_large.append(cluster[-1])
                            for mid in cluster[1:-1]:
                                idx_small.append(mid)
                        cluster = [sorted_idxs[ii]]
                if len(cluster) == 1:
                    idx_large.append(cluster[0])
                else:
                    idx_large.append(cluster[0])
                    idx_large.append(cluster[-1])
                    for mid in cluster[1:-1]:
                        idx_small.append(mid)
        idx_large = sorted(set(idx_large))
        idx_small = sorted(set(idx_small))
    except Exception:
        idx_large = []
        idx_small = []

    # Plot large boxes for outermost points
    if len(idx_large) > 0:
        is_zero = (vals_arr[idx_large] == 0)
        if np.any(is_zero):
            ax.scatter(mass_arr[idx_large][is_zero], caphi_arr[idx_large][is_zero],
                       marker='s', s=800, facecolors='lightgrey', edgecolors='black', linewidths=0.5, zorder=2)
        if np.any(~is_zero):
            ax.scatter(mass_arr[idx_large][~is_zero], caphi_arr[idx_large][~is_zero],
                       c=vals_arr[idx_large][~is_zero], cmap=plt.cm.viridis, norm=norm,
                       marker='s', s=800, edgecolors='black', linewidths=0.5, alpha=1.0, zorder=2)

    # Plot small dots for intermediate cluster members
    if len(idx_small) > 0:
        is_zero_small = (vals_arr[idx_small] == 0)
        if np.any(is_zero_small):
            ax.scatter(mass_arr[idx_small][is_zero_small], caphi_arr[idx_small][is_zero_small],
                       marker='o', s=30, c='lightgrey', edgecolors='none', alpha=0.8, zorder=3)
        if np.any(~is_zero_small):
            ax.scatter(mass_arr[idx_small][~is_zero_small], caphi_arr[idx_small][~is_zero_small],
                       c=vals_arr[idx_small][~is_zero_small], cmap=plt.cm.viridis, norm=norm,
                       marker='o', s=30, edgecolors='none', alpha=0.8, zorder=3)

    # For colorbar, use all nonzero points (invisible scatter)
    nonzero_all_mask = (vals_arr != 0)
    if np.any(nonzero_all_mask):
        sc = ax.scatter(mass_arr[nonzero_all_mask], caphi_arr[nonzero_all_mask],
                        c=vals_arr[nonzero_all_mask], cmap=plt.cm.viridis, norm=norm, marker='o', s=0)
    else:
        sc = ax.scatter([], [], c=[], cmap=plt.cm.viridis, norm=norm)

    # Annotate the outermost (large) boxes with readable text
    if len(arr) > 0 and len(idx_large) > 0:
        overall_vmax = np.nanmax(vals_arr) if vals_arr.size > 0 else 0
        if show_percentage is None:
            show_as_percentage = (overall_vmax <= 1.0)
        else:
            show_as_percentage = bool(show_percentage)
        for idx in idx_large:
            try:
                mval = mass_arr[idx]
                cval = caphi_arr[idx]
                value = vals_arr[idx]
            except Exception:
                continue
            if np.isnan(value):
                continue
            if show_as_percentage:
                text = f'{value*100:.2f}%' if value > 0 else '0%'
            else:
                try:
                    text = f'{value:.{sigfig}g}'
                except ValueError:
                    text = str(value)
            ax.text(mval, cval, text,
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    cbar = plt.colorbar(sc, ax=ax, label=colorbar_label, pad=0.02)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass [GeV]', fontsize=14)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=14)
    ax.set_title(title, fontsize=16, pad=16)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='pp->Z a signal events heatmap')
    parser.add_argument('--higgs-dir', type=str,
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting/pp_production',
                        help='Base pp_production directory (contains decay channel folders)')
    parser.add_argument('--ufo-dir', type=str,
                        default='/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified',
                        help='UFO directory providing decays/parameters')
    parser.add_argument('--scan-cross-section-dir', type=str,
                        default='/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_Z_FINAL_Default_Lifetime_With_Reweighting/Events_For_Cross_Section_Data/ALP_axZ_scan_1/Events',
                        help='Directory containing scan_run_*.txt files with cross-section info (σ(pp→Z a))')
    parser.add_argument('--output-dir', type=str, default='/usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting/pp_production/Plots', help='Where to save plots and CSVs')
    parser.add_argument('--luminosity', type=float, default=3000.0, help='Integrated luminosity [fb^-1]')
    parser.add_argument('--selection-eff', type=float, default=0.5, help='Selection efficiency')
    parser.add_argument('--br-z-visible', type=float, default=0.8, help='BR(Z->visible)')
    parser.add_argument('--save-csv', dest='save_csv', action='store_true',
                        help='Save extended CSV of per-point numbers (default: enabled)')
    parser.add_argument('--no-save-csv', dest='save_csv', action='store_false',
                        help='Do not save extended CSV')
    parser.add_argument('--acceptance-percent', dest='acceptance_percent', action='store_true',
                        help='Show acceptance as percentage')
    parser.add_argument('--no-acceptance-percent', dest='acceptance_percent', action='store_false',
                        help='Show acceptance as fraction (not percentage)')
    parser.set_defaults(save_csv=True, acceptance_percent=False)
    args = parser.parse_args()

    print('Building UFO parameter namespace...')
    ns = build_ufo_namespace(args.ufo_dir)

    # Read production cross-sections σ(pp->Z a) from scan files
    scan_map = {}
    if args.scan_cross_section_dir:
        print(f'Loading cross-section scan files from {args.scan_cross_section_dir}...')
        scan_map = parse_scan_cross_section_files(args.scan_cross_section_dir)
        if not scan_map:
            print('Warning: no scan cross-section data found; σ will be set to 0 for missing points')
    else:
        print('No --scan-cross-section-dir provided; σ will be set to 0 for all points')

    # discover decay channels in production dir
    base = Path(args.higgs_dir)
    chans = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.endswith('_Decay_Channel'):
            chans.append(p.name.replace('_Decay_Channel', ''))

    if not chans:
        raise RuntimeError(f'No decay channel folders found under {args.higgs_dir}')

    print('Found channels:', chans)

    # load acceptances per channel
    channel_acceptances = {}
    for ch in chans:
        # find selection_cutflow CSV inside folder
        folder = base / f"{ch}_Decay_Channel"
        pattern = str(folder / 'selection_cutflow_*_decay_channel.csv')
        files = glob.glob(pattern)
        if not files:
            # try generic file name fallback
            files = glob.glob(str(folder / 'selection_cutflow_*.csv'))
        if not files:
            raise RuntimeError(f"No selection_cutflow CSV found for channel {ch} in {folder}; cannot compute acceptances")
        csv_path = files[0]
        print(f'Loading acceptances for {ch} from {csv_path}...')
        try:
            channel_acceptances[ch] = load_channel_acceptances(csv_path)
        except Exception as e:
            raise RuntimeError(f'Error loading {csv_path}: {e}')

    # gather all mass/CaPhi grid points from union of channels
    keys = set()
    for ch_map in channel_acceptances.values():
        keys.update(ch_map.keys())
    if not keys:
        raise RuntimeError('No grid points found in any channel selection CSVs')
    keys = sorted(keys)

    results = []
    lumi_pb = args.luminosity * 1000.0

    for (m_a, caphi) in keys:
        # build local ns for this point
        ns_point = dict(ns)
        ns_point['Ma'] = m_a
        ns_point['CaPhi'] = caphi

        # evaluate ALP BRs for available channels using analytic formula
        brs, gammas, total_gamma = compute_alp_brs_for_point(ns_point, chans)

        # compute acceptance_overall = sum_i BR_i * acceptance_i
        acceptance = 0.0
        for ch in chans:
            acc_map = channel_acceptances.get(ch, {})
            acc = acc_map.get((m_a, caphi), 0.0)
            acceptance += brs.get(ch, 0.0) * acc

        # obtain production cross-section σ(pp->Z a) from scan_map
        try:
            sigma_pb = lookup_scan_sigma_br(scan_map, m_a, caphi)
            if sigma_pb is None:
                print(f'Warning: cross-section value is missing for mass={m_a}, CaPhi={caphi}; setting sigma_pb=0')
                sigma_pb = 0.0
        except KeyError:
            print(f'Warning: no cross-section entry found for mass={m_a}, CaPhi={caphi}; setting sigma_pb=0')
            sigma_pb = 0.0

        N_signal = lumi_pb * sigma_pb * acceptance * args.selection_eff * args.br_z_visible

        results.append({
            'mass': float(m_a),
            'CaPhi': float(caphi),
            'acceptance': float(acceptance),
            'cross_section_pb': float(sigma_pb),
            'Gamma_total_alp_GeV': float(total_gamma),
            'N_signal': float(N_signal)
        })

    df_res = pd.DataFrame(results)

    os.makedirs(args.output_dir, exist_ok=True)

    # save extended CSV if requested
    if args.save_csv:
        out_csv = os.path.join(args.output_dir, 'ppZ_signal_events_data.csv')
        df_res.to_csv(out_csv, index=False)
        print(f'Saved extended data to {out_csv}')

    # prepare heatmaps
    mass_vals, caphi_vals, heat_sig = prepare_heatmap_data_from_grid(df_res, 'N_signal')
    _, _, heat_acc = prepare_heatmap_data_from_grid(df_res, 'acceptance')
    _, _, heat_xsec = prepare_heatmap_data_from_grid(df_res, 'cross_section_pb')

    # plot
    print('Generating plots...')
    plot_heatmap(mass_vals, caphi_vals, heat_sig, os.path.join(args.output_dir, 'signal_events_heatmap_ppZ.png'),
                 title=f'Expected Signal Events (pp → Z a, L={args.luminosity} fb^-1)', use_log_scale=True,
                 colorbar_label='Expected Signal Events', show_percentage=False, sigfig=3)

    plot_heatmap(mass_vals, caphi_vals, heat_acc, os.path.join(args.output_dir, 'acceptance_heatmap_ppZ.png'),
                 title='Geometric & Kinematic Acceptance (weighted by ALP BR)', use_log_scale=True,
                 colorbar_label='Acceptance ε', show_percentage=args.acceptance_percent, sigfig=3)

    plot_heatmap(mass_vals, caphi_vals, heat_xsec, os.path.join(args.output_dir, 'cross_section_heatmap_ppZ.png'),
                 title='Production Cross-Section σ(pp→Z a) [pb]', use_log_scale=True,
                 colorbar_label='σ(pp→Z a) [pb]', show_percentage=False, sigfig=3)

    # summary
    print('\nSUMMARY')
    print(f'Points: {len(df_res)}')
    if not df_res['N_signal'].isna().all():
        print(f"Total expected events (sum): {df_res['N_signal'].sum():.3f}")
        print(f"Max: {df_res['N_signal'].max():.3f}")
    print(f"Acceptance range: {df_res['acceptance'].min():.3e} - {df_res['acceptance'].max():.3e}")
    print(f"Cross-section range (pb): {df_res['cross_section_pb'].min():.3e} - {df_res['cross_section_pb'].max():.3e}")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()