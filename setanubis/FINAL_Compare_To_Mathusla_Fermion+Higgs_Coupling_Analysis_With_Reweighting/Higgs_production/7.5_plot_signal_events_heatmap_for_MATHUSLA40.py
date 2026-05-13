"""
Higgs production: plot heat map of expected signal events vs ALP mass and coupling CaPhi.

This script computes N_signal for pp -> h (ggF) -> Z a, where the ALP (a)
may decay into several channels. The acceptance for each channel is read
from that channel's selection cutflow CSV; the overall geometric/kinematic
acceptance is computed as a BR-weighted average over channels:

    acceptance_overall = sum_i BR(a->i) * (N_{i,sel} / N_{i,gen})

Cross-section and BR(h->Z a) are computed using the helper in
`6_Higgs_ggF_Zax_Cross_Section.py` (imported dynamically). ALP partial
widths / branching ratios are computed using the analytic fermionic
partial-width formula used in the per-channel extraction scripts
(see `*_Decay_Channel/2_extract_selection_data.py`).

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


# ALP BR calculation removed: this script is now mu-heatmaps only.
# If needed in the future, reintroduce a BR/width helper here.


# Channel acceptance loader removed (not needed for mu-heatmaps only flow).


def prepare_heatmap_data_from_grid(grid_df: pd.DataFrame, value_column='N_signal', x_column='mass'):
    """Prepare arrays for heatmap plotting from a DataFrame of points.
    Expects `x_column`, `CaPhi`, and `value_column` columns.
    Returns x_vals (sorted), caphi_vals (sorted), heatmap 2D (CaPhi x x_column).
    """
    df = grid_df.copy()
    x_vals = np.sort(df[x_column].unique())
    caphi_vals = np.sort(df['CaPhi'].unique())
    heat = np.full((len(caphi_vals), len(x_vals)), np.nan)
    # fill
    for _, row in df.iterrows():
        i = int(np.searchsorted(caphi_vals, row['CaPhi']))
        j = int(np.searchsorted(x_vals, row[x_column]))
        heat[i, j] = float(row[value_column]) if not np.isnan(row[value_column]) else np.nan
    return x_vals, caphi_vals, heat


def plot_heatmap(mass_vals, caphi_vals, heatmap_data, output_path,
                                 title="Expected Signal Events", use_log_scale=True,
                                 colorbar_label="N_signal", vmin_override=None, vmax_override=None,
                                                                 show_percentage=None, sigfig=3,
                                                                 xlabel='ALP Mass [GeV]', ylabel=r'Coupling $C_{a\phi}$'):
        """Scatter/box-style heatmap used throughout the Higgs plots.

        New args:
        - show_percentage: if True, annotate values as percentages; if False,
            annotate as numeric values. If None (default), auto-detect by
            checking if the maximum value <= 1.0.
        - sigfig: number of significant figures to use when annotating numeric
            values (applies when not using percentages).
        """
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
        # dots for intermediate points (match per-channel plotting style).
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
            # Determine whether to display annotated values as percentages.
            # If `show_percentage` is None, fall back to automatic heuristic (max<=1),
            # otherwise honor explicit boolean request.
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
                    # Format numeric values with requested significant figures.
                    try:
                        text = f'{value:.{sigfig}g}'
                    except Exception:
                        text = f'{value:.{sigfig}g}'
                ax.text(mval, cval, text,
                        ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold',
                        path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        cbar = plt.colorbar(sc, ax=ax, label=colorbar_label, pad=0.02)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16, pad=16)
        ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Higgs signal events heatmap: pp->h->Z a')
    parser.add_argument('--higgs-dir', type=str,
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Compare_To_Mathusla_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production',
                        help='Base Higgs_production directory (contains decay channel folders)')
    parser.add_argument('--ufo-dir', type=str,
                        default='/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified',
                        help='UFO directory providing decays/parameters')
    parser.add_argument('--cross-section-script', type=str,
                        default='/usera/fs568/set-anubis/setanubis/FINAL_Compare_To_Mathusla_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/6_Higgs_ggF_Zax_Cross_Section.py',
                        help='Path to Higgs cross-section helper')
    parser.add_argument('--output-dir', type=str, default='/usera/fs568/set-anubis/setanubis/FINAL_Compare_To_Mathusla_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/Plots', help='Where to save plots and CSVs')
    parser.add_argument('--luminosity', type=float, default=3000.0, help='Integrated luminosity [fb^-1]')
    parser.add_argument('--selection-eff', type=float, default=0.5, help='Selection efficiency')
    parser.add_argument('--br-z-visible', type=float, default=0.8, help='BR(Z->visible)')
    parser.add_argument('--sigma-ggf', type=float, default=54.61, help='Higgs ggF cross-section [pb] (default ~54.61 pb)')
    parser.add_argument('--gamma-h-sm', type=float, default=None, help='SM Higgs total width [GeV] (default from UFO if available)')
    parser.add_argument('--save-csv', dest='save_csv', action='store_true',
                        help='Save extended CSV of per-point numbers (default: enabled)')
    parser.add_argument('--no-save-csv', dest='save_csv', action='store_false',
                        help='Do not save extended CSV')
    parser.add_argument('--acceptance-percent', dest='acceptance_percent', action='store_true',
                        help='Show acceptance as percentage')
    parser.add_argument('--no-acceptance-percent', dest='acceptance_percent', action='store_false',
                        help='Show acceptance as fraction (not percentage)')
    parser.add_argument('--czh-eff', type=float, default=0.72,
                        help='Effective coupling C_Zh^eff for h->Z a (dimensionless)')
    parser.add_argument('--mu-heatmaps', dest='mu_heatmaps', action='store_true',
                        help='Produce Caphi vs C_Zh heatmaps for mumu channel per BR_mu values (mass filter)')
    parser.add_argument('--czh-grid', dest='czh_grid', type=str, default='log:0.001,10,200',
                        help='Grid specification for C_Zh values: "log:start,stop,n" or "lin:start,stop,n" or comma-list')
    parser.add_argument('--mass', dest='mass', type=float, default=1.0, help='ALP mass [GeV] to plot (default 1.0)')
    parser.set_defaults(save_csv=True, acceptance_percent=False)
    args = parser.parse_args()
    # If the script is invoked with no CLI arguments, enable muon per-BR
    # heatmaps by default to match the user's expected behavior when running
    # the script without flags.
    if len(sys.argv) == 1:
        args.mu_heatmaps = True
        print('No CLI arguments provided — enabling --mu-heatmaps by default.')

    print('Building UFO parameter namespace...')
    ns = build_ufo_namespace(args.ufo_dir)

    print('Loading Higgs cross-section helper...')
    higgs_mod = load_module_from_path('higgs_xsec', args.cross_section_script)

    # SM Higgs total width: prepare early for mu-heatmaps mode
    if args.gamma_h_sm is not None:
        gamma_h_sm = float(args.gamma_h_sm)
    elif 'WH' in ns:
        gamma_h_sm = float(ns['WH'])
    else:
        gamma_h_sm = None

    # base path for decay channel data
    base = Path(args.higgs_dir)

    # SM Higgs total width: require explicit input to avoid silent fallbacks
    if gamma_h_sm is None:
        raise RuntimeError("SM Higgs total width not provided: pass --gamma-h-sm or ensure 'WH' is present in the UFO parameters")

    def parse_czh_grid(grid_spec: str):
        """Parse a grid specification string into a numpy array of floats.

        Supported formats:
        - 'log:start,stop,n' -> logspace from start..stop with n points
        - 'lin:start,stop,n' -> linspace
        - 'v1,v2,v3' -> explicit comma-separated list
        """
        if isinstance(grid_spec, (list, tuple, np.ndarray)):
            return np.array(grid_spec, dtype=float)
        s = str(grid_spec)
        if ':' in s:
            prefix, rest = s.split(':', 1)
            parts = [p.strip() for p in rest.split(',') if p.strip()]
            if prefix.lower() == 'log' and len(parts) == 3:
                start = float(parts[0])
                stop = float(parts[1])
                n = int(parts[2])
                return np.logspace(math.log10(start), math.log10(stop), num=n)
            if prefix.lower() == 'lin' and len(parts) == 3:
                start = float(parts[0])
                stop = float(parts[1])
                n = int(parts[2])
                return np.linspace(start, stop, num=n)
        # fallback: comma-separated list
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return np.array([float(p) for p in parts], dtype=float)

    def compute_mu_heatmaps():
        """Produce Caphi vs C_Zh heatmaps for mumu channel per BR_mu (for a fixed mass).

        Uses the mumu `selection_cutflow` CSV to extract per-(mass,CaPhi) acceptance
        and then computes N_signal for a grid of C_Zh values and each BR_mu found in the CSV.
        """
        folder = base / 'mumu_Decay_Channel'
        pattern = str(folder / 'selection_cutflow_*_decay_channel_MATHUSLA40.csv')
        files = glob.glob(pattern)
        if not files:
            files = glob.glob(str(folder / 'selection_cutflow_*.csv'))
        if not files:
            raise RuntimeError(f'No mumu selection_cutflow CSV found in {folder}')
        csv_path = files[0]
        print(f'Loading mumu selection CSV from {csv_path}...')
        df_full = pd.read_csv(csv_path)
        # only keep successful rows if present
        if 'status' in df_full.columns:
            df_full = df_full[df_full['status'] == 'success'].copy()

        # ensure numeric types
        df_full['mass'] = df_full['mass'].astype(float)
        df_full['CaPhi'] = df_full['CaPhi'].astype(float)

        # select rows matching requested mass
        df_mass = df_full[np.isclose(df_full['mass'].astype(float), float(args.mass))]
        if df_mass.shape[0] == 0:
            raise RuntimeError(f'No entries for mass={args.mass} found in {csv_path}')

        # BR_mu values for this mass
        if 'BR_mu' not in df_mass.columns:
            raise RuntimeError('No BR_mu column found in mumu selection CSV')
        br_vals = sorted(df_mass['BR_mu'].astype(float).unique())
        if len(br_vals) == 0:
            raise RuntimeError('No BR_mu values found in mumu CSV')

        # CaPhi values (use union present for this mass so axis is consistent across BRs)
        caphi_vals = sorted(df_mass['CaPhi'].astype(float).unique())

        # parse C_Zh grid
        czh_list = parse_czh_grid(args.czh_grid)

        # required SM / UFO parameters
        if 'MH' not in ns or 'MZ' not in ns:
            raise RuntimeError("Missing 'MH' or 'MZ' in UFO namespace; needed for cross-section calls")
        m_h = float(ns['MH'])
        m_Z = float(ns['MZ'])
        if 'fa' in ns:
            f_a = float(ns['fa'])
        elif 'FA' in ns:
            f_a = float(ns['FA'])
        else:
            raise RuntimeError("Missing required parameter 'fa' in UFO namespace")

        lumi_pb = args.luminosity * 1000.0
        os.makedirs(args.output_dir, exist_ok=True)

        # Helper: build acceptance map from a selection DataFrame (group by mass,CaPhi)
        def build_acc_map_from_df(df_in):
            if 'nLLP_original' in df_in.columns and 'nLLP_Final' in df_in.columns:
                orig_col = 'nLLP_original'
                final_col = 'nLLP_Final'
            elif 'nLLP_original' in df_in.columns and 'n_surviving_llps' in df_in.columns:
                orig_col = 'nLLP_original'
                final_col = 'n_surviving_llps'
            else:
                raise RuntimeError(f"No suitable LLP count columns found in {csv_path} for acceptance calculation")

            grouped = df_in.groupby(['mass', 'CaPhi'])[[orig_col, final_col]].sum().reset_index()
            acc_map_local = {}
            for _, row in grouped.iterrows():
                m = float(row['mass'])
                c = float(row['CaPhi'])
                orig = float(row[orig_col])
                final = float(row[final_col])
                acc_map_local[(m, c)] = (final / orig) if orig > 0 else 0.0
            return acc_map_local

        for br in br_vals:
            # filter rows for this BR
            df_br = df_mass[np.isclose(df_mass['BR_mu'].astype(float), float(br))]
            if df_br.shape[0] == 0:
                print(f'  Warning: no rows for BR_mu={br} at mass={args.mass}; skipping')
                continue

            acc_map_br = build_acc_map_from_df(df_br)

            results_br = []
            for czh in czh_list:
                sigma_pb, br_h_za = higgs_mod.higgs_cross_section(m_h, m_Z, float(args.mass), f_a, float(czh), args.sigma_ggf, float(gamma_h_sm))
                for caphi in caphi_vals:
                    acc = acc_map_br.get((float(args.mass), float(caphi)), 0.0)
                    N_signal = lumi_pb * sigma_pb * float(br) * float(acc) * args.selection_eff * args.br_z_visible
                    # [FIXED]: Store coupling in C_Zh and preserve the fixed mass in its own column
                    results_br.append({'C_Zh': float(czh), 'mass': float(args.mass), 'CaPhi': float(caphi), 'BR_mu': float(br), 'acceptance': float(acc), 'cross_section_pb': float(sigma_pb), 'BR_h_to_Za': float(br_h_za), 'N_signal': float(N_signal)})

            df_res = pd.DataFrame(results_br)

            # save CSV for this BR
            out_csv = os.path.join(args.output_dir, f'Simulated_MATHUSLA40_higgs_signal_events_data_mumu_BR_{str(br).replace(".","p")}.csv')
            df_res.to_csv(out_csv, index=False)
            print(f'Saved mumu extended data to {out_csv}')

            # prepare heatmap (x: C_Zh, y: CaPhi)
            # [FIXED]: Explicitly pass x_column='C_Zh'
            mass_vals_grid, caphi_vals_grid, heat_sig = prepare_heatmap_data_from_grid(df_res, 'N_signal', x_column='C_Zh')

            xlabel = r'$C_{Zh}^{eff}$'
            ylabel = r'Coupling $C_{a\phi}$'

            out_png = os.path.join(args.output_dir, f'Simulated_MATHUSLA40_signal_events_heatmap_mumu_BR_{str(br).replace(".","p")}.png')
            plot_heatmap(mass_vals_grid, caphi_vals_grid, heat_sig, out_png,
                         title=(f'Simulated MATHUSLA40 Expected Signal Events (mumu), BR_mu={br}, mass={args.mass} GeV'),
                         use_log_scale=True, colorbar_label='Expected Signal Events', show_percentage=False,
                         sigfig=3, xlabel=xlabel, ylabel=ylabel)
            print(f'Saved mumu heatmap to {out_png}')

        print('Finished muon-channel heatmaps')

    # Run mu-heatmaps only (unconditional)
    compute_mu_heatmaps()
    return


if __name__ == '__main__':
    main()