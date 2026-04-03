"""
Plot cross-section heatmap from raw MadGraph `scan_run` text files

This script parses `scan_run_*.txt` files under a `Generated_Events_1/ALP_axZ_scan_*`
structure, extracts `mass`, the varying ALP parameter (taken from the alppars columns),
and `cross` to build a heatmap of cross-sections vs mass and coupling.

Usage:
  python plot_cross_section_heatmap_from_raw.py --generated-path /raid/.../Generated_Events_1 --output-dir out

"""
import os
import re
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import cmath

# Ensure the UFO model directory is on sys.path so local imports like
# `import function_library` inside the UFO package work.
_ufo_dir = os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..', '..', 'Assets', 'UFO', 'ALP_linear_UFO_WIDTH_modified')
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


def find_scan_dirs(generated_path):
    pattern = os.path.join(generated_path, 'ALP_axZ_scan_*')
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


def read_scan_run_file(scan_events_dir):
    """Return a DataFrame read from the scan_run file in the provided Events dir."""
    files = glob.glob(os.path.join(scan_events_dir, 'scan_run_*.txt'))
    if not files:
        return None
    f = files[0]
    # Read whitespace-separated table; header line may start with '#'
    df = pd.read_csv(f, sep=r'\s+', header=0)
    # Clean column names (remove leading '#')
    df.columns = [c.lstrip('#').strip() for c in df.columns]
    return df


def collect_cross_section_data(generated_path):
    rows = []
    scan_dirs = find_scan_dirs(generated_path)
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        m = re.search(r'ALP_axZ_scan_(\d+)', scan_name)
        scan_num = int(m.group(1)) if m else None
        events_dir = os.path.join(scan_dir, 'Events')
        if not os.path.isdir(events_dir):
            continue
        df_scan = read_scan_run_file(events_dir)
        if df_scan is None:
            continue

        # Determine which alppars column to use for coupling: pick the last alppars column
        alppars_cols = [c for c in df_scan.columns if c.startswith('alppars')]
        if len(alppars_cols) == 0:
            # fallback: try columns named alppars#N pattern
            alppars_cols = [c for c in df_scan.columns if 'alppars' in c]
        if len(alppars_cols) == 0:
            raise RuntimeError(f"No alppars columns found in {events_dir}")
        # Use the last alppars column (this matches existing generated data where the last param varies)
        caphi_col = alppars_cols[-1]

        for _, r in df_scan.iterrows():
            run_name_col = None
            # prefer an explicit run_name column if present
            for cand in ['run_name']:
                if cand in df_scan.columns:
                    run_name_col = cand
                    break
            # fallback: first column is run name
            if run_name_col is None:
                run_name_col = df_scan.columns[0]

            run_name = str(r[run_name_col])
            run_m = re.search(r'run_(\d+)', run_name)
            run_num = int(run_m.group(1)) if run_m else None

            mass = float(r['mass']) if 'mass' in df_scan.columns else float(r[df_scan.columns[1]])
            caphi = float(r[caphi_col])
            cross = float(r['cross']) if 'cross' in df_scan.columns else float(r[df_scan.columns[-1]])

            rows.append({'scan': scan_num, 'run': run_num, 'mass': mass, 'CaPhi': caphi, 'cross': cross})

    df = pd.DataFrame(rows)
    return df


def prepare_heatmap_data(df, value_column='cross'):
    df_filtered = df.copy()
    mass_vals = np.sort(df_filtered['mass'].unique())
    caphi_vals = np.sort(df_filtered['CaPhi'].unique())
    heatmap_data = np.zeros((len(caphi_vals), len(mass_vals)))
    heatmap_data[:] = np.nan

    for i, caphi in enumerate(caphi_vals):
        for j, mass in enumerate(mass_vals):
            mask = (df_filtered['mass'] == mass) & (df_filtered['CaPhi'] == caphi)
            matching = df_filtered[mask]
            if len(matching) > 0:
                heatmap_data[i, j] = matching[value_column].mean()
            else:
                heatmap_data[i, j] = np.nan

    return mass_vals, caphi_vals, heatmap_data


def plot_heatmap(mass_vals, caphi_vals, heatmap_data, output_path,
                 title='Cross-section heatmap', colorbar_label='Cross-section (pb)', use_log_scale=True,
                 marker_size=800):
    fig, ax = plt.subplots(figsize=(12, 9))
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    positive_data = valid_data[valid_data > 0]
    if len(positive_data) > 0:
        vmin = np.min(positive_data)
        vmax = np.max(positive_data)
    else:
        vmin = 1e-12
        vmax = 1.0

    if use_log_scale and vmax > 0:
        norm = colors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
        cmap = plt.cm.viridis
    else:
        norm = None
        cmap = plt.cm.viridis

    # scatter points for non-NaN cells at representative positions
    all_points = []
    for i, caphi in enumerate(caphi_vals):
        for j, mass in enumerate(mass_vals):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                all_points.append((mass, caphi, val))

    im = None
    if len(all_points) > 0:
        arr = np.array(all_points, dtype=object)
        mass_arr = arr[:, 0].astype(float)
        caphi_arr = arr[:, 1].astype(float)
        vals = arr[:, 2].astype(float)
        # colored squares for positive values
        mask_pos = vals > 0
        if np.any(mask_pos):
            im = ax.scatter(mass_arr[mask_pos], caphi_arr[mask_pos], c=vals[mask_pos], cmap=cmap, norm=norm,
                            s=marker_size, marker='s', edgecolors='black', linewidths=0.5, zorder=2)
        # zeros -> light grey squares
        if np.any(~mask_pos):
            ax.scatter(mass_arr[~mask_pos], caphi_arr[~mask_pos], s=marker_size, marker='s', facecolors='lightgrey',
                       edgecolors='black', linewidths=0.5, zorder=1)

        # Annotate each square with a formatted number
        for x, y, v in zip(mass_arr, caphi_arr, vals):
            if np.isnan(v):
                continue
            if v == 0:
                text = '0'
            elif abs(v) < 1e-2 or abs(v) >= 1e4:
                text = f'{v:.2e}'
            elif abs(v) < 1:
                text = f'{v:.2e}'
            else:
                text = f'{v:.2f}'
            txt = ax.text(x, y, text, ha='center', va='center', fontsize=8, color='white', zorder=3)
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    else:
        im = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

    cbar = plt.colorbar(im, ax=ax, label=colorbar_label, pad=0.02) if im is not None else None
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass [GeV]')
    ax.set_ylabel(r'Coupling $C_{a\phi}$')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot cross-section heatmap from raw scan_run files')
    parser.add_argument('--generated-path', type=str, required=False,
                        default='/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_Z_Runs_2/Generated_Events_1',
                        help='Path to Generated_Events_1 directory')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plot')
    parser.add_argument('--save-csv', action='store_true', help='Save parsed CSV of cross-sections')
    parser.add_argument('--marker-size', type=int, default=800, help='Marker (square) size in points^2 for heatmap cells')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = collect_cross_section_data(args.generated_path)
    if df is None or df.empty:
        print('No data found. Exiting.')
        return

    # Print parsed statistics
    print(f'Parsed {len(df)} parameter points from generated files')
    if 'cross' in df.columns:
        vals = df['cross'].dropna().values
        if len(vals) > 0:
            print(f'Cross-section (pb): min={vals.min():.3e}, max={vals.max():.3e}, mean={vals.mean():.3e}, median={np.median(vals):.3e}')
        else:
            print('No cross-section values found (all NaN)')

    if args.save_csv:
        outcsv = os.path.join(args.output_dir, 'parsed_cross_sections.csv')
        df.to_csv(outcsv, index=False)
        print(f'Saved parsed CSV to {outcsv}')

    mass_vals, caphi_vals, heatmap = prepare_heatmap_data(df, 'cross')
    outpng = os.path.join(args.output_dir, 'Associated_cross_section_heatmap_for_quark_currents_to_compare_to_higgs_ggF_cross_section.png')
    plot_heatmap(mass_vals, caphi_vals, heatmap, outpng,
                 title='Production Cross-Section (pp → Z + ALP) from raw scan_run files',
                 colorbar_label='Cross-section (pb)', use_log_scale=True, marker_size=args.marker_size)

    print(f'Wrote heatmap to {outpng}')

    # --- Higgs-derived cross-sections on same grid ---
    # Try to import Higgs_Cross_Section module and compute sigma(pp->h)*BR(h->Za)
    try:
        import importlib.util
        hcs_path = os.path.join(os.path.dirname(__file__), 'Higgs_ggF_Zax_Cross_Section.py')
        hcs_path = os.path.abspath(hcs_path)
        spec = importlib.util.spec_from_file_location('Higgs_Cross_Section', hcs_path)
        hcs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hcs)

        # obtain defaults from module if available
        m_h = ns['MH']
        m_Z = ns['MZ']
        y_t = ns['yt']
        N_c = 3
        T_t3 = 0.5
        f_a = 1000
        sigma_ggF = 54.61
        gamma_h_SM = 4.07e-3

        # compute higgs cross-section grid matching the same mass and CaPhi points
        higgs_heatmap = np.zeros_like(heatmap)
        higgs_heatmap[:] = np.nan
        for i, caphi in enumerate(caphi_vals):
            for j, mass in enumerate(mass_vals):
                sigma_pb, br = hcs.higgs_cross_section(m_h, m_Z, mass, caphi, f_a, N_c, y_t, T_t3, sigma_ggF, gamma_h_SM)
                higgs_heatmap[i, j] = sigma_pb

        outpng_h = os.path.join(args.output_dir, 'higgs_ggF_Za_cross_section_onGrid_heatmap.png')
        plot_heatmap(mass_vals, caphi_vals, higgs_heatmap, outpng_h,
                     title='Higgs-derived Cross-Section (pp → h → Z a) on same grid',
                     colorbar_label='sigma(pp->h)*BR(h->Za) [pb]', use_log_scale=True, marker_size=args.marker_size)
        print(f'Wrote Higgs-derived heatmap to {outpng_h}')
    except Exception as e:
        print(f'Could not compute Higgs-derived cross-sections: {e}')


if __name__ == '__main__':
    main()
