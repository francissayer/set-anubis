"""
Plot heat map of expected signal events vs mass and coupling CaPhi.

CORRECTED SIGNAL CALCULATION:
    N_signal = L_int × σ(pp→W-+ALP) × ε_acceptance × ε_selection × BR(W→visible) × BR(ALP→visible)

WHERE:
    - L_int: Integrated luminosity (default: 3000 fb⁻¹ for HL-LHC)
    - σ(pp→Z+ALP): Cross-section from MadGraph scan_run text files ('cross' column)
    - ε_acceptance: Geometric/kinematic acceptance = nLLP_Final / nLLP_original
    - ε_selection: Selection efficiency (default: 0.5 = 50%)
    - BR(Z→visible): Z branching ratio to visible final states (default: 0.8)
    - BR(ALP→visible): ALP branching ratio to visible (default: 1.0 for fermion-coupled)

This script extracts the actual cross-section from MadGraph scan_run files rather than
making assumptions about event weights.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from pathlib import Path
import glob


def extract_cross_section_from_banner(scan, run, runs_base_path='/usera/fs568/set-anubis/ALP_W-_Runs'):
    """Extract cross-section from MadGraph scan_run text file.
    
    Parameters:
    -----------
    scan : int
        Scan number
    run : int
        Run number
    runs_base_path : str
        Base path to ALP_Z_Runs directory
    
    Returns:
    --------
    float or None
        Cross-section in pb, or None if not found
    """
    scan_dir = f'ALP_axW-_scan_{scan}'
    events_dir = os.path.join(runs_base_path, scan_dir, 'Events')
    
    # Find the scan_run file (could be scan_run_0[1-5].txt, scan_run_01[-]run_01.txt, etc.)
    scan_run_files = glob.glob(os.path.join(events_dir, 'scan_run_*.txt'))
    
    if not scan_run_files:
        print(f"No scan_run file found for Scan {scan}")
        return None
    
    try:
        # Read the file (should be space-separated)
        scan_file = scan_run_files[0]  # Take the first match
        df_scan = pd.read_csv(scan_file, sep=r'\s+')
        
        # The first column might be '#run_name' or 'run_name' depending on pandas version
        run_col = df_scan.columns[0]
        
        # Look for the matching run
        run_name = f'run_{run:02d}'
        matching_rows = df_scan[df_scan[run_col] == run_name]
        
        if len(matching_rows) == 0:
            print(f"Run {run_name} not found in {scan_file}")
            return None
        
        # Extract cross-section from 'cross' column
        cross_section = float(matching_rows.iloc[0]['cross'])
        return cross_section
        
    except Exception as e:
        print(f"Error reading scan_run file for Scan {scan}, Run {run}: {e}")
        return None


def load_cutflow_data(csv_path):
    """Load cutflow data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def calculate_signal_events(df, integrated_lumi=3000, selection_eff=0.5, br_z_visible=0.8, br_alp_visible=1.0, runs_base_path='/usera/fs568/set-anubis/ALP_W-_Runs'):
    """Calculate N_signal = L_int × σ × ε_acceptance × ε_selection × BR_Z × BR_ALP for each point.
    
    Extracts cross-sections from MadGraph scan_run text files and calculates acceptance
    from unweighted event counts.
    """
    df_result = df.copy()
    lumi_pb = integrated_lumi * 1000.0  # Convert fb^-1 to pb^-1
    
    # Calculate acceptance from unweighted counts
    df_result['acceptance'] = df_result['nLLP_Final'] / df_result['nLLP_original'].replace(0, np.nan)
    
    # Extract cross-sections from scan_run files
    df_result['cross_section_pb'] = np.nan
    
    for idx, row in df_result.iterrows():
        scan = int(row['scan'])
        run = int(row['run'])
        xsec = extract_cross_section_from_banner(scan, run, runs_base_path)
        if xsec is not None:
            df_result.at[idx, 'cross_section_pb'] = xsec
    
    # Calculate signal events: N = L × σ × ε_acceptance × ε_selection × BR_Z × BR_ALP
    df_result['N_signal'] = (lumi_pb * 
                             df_result['cross_section_pb'] * 
                             df_result['acceptance'] * 
                             selection_eff *
                             br_z_visible * 
                             br_alp_visible)
    
    # Print progress
    for _, row in df_result.iterrows():
        if not np.isnan(row['cross_section_pb']):
            print(f"Scan {int(row['scan'])}, Run {int(row['run'])}: "
                  f"σ={row['cross_section_pb']:.4e} pb, "
                  f"ε={row['acceptance']:.6e}, "
                  f"N_signal={row['N_signal']:.3f}")
        else:
            print(f"Scan {int(row['scan'])}, Run {int(row['run'])}: "
                  f"Cross-section not found in scan_run file")
    
    return df_result


def prepare_heatmap_data(df, value_column='N_signal'):
    """
    Prepare data for heat map plotting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing mass, CaPhi, and the value column
    value_column : str
        Column name to use for heatmap values
    
    Returns:
    --------
    mass_vals : np.array
        Unique mass values (sorted)
    caphi_vals : np.array
        Unique CaPhi values (sorted)
    heatmap_data : np.array
        2D array of values (CaPhi x Mass)
    """
    # Filter out rows with unsuccessful status if present
    if 'status' in df.columns:
        df_filtered = df[df['status'] == 'success'].copy()
        print(f"Filtered to {len(df_filtered)} successful runs")
    else:
        df_filtered = df.copy()
    
    # Get unique mass and CaPhi values
    mass_vals = np.sort(df_filtered['mass'].unique())
    caphi_vals = np.sort(df_filtered['CaPhi'].unique())
    
    print(f"Mass values: {mass_vals}")
    print(f"CaPhi values: {caphi_vals}")
    
    # Create 2D array for heatmap (rows=CaPhi, cols=Mass)
    heatmap_data = np.zeros((len(caphi_vals), len(mass_vals)))
    
    # Fill heatmap data
    for i, caphi in enumerate(caphi_vals):
        for j, mass in enumerate(mass_vals):
            # Find matching rows
            mask = (df_filtered['mass'] == mass) & (df_filtered['CaPhi'] == caphi)
            matching_rows = df_filtered[mask]
            
            if len(matching_rows) > 0:
                # If multiple runs for same parameters, take the mean
                value = matching_rows[value_column].mean()
                # Only use non-NaN values
                if not np.isnan(value):
                    heatmap_data[i, j] = value
                else:
                    heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = np.nan
    
    return mass_vals, caphi_vals, heatmap_data


def plot_heatmap(mass_vals, caphi_vals, heatmap_data, output_path, 
                 title="Expected Signal Events", use_log_scale=True, 
                 colorbar_label="N_signal", vmin_override=None, vmax_override=None):
    """
    Create and save heat map plot.
    
    Parameters:
    -----------
    mass_vals : array
        Mass values for x-axis
    caphi_vals : array
        CaPhi coupling values for y-axis
    heatmap_data : 2D array
        Heat map data (CaPhi x Mass)
    output_path : str
        Path to save the output figure
    title : str
        Title for the plot
    use_log_scale : bool
        If True, use logarithmic color scale
    colorbar_label : str
        Label for the colorbar
    vmin_override : float, optional
        Override minimum value for color scale
    vmax_override : float, optional
        Override maximum value for color scale
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Determine color scale using positive (non-zero) values
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    positive_data = valid_data[valid_data > 0]

    if len(positive_data) > 0:
        vmin = vmin_override if vmin_override is not None else np.min(positive_data)
        vmax = vmax_override if vmax_override is not None else np.max(positive_data)
    else:
        vmin = vmin_override if vmin_override is not None else 1e-12
        vmax = vmax_override if vmax_override is not None else 1.0

    if use_log_scale and vmax > 0:
        norm = colors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
        cmap = plt.cm.viridis
    else:
        norm = None
        cmap = plt.cm.viridis

    # Unified: collect all points (mass, caphi, value)
    all_points = []
    for i, caphi in enumerate(caphi_vals):
        for j, mass in enumerate(mass_vals):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                all_points.append((mass, caphi, value))

    im = None
    if len(all_points) > 0:
        all_points_arr = np.array(all_points, dtype=object)
        mass_arr = all_points_arr[:, 0].astype(float)
        caphi_arr = all_points_arr[:, 1].astype(float)
        values_arr = all_points_arr[:, 2].astype(float)
        idx_large = []
        idx_small = []
        log_mass_arr = np.log10(mass_arr)
        for c in np.unique(caphi_arr):
            mask = (caphi_arr == c)
            masses = mass_arr[mask]
            log_masses = log_mass_arr[mask]
            vals = values_arr[mask]
            idxs = np.where(mask)[0]
            if len(masses) == 1:
                idx_large.append(idxs[0])
            else:
                sorted_idx = np.argsort(masses)
                sorted_masses = masses[sorted_idx]
                sorted_log_masses = log_masses[sorted_idx]
                sorted_vals = vals[sorted_idx]
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

        # Plot large boxes for outermost points (grey for zero, colored for nonzero)
        if len(idx_large) > 0:
            is_zero = (values_arr[idx_large] == 0)
            if np.any(is_zero):
                ax.scatter(mass_arr[idx_large][is_zero], caphi_arr[idx_large][is_zero],
                           marker='s', s=800, facecolors='lightgrey', edgecolors='black', linewidths=0.5,
                           label='0 LLPs', zorder=2)
            if np.any(~is_zero):
                ax.scatter(mass_arr[idx_large][~is_zero], caphi_arr[idx_large][~is_zero],
                           c=values_arr[idx_large][~is_zero], cmap=cmap, norm=norm,
                           marker='s', s=800, edgecolors='black', linewidths=0.5, alpha=1.0, zorder=2)

        # Plot small dots for intermediate points (grey for zero, colored for nonzero)
        if len(idx_small) > 0:
            idx_small_only = [i for i in idx_small if i not in idx_large]
            if idx_small_only:
                is_zero = (values_arr[idx_small_only] == 0)
                if np.any(is_zero):
                    ax.scatter(mass_arr[idx_small_only][is_zero], caphi_arr[idx_small_only][is_zero],
                               marker='o', s=30, c='lightgrey', edgecolors='none', alpha=0.8, zorder=3)
                if np.any(~is_zero):
                    ax.scatter(mass_arr[idx_small_only][~is_zero], caphi_arr[idx_small_only][~is_zero],
                               c=values_arr[idx_small_only][~is_zero], cmap=cmap, norm=norm,
                               marker='o', s=30, edgecolors='none', alpha=0.8, zorder=3)

        # For colorbar, use all nonzero points
        im = ax.scatter(mass_arr[values_arr != 0], caphi_arr[values_arr != 0], c=values_arr[values_arr != 0], cmap=cmap, norm=norm,
                       marker='o', s=0)  # invisible, for colorbar
    else:
        im = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='o', s=0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=colorbar_label, pad=0.02)

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('ALP Mass [GeV]', fontsize=14)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # Annotate every large box (idx_large) with formatted number
    if len(all_points) > 0 and len(idx_large) > 0:
        for idx in idx_large:
            mval = mass_arr[idx]
            cval = caphi_arr[idx]
            value = values_arr[idx]
            if value == 0:
                text = '0'
            elif abs(value) < 1:
                text = f'{value:.2e}'
            else:
                text = f'{value:.2f}'
            ax.text(mval, cval, text,
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heat map to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot heat map of expected signal events vs mass and coupling CaPhi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal event calculation:
    N_signal = L_int × σ(pp→Z+ALP) × ε_acceptance × ε_selection × BR(Z→visible) × BR(ALP→visible)
    
    Where:
    - L_int: Integrated luminosity (default: 3000 fb⁻¹ for HL-LHC)
    - σ(pp→Z+ALP): Cross-section extracted from MadGraph scan_run files
    - ε_acceptance: Geometric/kinematic acceptance (nLLP_Final / nLLP_original)
    - ε_selection: Selection efficiency (default: 0.5 = 50%%)
    - BR(Z→visible): Z branching ratio to visible final states (default: 0.8)
    - BR(ALP→visible): ALP branching ratio to visible (default: 1.0 for fermion-coupled)
        """
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        default='/usera/fs568/set-anubis/setanubis/selection_cutflow_data_Wminus.csv',
        help='Path to cutflow CSV file'
    )
    parser.add_argument(
        '--runs-path',
        type=str,
        default='/usera/fs568/set-anubis/ALP_W-_Runs',
        help='Base path to ALP_W-_Runs directory with scan_run files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save output plots (default: current directory)'
    )
    parser.add_argument(
        '--luminosity',
        type=float,
        default=3000.0,
        help='Integrated luminosity in fb^-1 (default: 3000 for HL-LHC)'
    )
    parser.add_argument(
        '--selection-eff',
        type=float,
        default=0.5,
        help='Selection efficiency (default: 0.5 = 50%%)'
    )
    parser.add_argument(
        '--br-z-visible',
        type=float,
        default=0.8,
        help='Branching ratio of Z to visible final states (default: 0.8)'
    )
    parser.add_argument(
        '--br-alp-visible',
        type=float,
        default=1.0,
        help='Branching ratio of ALP to visible final states (default: 1.0 for fermion-coupled)'
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        default=True,
        help='Use logarithmic color scale (default: True)'
    )
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save extended CSV with cross-sections and signal event calculations'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("="*70)
    print("Loading cutflow data...")
    print("="*70)
    df = load_cutflow_data(args.csv)
    
    # Calculate signal events
    print("\n" + "="*70)
    print("Extracting cross-sections from scan_run files and calculating signal events...")
    print("="*70)
    print(f"Integrated luminosity: {args.luminosity} fb^-1")
    print(f"Selection efficiency: {args.selection_eff} ({args.selection_eff*100:.0f}%)")
    print(f"BR(Z→visible): {args.br_z_visible}")
    print(f"BR(ALP→visible): {args.br_alp_visible}")
    print()
    
    df_with_signals = calculate_signal_events(
        df, 
        integrated_lumi=args.luminosity,
        selection_eff=args.selection_eff,
        br_z_visible=args.br_z_visible,
        br_alp_visible=args.br_alp_visible,
        runs_base_path=args.runs_path
    )
    
    # Save extended CSV if requested
    if args.save_csv:
        output_csv = os.path.join(args.output_dir, 'signal_events_data.csv')
        df_with_signals.to_csv(output_csv, index=False)
        print(f"\nSaved extended data to {output_csv}")
    
    # Prepare heatmap data
    print("\n" + "="*70)
    print("Preparing heat map data...")
    print("="*70)
    
    # Filter out rows where signal calculation failed
    df_valid = df_with_signals[~df_with_signals['N_signal'].isna()].copy()
    print(f"Valid parameter points: {len(df_valid)}/{len(df_with_signals)}")
    
    if len(df_valid) == 0:
        print("ERROR: No valid data points with cross-sections found!")
        return
    
    mass_vals, caphi_vals, heatmap_signal = prepare_heatmap_data(df_valid, 'N_signal')
    _, _, heatmap_acceptance = prepare_heatmap_data(df_valid, 'acceptance')
    _, _, heatmap_xsec = prepare_heatmap_data(df_valid, 'cross_section_pb')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)
    
    # 1. Signal events heatmap
    plot_heatmap(
        mass_vals, caphi_vals, heatmap_signal,
        os.path.join(args.output_dir, 'signal_events_heatmap_Wminus.png'),
        title=f'Expected Signal Events (pp → W- + ALP, L={args.luminosity} fb⁻¹)',
        use_log_scale=args.log_scale,
        colorbar_label='Expected Signal Events'
    )
    
    # 2. Acceptance heatmap
    plot_heatmap(
        mass_vals, caphi_vals, heatmap_acceptance,
        os.path.join(args.output_dir, 'acceptance_heatmap_Wminus.png'),
        title='Geometrical and Kinematic Acceptance (pp → W- + ALP)',
        use_log_scale=True,
        colorbar_label='Acceptance ε'
    )
    
    # 3. Cross-section heatmap
    plot_heatmap(
        mass_vals, caphi_vals, heatmap_xsec,
        os.path.join(args.output_dir, 'cross_section_heatmap_Wminus.png'),
        title='Production Cross-Section (pp → W- + ALP)',
        use_log_scale=True,
        colorbar_label='Cross-section (pb)'
    )
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Mass range: {mass_vals.min():.3f} - {mass_vals.max():.3f} GeV")
    print(f"CaPhi range: {caphi_vals.min():.2e} - {caphi_vals.max():.2e}")
    
    valid_signals = df_valid['N_signal'].values
    valid_acceptance = df_valid['acceptance'].values
    valid_xsec = df_valid['cross_section_pb'].values
    
    print(f"\nExpected signal events (L={args.luminosity} fb⁻¹):")
    print(f"  Total: {np.sum(valid_signals):.3f}")
    print(f"  Max: {np.max(valid_signals):.3f}")
    print(f"  Mean: {np.mean(valid_signals):.3f}")
    print(f"  Median: {np.median(valid_signals):.3f}")
    
    print(f"\nAcceptance (ε):")
    print(f"  Max: {np.max(valid_acceptance):.6e}")
    print(f"  Min: {np.min(valid_acceptance[valid_acceptance > 0]):.6e}" if np.any(valid_acceptance > 0) else "  Min: 0")
    print(f"  Mean: {np.mean(valid_acceptance):.6e}")
    
    print(f"\nCross-section (σ):")
    print(f"  Max: {np.max(valid_xsec):.6e} pb")
    print(f"  Min: {np.min(valid_xsec):.6e} pb")
    print(f"  Mean: {np.mean(valid_xsec):.6e} pb")
    
    # Find optimal points
    idx_best_signal = np.argmax(valid_signals)
    best_row = df_valid.iloc[idx_best_signal]
    
    print(f"\nBest point (highest signal):")
    print(f"  Mass = {best_row['mass']:.3f} GeV, CaPhi = {best_row['CaPhi']:.2e}")
    print(f"  Cross-section = {best_row['cross_section_pb']:.6e} pb")
    print(f"  Acceptance = {best_row['acceptance']:.6e}")
    print(f"  Expected signal events = {best_row['N_signal']:.3f}")
    
    print(f"\nPlots saved to {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
