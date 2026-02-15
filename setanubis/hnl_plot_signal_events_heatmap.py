"""
Plot heat map of expected signal events vs mass and mixing angle numixing.

CORRECTED SIGNAL CALCULATION:
    N_signal = L_int × σ(pp→N+ℓ) × ε_acceptance × ε_selection × BR(N→ℓℓν)

WHERE:
    - L_int: Integrated luminosity (default: 3000 fb⁻¹ for HL-LHC)
    - σ(pp→N+ℓ): Cross-section from MadGraph scan_run text files ('cross' column)
    - ε_acceptance: Geometric/kinematic acceptance = nLLP_Final / nLLP_original
    - ε_selection: Selection efficiency (default: 0.5 = 50%)
    - BR(N→ℓℓν): HNL branching ratio to visible leptons (default: 0.25 for leptonic decays)

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


def extract_cross_section_from_banner(scan, run, runs_base_path='/usera/fs568/set-anubis/HNL_Runs_test'):
    """Extract cross-section from MadGraph scan_run text file.
    
    Parameters:
    -----------
    scan : int
        Scan number
    run : int
        Run number
    runs_base_path : str
        Base path to HNL_Runs_test directory
    
    Returns:
    --------
    float or None
        Cross-section in pb, or None if not found
    """
    scan_dir = f'HNL_Condor_CCDY_qqmu_Scan_{scan}'
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


def calculate_signal_events(df, integrated_lumi=3000, selection_eff=0.5, br_hnl_visible=0.25, runs_base_path='/usera/fs568/set-anubis/HNL_Runs_test'):
    """Calculate N_signal = L_int × σ × ε_acceptance × ε_selection × BR_HNL for each point.
    
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
    
    # Calculate signal events: N = L × σ × ε_acceptance × ε_selection × BR_HNL
    df_result['N_signal'] = (lumi_pb * 
                             df_result['cross_section_pb'] * 
                             df_result['acceptance'] * 
                             selection_eff *
                             br_hnl_visible)
    
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
        Data containing mass, numixing, and the value column
    value_column : str
        Column name to use for heatmap values
    
    Returns:
    --------
    mass_vals : np.array
        Unique mass values (sorted)
    numixing_vals : np.array
        Unique numixing values (sorted)
    heatmap_data : np.array
        2D array of values (numixing x Mass)
    """
    # Filter out rows with unsuccessful status if present
    if 'status' in df.columns:
        df_filtered = df[df['status'] == 'success'].copy()
        print(f"Filtered to {len(df_filtered)} successful runs")
    else:
        df_filtered = df.copy()
    
    # Get unique mass and numixing values
    mass_vals = np.sort(df_filtered['mass'].unique())
    numixing_vals = np.sort(df_filtered['numixing'].unique())
    
    print(f"Mass values: {mass_vals}")
    print(f"numixing values: {numixing_vals}")
    
    # Create 2D array for heatmap (rows=numixing, cols=Mass)
    heatmap_data = np.zeros((len(numixing_vals), len(mass_vals)))
    
    # Fill heatmap data
    for i, numixing in enumerate(numixing_vals):
        for j, mass in enumerate(mass_vals):
            # Find matching rows
            mask = (df_filtered['mass'] == mass) & (df_filtered['numixing'] == numixing)
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
    
    return mass_vals, numixing_vals, heatmap_data


def plot_heatmap(mass_vals, numixing_vals, heatmap_data, output_path, 
                 title="Expected Signal Events", use_log_scale=True, 
                 colorbar_label="N_signal", vmin_override=None, vmax_override=None):
    """
    Create and save heat map plot.
    
    Parameters:
    -----------
    mass_vals : array
        Mass values for x-axis
    numixing_vals : array
        numixing (mixing angle) values for y-axis
    heatmap_data : 2D array
        Heat map data (numixing x Mass)
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

    # Determine color scale
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

    # Unified: collect all points (mass, numixing, value)
    all_points = []
    for i, numixing in enumerate(numixing_vals):
        for j, mass in enumerate(mass_vals):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                all_points.append((mass, numixing, value))

    im = None
    if len(all_points) > 0:
        all_points_arr = np.array(all_points, dtype=object)
        mass_arr = all_points_arr[:,0].astype(float)
        numixing_arr = all_points_arr[:,1].astype(float)
        values_arr = all_points_arr[:,2].astype(float)
        idx_large = []
        idx_small = []
        log_mass_arr = np.log10(mass_arr)
        for c in np.unique(numixing_arr):
            mask = (numixing_arr == c)
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
                ax.scatter(mass_arr[idx_large][is_zero], numixing_arr[idx_large][is_zero],
                           marker='s', s=800, facecolors='lightgrey', edgecolors='black', linewidths=0.5,
                           label='0 LLPs', zorder=2)
            if np.any(~is_zero):
                ax.scatter(mass_arr[idx_large][~is_zero], numixing_arr[idx_large][~is_zero],
                           c=values_arr[idx_large][~is_zero], cmap=cmap, norm=norm,
                           marker='s', s=800, edgecolors='black', linewidths=0.5, alpha=1.0, zorder=2)

        # Plot small dots for intermediate points (grey for zero, colored for nonzero)
        if len(idx_small) > 0:
            idx_small_only = [i for i in idx_small if i not in idx_large]
            if idx_small_only:
                is_zero = (values_arr[idx_small_only] == 0)
                if np.any(is_zero):
                    ax.scatter(mass_arr[idx_small_only][is_zero], numixing_arr[idx_small_only][is_zero],
                               marker='o', s=30, c='lightgrey', edgecolors='none', alpha=0.8, zorder=3)
                if np.any(~is_zero):
                    ax.scatter(mass_arr[idx_small_only][~is_zero], numixing_arr[idx_small_only][~is_zero],
                               c=values_arr[idx_small_only][~is_zero], cmap=cmap, norm=norm,
                               marker='o', s=30, edgecolors='none', alpha=0.8, zorder=3)

        # For colorbar, use all nonzero points
        im = ax.scatter(mass_arr[values_arr != 0], numixing_arr[values_arr != 0], c=values_arr[values_arr != 0], cmap=cmap, norm=norm,
                       marker='o', s=0)  # invisible, for colorbar
    else:
        im = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='s')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=colorbar_label, pad=0.02)

    # Set logarithmic scales for axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('HNL Mass [GeV]', fontsize=14)
    ax.set_ylabel(r'Mixing Angle $|V_{\mu N}|$', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # Annotate every large box (idx_large) with a number ("0" for zero, value for nonzero)
    if len(all_points) > 0 and len(idx_large) > 0:
        for idx in idx_large:
            mval = mass_arr[idx]
            cval = numixing_arr[idx]
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
        description='Plot heat map of expected signal events vs mass and mixing angle numixing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal event calculation:
    N_signal = L_int × σ(pp→N+ℓ) × ε_acceptance × ε_selection × BR(N→ℓℓν)
    
    Where:
    - L_int: Integrated luminosity (default: 3000 fb⁻¹ for HL-LHC)
    - σ(pp→N+ℓ): Cross-section extracted from MadGraph scan_run files
    - ε_acceptance: Geometric/kinematic acceptance (nLLP_Final / nLLP_original)
    - ε_selection: Selection efficiency (default: 0.5 = 50%%)
    - BR(N→ℓℓν): HNL branching ratio to visible leptons (default: 0.25 for leptonic decays)
        """
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        default='/usera/fs568/set-anubis/setanubis/hnl_muon_selection_cutflow_data.csv',
        help='Path to cutflow CSV file'
    )
    parser.add_argument(
        '--runs-path',
        type=str,
        default='/usera/fs568/set-anubis/HNL_Runs_test',
        help='Base path to HNL_Runs_test directory with scan_run files'
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
        '--br-hnl-visible',
        type=float,
        default=0.25,
        help='Branching ratio of HNL to visible leptons (default: 0.25 for leptonic decays)'
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
    print(f"BR(N→ℓℓν): {args.br_hnl_visible}")
    print()
    
    df_with_signals = calculate_signal_events(
        df, 
        integrated_lumi=args.luminosity,
        selection_eff=args.selection_eff,
        br_hnl_visible=args.br_hnl_visible,
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
    
    mass_vals, numixing_vals, heatmap_signal = prepare_heatmap_data(df_valid, 'N_signal')
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
        mass_vals, numixing_vals, heatmap_signal,
        os.path.join(args.output_dir, 'signal_events_heatmap.png'),
        title=f'Expected Signal Events (L={args.luminosity} fb⁻¹)',
        use_log_scale=args.log_scale,
        colorbar_label='Expected Signal Events'
    )
    
    # 2. Acceptance heatmap
    plot_heatmap(
        mass_vals, numixing_vals, heatmap_acceptance,
        os.path.join(args.output_dir, 'acceptance_heatmap.png'),
        title='Geometrical and Kinematic Acceptance',
        use_log_scale=True,
        colorbar_label='Acceptance ε'
    )
    
    # 3. Cross-section heatmap
    plot_heatmap(
        mass_vals, numixing_vals, heatmap_xsec,
        os.path.join(args.output_dir, 'cross_section_heatmap.png'),
        title='Production Cross-Section (pp → N + ℓ)',
        use_log_scale=True,
        colorbar_label='Cross-section (pb)'
    )
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Mass range: {mass_vals.min():.3f} - {mass_vals.max():.3f} GeV")
    print(f"numixing range: {numixing_vals.min():.2e} - {numixing_vals.max():.2e}")
    
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
    print(f"  Mass = {best_row['mass']:.3f} GeV, numixing = {best_row['numixing']:.2e}")
    print(f"  Cross-section = {best_row['cross_section_pb']:.6e} pb")
    print(f"  Acceptance = {best_row['acceptance']:.6e}")
    print(f"  Expected signal events = {best_row['N_signal']:.3f}")
    
    print(f"\nPlots saved to {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
