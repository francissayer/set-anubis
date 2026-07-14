"""
Plot heat map of expected signal events vs mass and coupling CaPhi.

CORRECTED SIGNAL CALCULATION:
    N_signal = L_int × σ(pp→Z+ALP) × ε_acceptance × ε_selection × BR(Z→visible)

WHERE:
    - L_int: Integrated luminosity (default: 3000 fb⁻¹ for HL-LHC)
    - σ(pp→Z+ALP): Cross-section from MadGraph scan_run text files ('cross' column)
    - ε_acceptance: Geometric/kinematic acceptance = nLLP_Final / nLLP_original
    - ε_selection: Selection efficiency (default: 0.5 = 50%)
    - BR(Z→visible): Z branching ratio to visible final states (default: 0.8)

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
import mplhep as hep

# Use ATLAS style for consistent aesthetics with sensitivity-contour plots
plt.style.use(hep.style.ATLAS)


def extract_cross_section_from_banner(scan, run, runs_base_path='/usera/fs568/set-anubis/ALP_W+_Runs'):
    """Extract cross-section from MadGraph scan_run text file.
    
    Parameters:
    -----------
    scan : int
        Scan number
    run : int
        Run number
    runs_base_path : str
        Base path to ALP_W+_Runs directory
    
    Returns:
    --------
    float or None
        Cross-section in pb, or None if not found
    """
    scan_dir = f'ALP_axW+_scan_{scan}'
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


def calculate_signal_events(df, integrated_lumi=3000, selection_eff=0.5, br_z_visible=1.0, runs_base_path='/usera/fs568/set-anubis/ALP_W+_Runs'):
    """Calculate N_signal = L_int × σ × ε_acceptance × ε_selection × BR_Z for each point.
    
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
    
    # Calculate signal events: N = L × σ × ε_acceptance × ε_selection × BR_Z
    df_result['N_signal'] = (lumi_pb * 
                             df_result['cross_section_pb'] * 
                             df_result['acceptance'] * 
                             selection_eff *
                             br_z_visible)
    
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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine color scale using positive (non-zero) values
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    positive_data = valid_data[valid_data > 0]

    if len(positive_data) > 0:
        vmin = vmin_override if vmin_override is not None else np.min(positive_data)
        vmax = vmax_override if vmax_override is not None else np.max(positive_data)
    else:
        vmin = vmin_override if vmin_override is not None else 1e-12
        vmax = vmax_override if vmax_override is not None else 1.0

    # Use a copy of the colormap so we can set the 'bad' (NaN) color
    cmap = plt.get_cmap('viridis').copy()
    try:
        cmap.set_bad('lightgrey')
    except Exception:
        pass

    if use_log_scale and vmax > 0:
        norm = colors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    else:
        norm = None

    # Draw grid markers: empty grey circles for full grid, colored filled circles for positive values
    mv = np.asarray(mass_vals, dtype=float)
    cv = np.asarray(caphi_vals, dtype=float)
    GX, GY = np.meshgrid(mv, cv)
    GXr = GX.ravel()
    GYr = GY.ravel()

    # empty circles for the full grid layout (like contours script)
    ax.scatter(GXr, GYr, facecolors='none', edgecolors='lightgrey', s=120, linewidths=0.8, zorder=2)

    # Colored filled circles for positive values
    vals_flat = heatmap_data.ravel()
    pos_mask = ~np.isnan(vals_flat) & (vals_flat > 0)
    if np.any(pos_mask):
        mass_pos = GXr[pos_mask]
        caphi_pos = GYr[pos_mask]
        vals_pos = vals_flat[pos_mask].astype(float)
        sc = ax.scatter(mass_pos, caphi_pos, c=vals_pos, cmap=cmap, norm=norm,
                        marker='o', s=140, edgecolors='none', alpha=0.95, zorder=3)
    else:
        sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='o', s=0)

    # Add colorbar (use the colored scatter artist if available)
    if 'sc' not in locals() or sc is None:
        sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    cbar = fig.colorbar(sc, ax=ax, label=colorbar_label, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Use geometric bin edges (same logic as sensitivity contours) so the
    # discrete grid fills the plot without extra whitespace.
    try:
        mv = np.asarray(mass_vals, dtype=float)
        cv = np.asarray(caphi_vals, dtype=float)

        if mv.size > 1 and np.all(mv > 0):
            gx = np.sqrt(mv[:-1] * mv[1:])
            x_edges = np.concatenate(([mv[0] ** 2 / gx[0]], gx, [mv[-1] ** 2 / gx[-1]]))
            ax.set_xlim(x_edges[0], x_edges[-1])
        else:
            ax.set_xlim(float(mv.min()) * 0.9, float(mv.max()) * 1.1)

        if cv.size > 1 and np.all(cv > 0):
            gy = np.sqrt(cv[:-1] * cv[1:])
            y_edges = np.concatenate(([cv[0] ** 2 / gy[0]], gy, [cv[-1] ** 2 / gy[-1]]))
            ax.set_ylim(y_edges[0], y_edges[-1])
        else:
            ax.set_ylim(float(cv.min()) * 0.9, float(cv.max()) * 1.1)
    except Exception:
        try:
            ax.relim()
            ax.autoscale_view()
            ax.margins(0.06)
        except Exception:
            pass

    # Add unlabeled dashed grey vertical lines at twice SM charged-lepton and quark masses
    sm_m = {
        'd': 0.00504, 'u': 0.00255, 's': 0.101, 'c': 1.27,
        'b': 4.7, 't': 172.0, 'e': 0.000511, 'mu': 0.10566, 'tau': 1.777,
    }
    fermions = ['e', 'mu', 'tau', 'u', 'd', 's', 'c', 'b', 't']
    x0, x1 = ax.get_xlim()
    for f in fermions:
        m2 = 2.0 * sm_m[f]
        if m2 >= x0 and m2 <= x1:
            ax.axvline(m2, color='grey', linestyle='--', linewidth=2.0, zorder=4, alpha=0.8)

    # Labels and title (match contour aesthetics)
    ax.set_xlabel(r'ALP Mass $m_a$ [GeV]', fontsize=20)
    ax.set_ylabel(r'Fermion Coupling $C_{a\phi}$', fontsize=20)
    #ax.set_title(title, fontsize=14, pad=20)
    ax.tick_params(labelsize=14)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

    # (No numeric annotations on markers — use colored circles like contour plot)

    plt.tight_layout()
    out_path = Path(output_path)
    os.makedirs(out_path.parent or '.', exist_ok=True)
    # Save PNG (raster) and PDF (vector) for publication-quality output
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    try:
        pdf_path = out_path.with_suffix('.pdf')
        plt.savefig(str(pdf_path), bbox_inches='tight')
    except Exception:
        pass
    print(f"Saved heat map to {out_path}")
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
        default='/usera/fs568/set-anubis/setanubis/ALP_Wplus_Analysis_2/selection_cutflow_W+.csv',
        help='Path to cutflow CSV file'
    )
    parser.add_argument(
        '--runs-path',
        type=str,
        default='/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_1',
        help='Base path to ALP_W+_Runs directory with scan_run files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/usera/fs568/set-anubis/setanubis/ALP_Wplus_Analysis_2',
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
        default=1.0,
        help='Branching ratio of Z to visible final states (default: 1.0)'
    )
    # ALP visible branching ratio intentionally omitted — acceptance accounts for visible decays
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
    print()
    
    df_with_signals = calculate_signal_events(
        df, 
        integrated_lumi=args.luminosity,
        selection_eff=args.selection_eff,
        br_z_visible=args.br_z_visible,
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
        os.path.join(args.output_dir, 'signal_events_heatmap_Z.png'),
        title=f'Expected Signal Events (pp → Z + ALP, L={args.luminosity} fb⁻¹)',
        use_log_scale=args.log_scale,
        colorbar_label=r'Expected Number of Signal Events $N_\text{sig}$'
    )
    
    # 2. Acceptance heatmap
    plot_heatmap(
        mass_vals, caphi_vals, heatmap_acceptance,
        os.path.join(args.output_dir, 'acceptance_heatmap_Z.png'),
        title='Geometrical and Kinematic Acceptance (pp → Z + ALP)',
        use_log_scale=True,
        colorbar_label='Acceptance ε'
    )
    
    # 3. Cross-section heatmap
    plot_heatmap(
        mass_vals, caphi_vals, heatmap_xsec,
        os.path.join(args.output_dir, 'cross_section_heatmap_Z.png'),
        title='Production Cross-Section (pp → Z + ALP)',
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
