#!/usr/bin/env python3
"""
Plot ALP 3D distance from IP distributions.

This script loads processed ALP event data and creates visualizations of:
1. Distance from IP distributions for each mass
2. Overlay plot showing all masses
3. Statistical summary of distance from IP vs mass

The distance from IP = sqrt(x^2 + y^2 + z^2) represents how far the ALP
traveled from the interaction point before decaying. This gives physical
insight into decay lengths.

NOTE: This is different from r_perp = sqrt(x^2 + y^2), which is what the
NotInATLAS selection cut checks (perpendicular distance from beamline).

This version reads actual parameters from params.dat files to correctly identify
which runs correspond to which CaPhi values.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
DEFAULT_DATA_PATH = '/usera/fs568/set-anubis/ALP_Z_Runs'
DEFAULT_OUTPUT_DIR = '/usera/fs568/set-anubis/setanubis'
DEFAULT_CAPHI = 0.001  # 10^-3 coupling

# Reference distances for visualization
ATLAS_TRACKING_RADIUS = 9.5  # meters - typical scale of ATLAS inner detector
CAVERN_DETECTION_RADIUS = 20.0  # meters - approximate typical distance for visualization
# Note: These are approximate reference scales for 3D distance from IP
# ============================================================================


def parse_params_file(params_path):
    """Parse params.dat file to extract ALP parameters."""
    params = {}
    try:
        with open(params_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    parts = line.split('=')
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        try:
                            value = float(parts[1].strip())
                            params[key] = value
                        except ValueError:
                            pass
    except Exception as e:
        pass
    return params


def find_available_data_by_caphi(data_path, caphi_target):
    """
    Find all available data for the specified CaPhi value.
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing ALP data files
    caphi_target : float
        Target CaPhi value
    
    Returns:
    --------
    list of tuples: (scan_num, run_num, filepath, mass, caphi)
    """
    data_info = []
    
    scan_dirs = sorted(glob.glob(os.path.join(data_path, 'ALP_axZ_scan_*')))
    
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        match = re.search(r'scan_(\d+)', scan_name)
        if not match:
            continue
        scan_num = int(match.group(1))
        
        events_dir = os.path.join(scan_dir, 'Events')
        if not os.path.exists(events_dir):
            continue
        
        run_dirs = sorted([d for d in glob.glob(os.path.join(events_dir, 'run_*')) 
                          if 'decayed' not in d])
        
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            match_run = re.search(r'run_(\d+)', run_name)
            if not match_run:
                continue
            run_num = int(match_run.group(1))
            
            params_path = os.path.join(run_dir, 'params.dat')
            if not os.path.exists(params_path):
                continue
            
            params = parse_params_file(params_path)
            caphi = params.get('caphi', None)
            ma = params.get('ma', None)
            
            if caphi is None or ma is None:
                continue
            
            # Check if this matches our target CaPhi (with small tolerance)
            if abs(caphi - caphi_target) < 1e-6:
                # Check if pickle file exists
                pkl_file = os.path.join(data_path, f'ALP_Z_df_Scan_{scan_num}_Run_{run_num}.pkl')
                if os.path.exists(pkl_file):
                    data_info.append((scan_num, run_num, pkl_file, ma, caphi))
                    print(f"  Found Scan {scan_num}, Run {run_num}: mass = {ma:.4f} GeV, CaPhi = {caphi}")
    
    # Sort by mass
    data_info.sort(key=lambda x: x[3])
    return data_info


def load_alp_decay_distances(filepath, distance_cut=None):
    """
    Load ALP 3D distances from IP.
    
    Calculates distance_from_ip = sqrt(x^2 + y^2 + z^2), which represents
    how far the ALP traveled before decaying. This is the actual decay length.
    
    Also checks if ALPs are in the actual detection region (inCavern).
    
    Parameters:
    -----------
    filepath : str
        Path to pickle file
    distance_cut : float, optional
        Maximum distance to include (in meters)
    
    Returns:
    --------
    dict with keys: 'distances' (distance from IP values), 'mass', 'n_total', 
                    'n_decayed', 'n_in_detection_region'
    """
    df = pd.read_pickle(filepath)
    alp_data = df[df['PID'] == 9000005]
    
    # Calculate 3D distance from IP for each ALP
    # and check if in typical detection region
    dist_list = []
    in_detection_count = 0
    
    for _, row in alp_data.iterrows():
        # Only include ALPs that actually decayed
        if row['decayVertexDist'] > 1e-10:
            decay_vtx = row['decayVertex']
            # Coordinates are (x, y, z, t) relative to IP at origin in mm - convert to meters
            x_ip, y_ip, z_ip = decay_vtx[0] / 1000.0, decay_vtx[1] / 1000.0, decay_vtx[2] / 1000.0
            distance_from_ip = np.sqrt(x_ip**2 + y_ip**2 + z_ip**2)
            dist_list.append(distance_from_ip)
            
            # Check if in typical detection region: between ATLAS and cavern scales
            if ATLAS_TRACKING_RADIUS < distance_from_ip < CAVERN_DETECTION_RADIUS:
                in_detection_count += 1
    
    dist_array = np.array(dist_list)
    
    # Apply distance cut if specified
    if distance_cut is not None:
        dist_array = dist_array[dist_array <= distance_cut]
    
    mass = alp_data['mass'].iloc[0] if len(alp_data) > 0 else 0.0
    
    return {
        'distances': dist_array,  # 3D distance from IP
        'mass': mass,
        'n_total': len(alp_data),
        'n_decayed': len(dist_array),
        'n_in_detection_region': in_detection_count,  # ALPs in typical detection region (9.5m < d < 20m)
        'decay_fraction': len(dist_array) / len(alp_data) if len(alp_data) > 0 else 0
    }


def plot_individual_distributions(data_dict_list, output_dir, caphi, use_log=True):
    """
    Plot individual distance from IP distributions for each mass point.
    
    Parameters:
    -----------
    data_dict_list : list of dicts
        List of dictionaries from load_alp_decay_distances
    output_dir : str
        Directory to save plots
    use_log : bool
        Use logarithmic x-axis
    """
    n_plots = len(data_dict_list)
    n_cols = min(3, n_plots)  # Don't create more columns than needed
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, data_dict in enumerate(data_dict_list):
        ax = axes[idx]
        distances = data_dict['distances']
        mass = data_dict['mass']
        
        if len(distances) > 0:
            # Create histogram
            if use_log:
                bins = np.logspace(np.log10(max(distances.min(), 1e-3)), 
                                   np.log10(distances.max()), 50)
            else:
                bins = 50
            
            ax.hist(distances, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Statistics
            median_dist = np.median(distances)
            mean_dist = np.mean(distances)
            
            # Add vertical lines for median and mean
            ax.axvline(median_dist, color='red', linestyle='--', linewidth=2, 
                      label=f'Median: {median_dist:.2e} m')
            ax.axvline(mean_dist, color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_dist:.2e} m')
            
            # Add ATLAS tracking scale (lower reference)
            ax.axvline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.8, label=f'ATLAS scale (~{ATLAS_TRACKING_RADIUS} m)')
            # Add cavern detection scale (upper reference)
            ax.axvline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', 
                      linewidth=1.5, alpha=0.8, label=f'Cavern scale (~{CAVERN_DETECTION_RADIUS} m)')
            # Shade typical detection region between boundaries
            ax.axvspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green',
                      label='Typical Detection Region')
            
            if use_log:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'Distance from IP (m)', fontsize=10)
            ax.set_ylabel('Number of ALPs', fontsize=10)
            ax.set_title(f'$m_a$ = {mass:.3f} GeV\n'
                        f'$N_{{decay}}$ = {data_dict["n_decayed"]}, '
                        f'In Region (9.5-20m): {data_dict.get("n_in_detection_region", 0)}',
                        fontsize=11)
            ax.legend(fontsize=7, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No decayed ALPs', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(f'$m_a$ = {mass:.3f} GeV', fontsize=11)
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    plt.suptitle(fr'ALP Distance from IP Distributions ($C_{{a\phi}}$ = {caphi_str})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'alp_distance_ip_individual.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved individual distributions to: {output_path}")
    plt.close()


def plot_overlay_distribution(data_dict_list, output_dir, caphi, use_log=True):
    """
    Plot overlay of distance from IP distributions for all mass points.
    
    Parameters:
    -----------
    data_dict_list : list of dicts
        List of dictionaries from load_alp_decay_distances
    output_dir : str
        Directory to save plot
    use_log : bool
        Use logarithmic x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate colors using a colormap
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(data_dict_list)))
    
    for idx, (data_dict, color) in enumerate(zip(data_dict_list, colors_list)):
        distances = data_dict['distances']
        mass = data_dict['mass']
        
        if len(distances) > 0:
            # Create histogram
            if use_log:
                bins = np.logspace(-3, np.log10(max([d['distances'].max() 
                                   for d in data_dict_list if len(d['distances']) > 0])), 60)
            else:
                bins = 60
            
            ax.hist(distances, bins=bins, alpha=0.6, edgecolor='black', 
                   linewidth=0.5, label=f'$m_a$ = {mass:.3f} GeV', color=color)
    
    # Add ATLAS tracking scale (lower reference)
    ax.axvline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', 
              linewidth=2.5, alpha=0.8, label=f'ATLAS Scale (~{ATLAS_TRACKING_RADIUS} m)', zorder=10)
    # Add cavern detection scale (upper reference)
    ax.axvline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', 
              linewidth=2.5, alpha=0.8, label=f'Cavern Scale (~{CAVERN_DETECTION_RADIUS} m)', zorder=10)
    # Shade typical detection region between boundaries
    ax.axvspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green', 
              label='Typical Detection Region', zorder=0)
    
    if use_log:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Distance from IP (m)', fontsize=14)
    ax.set_ylabel('Number of ALPs', fontsize=14)
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    ax.set_title(fr'ALP Distance from IP Distributions: All Masses ($C_{{a\phi}}$ = {caphi_str})', 
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'alp_distance_ip_overlay.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overlay distribution to: {output_path}")
    plt.close()


def plot_statistics_vs_mass(data_dict_list, output_dir, caphi):
    """
    Plot distance from IP statistics (mean, median, percentiles) vs mass.
    
    Parameters:
    -----------
    data_dict_list : list of dicts
        List of dictionaries from load_alp_decay_distances
    output_dir : str
        Directory to save plot
    """
    masses = []
    medians = []
    means = []
    p25 = []  # 25th percentile
    p75 = []  # 75th percentile
    p10 = []  # 10th percentile
    p90 = []  # 90th percentile
    
    for data_dict in data_dict_list:
        distances = data_dict['distances']
        if len(distances) > 0:
            masses.append(data_dict['mass'])
            medians.append(np.median(distances))
            means.append(np.mean(distances))
            p25.append(np.percentile(distances, 25))
            p75.append(np.percentile(distances, 75))
            p10.append(np.percentile(distances, 10))
            p90.append(np.percentile(distances, 90))
    
    masses = np.array(masses)
    medians = np.array(medians)
    means = np.array(means)
    p25 = np.array(p25)
    p75 = np.array(p75)
    p10 = np.array(p10)
    p90 = np.array(p90)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot median and mean
    ax.plot(masses, medians, 'o-', linewidth=2, markersize=8, 
           label='Median', color='red', zorder=3)
    ax.plot(masses, means, 's-', linewidth=2, markersize=8, 
           label='Mean', color='blue', zorder=3)
    
    # Plot percentile bands
    ax.fill_between(masses, p25, p75, alpha=0.3, color='gray', 
                    label='25th-75th percentile')
    ax.fill_between(masses, p10, p90, alpha=0.2, color='lightgray', 
                    label='10th-90th percentile')
    
    # Add ATLAS tracking scale as horizontal line (lower reference)
    ax.axhline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'ATLAS Scale (~{ATLAS_TRACKING_RADIUS} m)', zorder=10)
    # Add cavern detection scale as horizontal line (upper reference)
    ax.axhline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'Cavern Scale (~{CAVERN_DETECTION_RADIUS} m)', zorder=10)
    # Shade typical detection region between boundaries
    ax.axhspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green', 
              label='Typical Detection Region', zorder=0)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass $m_a$ (GeV)', fontsize=14)
    ax.set_ylabel(r'Distance from IP (m)', fontsize=14)
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    ax.set_title(fr'ALP Distance from IP Statistics vs Mass ($C_{{a\phi}}$ = {caphi_str})', 
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # --- draw kinematic thresholds (if UFO model available) ---
    try:
        from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        UFO_PATH = os.path.join(CURRENT_DIR, '..', 'Assets', 'UFO', 'ALP_linear_UFO_WIDTH')
        sa = SetAnubisInterface(UFO_PATH)
        threshold_params = ['Me', 'MMU', 'MTA', 'MU', 'MD', 'MC', 'MS', 'MB', 'MW', 'MZ', 'MT']
        thresholds = {}
        for p in threshold_params:
            try:
                pv = sa.get_parameter_value(p)
                val = float(pv.real) if hasattr(pv, 'real') else float(pv)
                thresholds[p] = 2.0 * val
            except Exception:
                pass

        param_label = {'Me':'e', 'MMU':'mu', 'MTA':'tau', 'MU':'u', 'MD':'d', 'MC':'c', 'MS':'s', 'MB':'b', 'MW':'W', 'MZ':'Z', 'MT':'t'}
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        for pname, thr in thresholds.items():
            if thr >= masses.min() and thr <= masses.max():
                ax.axvline(thr, color='gray', linestyle='--', alpha=0.6)
                lab = param_label.get(pname, pname)
                ax.text(thr * 1.001, ymax * 0.9, f"2*{lab}", rotation=90, color='gray', fontsize=8, va='top', ha='left')
    except Exception:
        # If SetAnubis or UFO not available, skip thresholds silently
        pass

    output_path = os.path.join(output_dir, 'alp_distance_ip_vs_mass.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistics vs mass to: {output_path}")
    plt.close()


def print_summary_table(data_dict_list, caphi):
    """
    Print a summary table of distance from IP statistics for all masses.
    
    Parameters:
    -----------
    data_dict_list : list of dicts
        List of dictionaries from load_alp_decay_distances
    caphi : float
        CaPhi value
    """
    print("\n" + "="*125)
    print("ALP DISTANCE FROM IP SUMMARY")
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    print(f"CaPhi = {caphi_str}")
    print("="*125)
    print("NOTE: 'In Region' = ALPs with distance from IP between 9.5m and 20m (typical detection region)")
    print("-"*125)
    print(f"{'Mass (GeV)':<12} {'N_total':<10} {'N_decay':<10} {'In Region':<15} "
          f"{'Median (m)':<15} {'Mean (m)':<15} {'Min (m)':<15} {'Max (m)':<15}")
    print("-"*125)
    
    for data_dict in data_dict_list:
        mass = data_dict['mass']
        n_total = data_dict['n_total']
        n_decay = data_dict['n_decayed']
        n_in_region = data_dict.get('n_in_detection_region', 0)
        distances = data_dict['distances']
        
        if len(distances) > 0:
            median = np.median(distances)
            mean = np.mean(distances)
            min_d = distances.min()
            max_d = distances.max()
            
            print(f"{mass:<12.4f} {n_total:<10} {n_decay:<10} {n_in_region:<15} "
                  f"{median:<15.3e} {mean:<15.3e} {min_d:<15.3e} {max_d:<15.3e}")
        else:
            print(f"{mass:<12.4f} {n_total:<10} {n_decay:<10} {n_in_region:<15} "
                  f"{'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*125)


def main():
    """Main function to generate distance from IP plots."""
    parser = argparse.ArgumentParser(
        description='Plot ALP 3D distance from IP distributions at CaPhi = 10^-3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script plots distance from IP = sqrt(x^2 + y^2 + z^2), which represents
how far the ALP traveled from the interaction point before decaying (decay length).

This is different from r_perp = sqrt(x^2 + y^2), which is what the NotInATLAS
selection cut checks (perpendicular distance from beamline).
        """
    )
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH,
                       help=f'Path to data directory (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--caphi', type=float, default=DEFAULT_CAPHI,
                       help=f'CaPhi coupling value (default: {DEFAULT_CAPHI})')
    parser.add_argument('--max-distance', type=float, default=None,
                       help='Maximum distance to plot (meters)')
    parser.add_argument('--linear', action='store_true',
                       help='Use linear x-axis instead of logarithmic')
    
    args = parser.parse_args()
    
    caphi_str = f"{args.caphi:.6f}" if args.caphi < 0.01 else f"{args.caphi:.3f}"
    print("\n" + "="*80)
    print("ALP DISTANCE FROM IP ANALYSIS")
    print(f"CaPhi = {caphi_str}")
    print("="*80)
    print("\nNOTE: This plots 3D distance from IP = sqrt(x^2 + y^2 + z^2),")
    print("      which represents the ALP decay length (how far it traveled).\n")
    
    # Find available data files
    print(f"\nSearching for data with CaPhi = {caphi_str} in: {args.data_path}")
    data_info = find_available_data_by_caphi(args.data_path, args.caphi)
    
    if not data_info:
        print(f"\nError: No data files found for CaPhi = {caphi_str}")
        print(f"Make sure data files exist in: {args.data_path}")
        print(f"\nTip: Use extract_alp_parameters.py to see all available CaPhi values")
        return
    
    print(f"\nFound {len(data_info)} mass points")
    
    # Load distance data for all masses
    print("\nLoading distance from IP data...")
    data_dict_list = []
    for scan_num, run_num, filepath, mass, caphi in data_info:
        print(f"  Loading Scan {scan_num}, Run {run_num} (mass = {mass:.4f} GeV)...")
        data_dict = load_alp_decay_distances(filepath, distance_cut=args.max_distance)
        print(f"    Decayed: {data_dict['n_decayed']}, In Region (9.5-20m): {data_dict.get('n_in_detection_region', 0)}")
        data_dict_list.append(data_dict)
    
    # Print summary table
    print_summary_table(data_dict_list, args.caphi)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    use_log = not args.linear
    
    plot_individual_distributions(data_dict_list, args.output_dir, args.caphi, use_log=use_log)
    plot_overlay_distribution(data_dict_list, args.output_dir, args.caphi, use_log=use_log)
    plot_statistics_vs_mass(data_dict_list, args.output_dir, args.caphi)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
