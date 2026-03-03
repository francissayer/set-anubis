#!/usr/bin/env python3
"""
Plot ALP decay positions overlayed on ATLAS cavern geometry.

This script visualizes where ALPs actually decay in the detector/cavern space,
making it easy to understand which decays are in the detection region.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
DEFAULT_DATA_PATH = '/usera/fs568/set-anubis/ALP_Z_Runs'
DEFAULT_OUTPUT_DIR = '/usera/fs568/set-anubis/setanubis'
DEFAULT_CAPHI = 0.001  # 10^-3 coupling

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


def load_alp_decay_positions(filepath, max_decays=None, ip_offset=None):
    """
    Load ALP decay positions.
    
    Parameters:
    -----------
    filepath : str
        Path to pickle file
    max_decays : int, optional
        Maximum number of decays to load (for plotting performance)
    ip_offset : dict, optional
        IP position relative to cavern center {'x': ..., 'y': ..., 'z': ...}
        If provided, transforms coordinates from IP-relative to cavern-center-relative
    
    Returns:
    --------
    dict with keys: 'x', 'y', 'z' (arrays of decay positions), 'mass', 'n_total', 'n_decayed'
    """
    df = pd.read_pickle(filepath)
    alp_data = df[df['PID'] == 9000005]
    
    # Get decayed ALPs
    decayed = alp_data[alp_data['decayVertexDist'] > 1e-10]
    
    x_list = []
    y_list = []
    z_list = []
    
    n_loaded = 0
    for _, row in decayed.iterrows():
        if max_decays and n_loaded >= max_decays:
            break
        
        decay_vtx = row['decayVertex']
        # Coordinates are (x, y, z, t) relative to IP at origin in mm - convert to meters
        # Transform to cavern-center coordinates if IP offset provided
        if ip_offset is not None:
            x_list.append(decay_vtx[0] / 1000.0 + ip_offset['x'])
            y_list.append(decay_vtx[1] / 1000.0 + ip_offset['y'])
            z_list.append(decay_vtx[2] / 1000.0 + ip_offset['z'])
        else:
            x_list.append(decay_vtx[0] / 1000.0)
            y_list.append(decay_vtx[1] / 1000.0)
            z_list.append(decay_vtx[2] / 1000.0)
        n_loaded += 1
    
    mass = alp_data['mass'].iloc[0] if len(alp_data) > 0 else 0.0
    
    return {
        'x': np.array(x_list),
        'y': np.array(y_list),
        'z': np.array(z_list),
        'mass': mass,
        'n_total': len(alp_data),
        'n_decayed': len(decayed),
        'n_plotted': len(x_list)
    }


def plot_geometry_with_decays_xy(plotter, decay_data_list, output_path, caphi):
    """
    Plot XY view with decay positions in side-by-side subplots for each mass.
    """
    n_masses = len(decay_data_list)
    
    # Create grid layout (3 rows x 3 columns for up to 9 masses)
    n_cols = 3
    n_rows = int(np.ceil(n_masses / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_masses > 1 else [axes]
    
    # Use single color for all masses
    color = 'green'
    
    # Plot each mass in its own subplot
    for i, data_dict in enumerate(decay_data_list):
        ax = axes[i]
        x = data_dict['x']
        y = data_dict['y']
        mass = data_dict['mass']
        
        # Get base geometry for this subplot
        plotter._cav.plotCavernXY(ax, plotATLAS=True, plotAcceptance=True)
        
        if len(x) > 0:
            # Plot decay positions with negative zorder to appear below all markers
            ax.scatter(x, y, c=[color], alpha=0.5, s=15, edgecolors='none', zorder=-1)
        
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f'$m_a$ = {mass:.4f} GeV ({len(x)} decays)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-18, 18)
        ax.set_ylim(-18, 25)
    
    # Hide unused subplots
    for i in range(n_masses, len(axes)):
        axes[i].axis('off')
    
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    fig.suptitle(f'ALP Decay Positions - XY View (Z associated production, $C_{{a\\phi}}$ = {caphi_str})', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved XY plot to: {output_path}")
    plt.close()


def plot_geometry_with_decays_xz(plotter, decay_data_list, output_path, caphi):
    """
    Plot XZ view with decay positions in side-by-side subplots for each mass.
    """
    n_masses = len(decay_data_list)
    
    # Create grid layout (3 rows x 3 columns for up to 9 masses)
    n_cols = 3
    n_rows = int(np.ceil(n_masses / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_masses > 1 else [axes]
    
    # Use single color for all masses
    color = 'green'
    
    # Plot each mass in its own subplot
    for i, data_dict in enumerate(decay_data_list):
        ax = axes[i]
        x = data_dict['x']
        z = data_dict['z']
        mass = data_dict['mass']
        
        # Get base geometry for this subplot
        plotter._cav.plotCavernXZ(ax, plotATLAS=True)
        
        if len(x) > 0:
            # Plot decay positions with negative zorder to appear below all markers
            ax.scatter(x, z, c=[color], alpha=0.5, s=15, edgecolors='none', zorder=-1)
        
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('z [m]', fontsize=10)
        ax.set_title(f'$m_a$ = {mass:.4f} GeV ({len(x)} decays)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_masses, len(axes)):
        axes[i].axis('off')
    
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    fig.suptitle(f'ALP Decay Positions - XZ View (Z associated production, $C_{{a\\phi}}$ = {caphi_str})', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved XZ plot to: {output_path}")
    plt.close()


def plot_geometry_with_decays_zy(plotter, decay_data_list, output_path, caphi):
    """
    Plot ZY view with decay positions in side-by-side subplots for each mass.
    """
    n_masses = len(decay_data_list)
    
    # Create grid layout (3 rows x 3 columns for up to 9 masses)
    n_cols = 3
    n_rows = int(np.ceil(n_masses / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_masses > 1 else [axes]
    
    # Use single color for all masses
    color = 'green'
    
    # Plot each mass in its own subplot
    for i, data_dict in enumerate(decay_data_list):
        ax = axes[i]
        z = data_dict['z']
        y = data_dict['y']
        mass = data_dict['mass']
        
        # Get base geometry for this subplot
        plotter._cav.plotCavernZY(ax, plotATLAS=True, plotAcceptance=True)
        
        if len(z) > 0:
            # Plot decay positions with negative zorder to appear below all markers
            ax.scatter(z, y, c=[color], alpha=0.5, s=15, edgecolors='none', zorder=-1)
        
        ax.set_xlabel('z [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f'$m_a$ = {mass:.4f} GeV ({len(z)} decays)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_masses, len(axes)):
        axes[i].axis('off')
    
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    fig.suptitle(f'ALP Decay Positions - ZY View (Z associated production, $C_{{a\\phi}}$ = {caphi_str})', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ZY plot to: {output_path}")
    plt.close()


def main():
    """Main function to generate geometry + decay position plots."""
    parser = argparse.ArgumentParser(
        description='Plot ALP decay positions on cavern geometry at CaPhi = 10^-3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH,
                       help=f'Path to data directory (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--caphi', type=float, default=DEFAULT_CAPHI,
                       help=f'CaPhi coupling value (default: {DEFAULT_CAPHI})')
    parser.add_argument('--max-decays', type=int, default=None,
                       help='Maximum decays to plot per mass (default: None = all, use lower value for performance)')
    
    args = parser.parse_args()
    
    # Import geometry here after setting up paths
    sys.path.insert(0, '/usera/fs568/set-anubis/setanubis')
    from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern
    from SetAnubis.core.Geometry.adapters.plot_matplotlib import MatplotlibGeometryPlotter
    
    cav = ATLASCavern()

    # Use the new default plotting setup from ATLASCavern (as in test_plot.py)
    # Create simple RPCs and set the IP as the position origin so simulation
    # coordinates (which are IP-relative) map correctly to the cavern frame.
    try:
        cav.createSimpleRPCs([cav.archRadius-0.2, cav.archRadius-1.2], RPCthickness=0.06)
        cav.RPCMaxRadius = cav.archRadius-1.2-0.5
    except Exception:
        # If createSimpleRPCs is not available, continue — plotting still works.
        pass

    cav.posOrigin = [cav.IP["x"], cav.IP["y"], cav.IP["z"]]

    plotter = MatplotlibGeometryPlotter(cav)
    
    caphi_str = f"{args.caphi:.6f}" if args.caphi < 0.01 else f"{args.caphi:.3f}"
    print("\n" + "="*80)
    print("ALP DECAY POSITIONS ON GEOMETRY")
    print(f"CaPhi = {caphi_str}")
    print("="*80)
    
    # Find available data files
    print(f"\nSearching for data with CaPhi = {caphi_str} in: {args.data_path}")
    data_info = find_available_data_by_caphi(args.data_path, args.caphi)
    
    if not data_info:
        print(f"\nError: No data files found for CaPhi = {caphi_str}")
        return
    
    print(f"\nFound {len(data_info)} mass points")
    
    # Load decay position data for all masses
    print("\nLoading decay positions...")
    decay_data_list = []
    for scan_num, run_num, filepath, mass, caphi in data_info:
        print(f"  Loading Scan {scan_num}, Run {run_num} (mass = {mass:.4f} GeV)...")
        # Simulation positions are IP-relative in mm; convert to cavern-centre meters
        ip_offset = {"x": cav.posOrigin[0], "y": cav.posOrigin[1], "z": cav.posOrigin[2]}
        data_dict = load_alp_decay_positions(filepath, max_decays=args.max_decays, ip_offset=ip_offset)
        print(f"    Total: {data_dict['n_total']}, Decayed: {data_dict['n_decayed']}, "
              f"Plotted: {data_dict['n_plotted']}")
        decay_data_list.append(data_dict)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating geometry plots with decay positions...")
    
    # XY view (transverse view - looking down the beamline)
    xy_path = os.path.join(args.output_dir, 'alp_decays_geometry_Z_xy.pdf')
    plot_geometry_with_decays_xy(plotter, decay_data_list, xy_path, args.caphi)
    
    # XZ view (side view along x)
    xz_path = os.path.join(args.output_dir, 'alp_decays_geometry_Z_xz.pdf')
    plot_geometry_with_decays_xz(plotter, decay_data_list, xz_path, args.caphi)
    
    # ZY view (side view along z/beamline)
    zy_path = os.path.join(args.output_dir, 'alp_decays_geometry_Z_zy.pdf')
    plot_geometry_with_decays_zy(plotter, decay_data_list, zy_path, args.caphi)
    
    print("\n" + "="*80)
    print("PLOTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
