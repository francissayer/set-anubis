"""
Plot heat map of surviving LLPs vs mass and mixing angle numixing.

This script reads the hnl_selection_cutflow_data.csv file and creates a 2D heat map
showing the number of surviving LLPs after all selection cuts as a function of
the HNL mass and numixing (mixing angle) parameter.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from matplotlib.ticker import LogLocator, LogFormatter


def load_cutflow_data(csv_path):
    """Load cutflow data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_heatmap_data(df, use_weighted=False):
    """
    Prepare data for heat map plotting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cutflow data containing mass, numixing, and surviving LLP counts
    use_weighted : bool
        If True, use weighted counts (nLLP_Final_weighted), otherwise use raw counts (n_surviving_llps)
    
    Returns:
    --------
    mass_vals : np.array
        Unique mass values (sorted)
    numixing_vals : np.array
        Unique numixing values (sorted)
    heatmap_data : np.array
        2D array of surviving LLP counts (numixing x Mass)
    """
    # Choose the appropriate column
    if use_weighted:
        llp_column = 'nLLP_Final_weighted'
    else:
        llp_column = 'n_surviving_llps'
    
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
                heatmap_data[i, j] = matching_rows[llp_column].mean()
            else:
                heatmap_data[i, j] = np.nan
    
    return mass_vals, numixing_vals, heatmap_data


def plot_heatmap(mass_vals, numixing_vals, heatmap_data, output_path, 
                 title="Surviving LLPs", use_log_scale=True, use_weighted=False):
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
    use_weighted : bool
        If True, indicates weighted counts are being plotted
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Determine color scale
    vmin = np.nanmin(heatmap_data[heatmap_data > 0]) if np.any(heatmap_data > 0) else 1e-12
    vmax = np.nanmax(heatmap_data)

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
                       marker='o', s=0)
    else:
        im = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='s')

    # Add colorbar
    cbar_label = 'Weighted Surviving LLPs' if use_weighted else 'Surviving LLPs (count)'
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02)

    # Set logarithmic scales
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


def plot_multiple_views(mass_vals, numixing_vals, heatmap_data, heatmap_weighted, output_dir, df):
    """Create multiple views of the data."""
    if output_dir != '.':
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Unweighted counts with log scale
    plot_heatmap(mass_vals, numixing_vals, heatmap_data,
                os.path.join(output_dir, 'surviving_llps_heatmap.png'),
                title='Surviving LLPs vs Mass and Mixing Angle',
                use_log_scale=False, use_weighted=False)
    
    # 2. Weighted counts with log scale
    plot_heatmap(mass_vals, numixing_vals, heatmap_weighted,
                os.path.join(output_dir, 'surviving_llps_heatmap_weighted_log.png'),
                title='Surviving LLPs (Weighted) vs Mass and Mixing Angle',
                use_log_scale=True, use_weighted=True)
    
    # 3. Weighted counts with linear scale
    plot_heatmap(mass_vals, numixing_vals, heatmap_weighted,
                os.path.join(output_dir, 'surviving_llps_heatmap_weighted_linear.png'),
                title='Surviving LLPs (Weighted) vs Mass and Mixing Angle',
                use_log_scale=False, use_weighted=True)
    
    # 4. Create a plot showing survival efficiency (surviving / original)
    df_success = df[df['status'] == 'success'].copy()
    
    efficiency_data = np.zeros((len(numixing_vals), len(mass_vals)))
    for i, numixing in enumerate(numixing_vals):
        for j, mass in enumerate(mass_vals):
            mask = (df_success['mass'] == mass) & (df_success['numixing'] == numixing)
            matching = df_success[mask]
            if len(matching) > 0:
                row = matching.iloc[0]
                if row['nLLP_original_weighted'] > 0:
                    efficiency_data[i, j] = row['nLLP_Final_weighted'] / row['nLLP_original_weighted']
                else:
                    efficiency_data[i, j] = 0
    
    plot_heatmap(mass_vals, numixing_vals, efficiency_data,
                os.path.join(output_dir, 'surviving_llps_efficiency.png'),
                title='LLP Selection Efficiency vs Mass and Mixing Angle',
                use_log_scale=True, use_weighted=False)


def main():
    parser = argparse.ArgumentParser(
        description='Plot heat map of surviving LLPs vs mass and mixing angle numixing'
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        default='/usera/fs568/set-anubis/setanubis/hnl_muon_selection_cutflow_data.csv',
        help='Path to cutflow CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save output plots (default: current directory)'
    )
    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Use weighted counts instead of raw counts'
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        default=True,
        help='Use logarithmic color scale (default: True)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading cutflow data...")
    df = load_cutflow_data(args.csv)
    
    # Prepare heatmap data (unweighted)
    print("\nPreparing unweighted heat map data...")
    mass_vals, numixing_vals, heatmap_data = prepare_heatmap_data(df, use_weighted=False)
    
    # Prepare weighted heatmap data
    print("\nPreparing weighted heat map data...")
    _, _, heatmap_weighted = prepare_heatmap_data(df, use_weighted=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate multiple views
    print("\nGenerating plots...")
    plot_multiple_views(mass_vals, numixing_vals, heatmap_data, heatmap_weighted, args.output_dir, df)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Mass range: {mass_vals.min():.3f} - {mass_vals.max():.3f} GeV")
    print(f"numixing range: {numixing_vals.min():.2e} - {numixing_vals.max():.2e}")
    print(f"\nUnweighted surviving LLPs:")
    print(f"  Total: {np.nansum(heatmap_data):.0f}")
    print(f"  Max: {np.nanmax(heatmap_data):.0f}")
    print(f"  Mean (non-zero): {np.nanmean(heatmap_data[heatmap_data > 0]):.2f}")
    print(f"\nWeighted surviving LLPs:")
    print(f"  Total: {np.nansum(heatmap_weighted):.6e}")
    print(f"  Max: {np.nanmax(heatmap_weighted):.6e}")
    if np.any(heatmap_weighted > 0):
        print(f"  Mean (non-zero): {np.nanmean(heatmap_weighted[heatmap_weighted > 0]):.6e}")
    
    # Find optimal points (highest survival)
    if np.any(heatmap_weighted > 0):
        idx = np.unravel_index(np.nanargmax(heatmap_weighted), heatmap_weighted.shape)
        best_numixing = numixing_vals[idx[0]]
        best_mass = mass_vals[idx[1]]
        best_count = heatmap_weighted[idx[0], idx[1]]
        print(f"\nBest point (weighted):")
        print(f"  Mass = {best_mass:.3f} GeV, numixing = {best_numixing:.2e}")
        print(f"  Surviving LLPs (weighted) = {best_count:.6e}")
    
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
