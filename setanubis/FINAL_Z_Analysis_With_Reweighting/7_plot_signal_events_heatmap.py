"""
Combined Production: Sum signal events from Higgs and pp -> Za and plot.
Uses the exact visual style (grey zeros, box/dot clusters) from the Higgs script.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from pathlib import Path

# --- EXACT DATA PREPARATION FROM YOUR SCRIPT ---

def prepare_heatmap_data_from_grid(grid_df: pd.DataFrame, value_column='N_signal'):
    df = grid_df.copy()
    mass_vals = np.sort(df['mass'].unique())
    caphi_vals = np.sort(df['CaPhi'].unique())
    heat = np.full((len(caphi_vals), len(mass_vals)), np.nan)
    for _, row in df.iterrows():
        i = int(np.searchsorted(caphi_vals, row['CaPhi']))
        j = int(np.searchsorted(mass_vals, row['mass']))
        heat[i, j] = float(row[value_column]) if not np.isnan(row[value_column]) else np.nan
    return mass_vals, caphi_vals, heat

# --- EXACT PLOTTING LOGIC FROM YOUR HIGGS SCRIPT ---

def plot_heatmap(mass_vals, caphi_vals, heatmap_data, output_path,
                 title="Expected Signal Events", use_log_scale=True,
                 colorbar_label="N_signal", vmin_override=None, vmax_override=None,
                 show_percentage=None, sigfig=3):
    
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

    # --- CLUSTER LOGIC: DOTS vs BOXES ---
    idx_large = []
    idx_small = []
    try:
        log_mass_arr = np.log10(mass_arr)
        for c in np.unique(caphi_arr):
            mask = (caphi_arr == c)
            masses = mass_arr[mask]
            log_masses = log_mass_arr[mask]
            idxs = np.where(mask)[0]
            if len(masses) == 1:
                idx_large.append(idxs[0])
            else:
                sorted_idx = np.argsort(masses)
                sorted_log_masses = log_masses[sorted_idx]
                sorted_idxs = idxs[sorted_idx]
                cluster = [sorted_idxs[0]]
                for ii in range(1, len(sorted_log_masses)):
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

    # --- DRAWING LARGE BOXES (with grey zero handling) ---
    if len(idx_large) > 0:
        is_zero = (vals_arr[idx_large] == 0)
        if np.any(is_zero):
            ax.scatter(mass_arr[idx_large][is_zero], caphi_arr[idx_large][is_zero],
                       marker='s', s=800, facecolors='lightgrey', edgecolors='black', linewidths=0.5, zorder=2)
        if np.any(~is_zero):
            ax.scatter(mass_arr[idx_large][~is_zero], caphi_arr[idx_large][~is_zero],
                       c=vals_arr[idx_large][~is_zero], cmap=plt.cm.viridis, norm=norm,
                       marker='s', s=800, edgecolors='black', linewidths=0.5, alpha=1.0, zorder=2)

    # --- DRAWING SMALL DOTS (with grey zero handling) ---
    if len(idx_small) > 0:
        is_zero_small = (vals_arr[idx_small] == 0)
        if np.any(is_zero_small):
            ax.scatter(mass_arr[idx_small][is_zero_small], caphi_arr[idx_small][is_zero_small],
                       marker='o', s=30, c='lightgrey', edgecolors='none', alpha=0.8, zorder=3)
        if np.any(~is_zero_small):
            ax.scatter(mass_arr[idx_small][~is_zero_small], caphi_arr[idx_small][~is_zero_small],
                       c=vals_arr[idx_small][~is_zero_small], cmap=plt.cm.viridis, norm=norm,
                       marker='o', s=30, edgecolors='none', alpha=0.8, zorder=3)

    # --- COLORBAR (Invisible scatter trick) ---
    nonzero_all_mask = (vals_arr != 0)
    if np.any(nonzero_all_mask):
        sc = ax.scatter(mass_arr[nonzero_all_mask], caphi_arr[nonzero_all_mask],
                        c=vals_arr[nonzero_all_mask], cmap=plt.cm.viridis, norm=norm, marker='o', s=0)
    else:
        sc = ax.scatter([], [], c=[], cmap=plt.cm.viridis, norm=norm)

    # --- ANNOTATIONS ---
    if len(arr) > 0 and len(idx_large) > 0:
        overall_vmax = np.nanmax(vals_arr) if vals_arr.size > 0 else 0
        show_as_percentage = (overall_vmax <= 1.0) if show_percentage is None else bool(show_percentage)
        for idx in idx_large:
            mval, cval, value = mass_arr[idx], caphi_arr[idx], vals_arr[idx]
            if np.isnan(value): continue
            text = (f'{value*100:.2f}%' if value > 0 else '0%') if show_as_percentage else f'{value:.{sigfig}g}'
            ax.text(mval, cval, text, ha='center', va='center', color='white', fontsize=8, fontweight='bold',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    cbar = plt.colorbar(sc, ax=ax, label=colorbar_label, pad=0.02)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('ALP Mass [GeV]', fontsize=14); ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=14)
    ax.set_title(title, fontsize=16, pad=16)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- MERGE AND SUM DATA ---

def main():
    # Paths based on your setup
    current_dir = "/usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting"
    higgs_csv = os.path.join(current_dir, "Higgs_production/Plots/higgs_signal_events_data.csv")
    pp_csv = os.path.join(current_dir, "pp_production/Plots/ppZ_signal_events_data.csv")
    output_dir = os.path.join(current_dir, "Plots")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading CSVs...")
    df_h = pd.read_csv(higgs_csv)
    df_p = pd.read_csv(pp_csv)

    # Merge on Mass and Coupling
    df_combined = pd.merge(
        df_h[['mass', 'CaPhi', 'N_signal']], 
        df_p[['mass', 'CaPhi', 'N_signal']], 
        on=['mass', 'CaPhi'], 
        how='outer', 
        suffixes=('_h', '_p')
    ).fillna(0)

    # Sum total signal
    df_combined['N_signal'] = df_combined['N_signal_h'] + df_combined['N_signal_p']

    # Save combined CSV
    out_csv = os.path.join(output_dir, "combined_signal_events_data.csv")
    df_combined.to_csv(out_csv, index=False)
    print(f"Saved summed data to {out_csv}")

    # Plot using exact Higgs script style
    mass_vals, caphi_vals, heat_sig = prepare_heatmap_data_from_grid(df_combined, 'N_signal')
    
    plot_heatmap(
        mass_vals, caphi_vals, heat_sig, 
        os.path.join(output_dir, 'total_signal_events_heatmap.png'),
        title='Total Expected Signal Events (Higgs + pp Production)',
        colorbar_label='Expected Signal Events',
        show_percentage=False,
        sigfig=3
    )

    print(f"Success. Total Events Sum: {df_combined['N_signal'].sum():.3f}")
    print(f"Plots saved to {output_dir}/")

if __name__ == '__main__':
    main()