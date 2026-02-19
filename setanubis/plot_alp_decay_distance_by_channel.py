#!/usr/bin/env python3
"""
Plot ALP 3D distance from IP distributions grouped by decay channel.

This script mirrors the behaviour of `plot_alp_decay_distance_from_ip.py`
but groups and compares results across different ALP decay channels.

It searches the provided `data_path` for runs with the requested `CaPhi`
value, detects the decay channel for each run (from `params.dat` or the
pickle DataFrame), and creates:
 - per-channel per-mass individual histograms
 - overlay comparing channels (aggregated over masses)
 - statistics vs mass with separate lines per channel

Output files are saved to `output_dir`.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DEFAULTS
DEFAULT_DATA_PATH = '/usera/fs568/set-anubis/ALP_Z_Discontinuity_Check_Runs'
DEFAULT_OUTPUT_DIR = '/usera/fs568/set-anubis/setanubis'
DEFAULT_CAPHI = 0.001

# Reference distances
ATLAS_TRACKING_RADIUS = 9.5
CAVERN_DETECTION_RADIUS = 20.0


def parse_params_file(params_path):
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
                            params[key] = parts[1].strip()
    except Exception:
        pass
    return params


def find_available_data_by_caphi_and_channel(data_path, caphi_target):
    """
    Walk the `data_path` looking for run directories containing `params.dat`.
    For matching `caphi` values return list of (scan, run, pkl_file, mass, caphi, params_path).
    """
    data_info = []

    # Include both standard scan pattern and discontinuity-check pattern
    scan_dirs = sorted(glob.glob(os.path.join(data_path, 'ALP_axZ_scan_*')))
    if not scan_dirs:
        scan_dirs = sorted(glob.glob(os.path.join(data_path, 'ALP_axZ_discontinuity_check_scan_*')))

    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        match = re.search(r'scan_(\d+)', scan_name)
        scan_num = int(match.group(1)) if match else None

        events_dir = os.path.join(scan_dir, 'Events')
        if not os.path.exists(events_dir):
            continue

        run_dirs = sorted([d for d in glob.glob(os.path.join(events_dir, 'run_*')) if os.path.isdir(d)])

        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            match_run = re.search(r'run_(\d+)', run_name)
            run_num = int(match_run.group(1)) if match_run else None

            params_path = os.path.join(run_dir, 'params.dat')
            if not os.path.exists(params_path):
                continue

            params = parse_params_file(params_path)
            caphi = params.get('caphi', None)
            ma = params.get('ma', None)

            if caphi is None or ma is None:
                # try to read pkl to obtain mass if params missing
                pass

            try:
                caphi_val = float(caphi) if caphi is not None else None
            except Exception:
                caphi_val = None

            if caphi_val is None:
                continue

            if abs(caphi_val - caphi_target) < 1e-6:
                # search for matching pickle under data_path or directly in run_dir
                patterns = [
                    os.path.join(data_path, '**', f'ALP_Z_df_Scan_{scan_num}_Run_{run_num}.pkl'),
                    os.path.join(data_path, '**', f'ALP_Z_discontinuity_check_df_Scan_{scan_num}_Run_{run_num}.pkl'),
                    os.path.join(data_path, '**', f'ALP_Z_discontinuity_check_sampledfs_Scan_{scan_num}_Run_{run_num}.pkl.gz'),
                    os.path.join(run_dir, '*.pkl'),
                    os.path.join(data_path, f'ALP_Z_discontinuity_check_df_Scan_{scan_num}_Run_{run_num}.pkl'),
                ]
                pkl_candidates = []
                for pat in patterns:
                    found = glob.glob(pat, recursive=True)
                    if found:
                        pkl_candidates.extend(found)

                if not pkl_candidates:
                    continue

                pkl_file = pkl_candidates[0]
                data_info.append((scan_num, run_num, pkl_file, float(ma) if ma is not None else None, caphi_val, params_path))
                print(f"  Found Scan {scan_num}, Run {run_num}: mass = {ma}, CaPhi = {caphi_val}, pkl={os.path.basename(pkl_file)}")

    data_info.sort(key=lambda x: (x[4] if x[4] is not None else 0.0, x[3] if x[3] is not None else 0.0))
    return data_info


def detect_decay_channel_from_df(df, params):
    # Try common keys in params first
    for k in ('decaychannel', 'decay_channel', 'decaymode', 'decay_mode', 'decay'):
        if k in params:
            return str(params[k])

    # Try DataFrame columns
    col_candidates = ['decayChannel', 'decay_channel', 'decayMode', 'decay_mode', 'decay']
    for c in col_candidates:
        if c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) > 0:
                return str(vals[0])

    # Fallback to unknown
    return 'unknown'


def parse_madspin_decay(scan_dir):
    """Parse Cards/madspin_card.dat under scan_dir to find ax decay line."""
    try:
        ms_path = os.path.join(scan_dir, 'Cards', 'madspin_card.dat')
        if not os.path.exists(ms_path):
            return None
        with open(ms_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith('decay') and 'ax' in line:
                    # e.g. "decay ax > e- e+"
                    parts = line.split('>')
                    if len(parts) == 2:
                        rhs = parts[1].strip()
                        return rhs.replace(' ', '')
    except Exception:
        return None
    return None


def load_alp_decay_distances_and_channel(filepath, params_path=None, distance_cut=None):
    df = pd.read_pickle(filepath)
    alp_data = df[df['PID'] == 9000005]

    # detect channel
    params = parse_params_file(params_path) if params_path else {}
    channel = detect_decay_channel_from_df(alp_data, params)
    # If not found in params or df, try to parse madspin card at scan level
    if (channel is None or channel == 'unknown') and params_path:
        scan_dir = os.path.abspath(os.path.join(params_path, '..', '..', '..'))
        ms_ch = parse_madspin_decay(scan_dir)
        if ms_ch:
            channel = ms_ch

    dist_list = []
    in_detection_count = 0

    for _, row in alp_data.iterrows():
        if row.get('decayVertexDist', 0) > 1e-10:
            decay_vtx = row['decayVertex']
            x_ip, y_ip, z_ip = decay_vtx[0] / 1000.0, decay_vtx[1] / 1000.0, decay_vtx[2] / 1000.0
            distance_from_ip = np.sqrt(x_ip**2 + y_ip**2 + z_ip**2)
            dist_list.append(distance_from_ip)
            if ATLAS_TRACKING_RADIUS < distance_from_ip < CAVERN_DETECTION_RADIUS:
                in_detection_count += 1

    dist_array = np.array(dist_list)
    if distance_cut is not None:
        dist_array = dist_array[dist_array <= distance_cut]

    mass = None
    if 'mass' in alp_data.columns and len(alp_data) > 0:
        try:
            mass = float(alp_data['mass'].iloc[0])
        except Exception:
            mass = None

    return {
        'distances': dist_array,
        'mass': mass,
        'n_total': len(alp_data),
        'n_decayed': len(dist_array),
        'n_in_detection_region': in_detection_count,
        'decay_fraction': len(dist_array) / len(alp_data) if len(alp_data) > 0 else 0,
        'channel': channel
    }


def sanitize_label(s):
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', str(s))


def plot_per_channel_individuals(channel_to_data, output_dir, caphi, use_log=True):
    for channel, data_list in channel_to_data.items():
        if not data_list:
            continue
        n_plots = len(data_list)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        axes = axes.flatten()

        for idx, data in enumerate(data_list):
            ax = axes[idx]
            distances = data['distances']
            mass = data['mass'] if data['mass'] is not None else 0.0

            if len(distances) > 0:
                if use_log:
                    bins = np.logspace(np.log10(max(distances.min(), 1e-3)), np.log10(distances.max()), 50)
                else:
                    bins = 50
                ax.hist(distances, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
                median_dist = np.median(distances)
                mean_dist = np.mean(distances)
                ax.axvline(median_dist, color='red', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2e} m')
                ax.axvline(mean_dist, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2e} m')
                ax.axvline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axvline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axvspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green')
                if use_log:
                    ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Distance from IP (m)')
                ax.set_ylabel('Number of ALPs')
                ax.set_title(f'$m_a$ = {mass:.3f} GeV\\n$N_{{decay}}$ = {data["n_decayed"]}, In Region: {data.get("n_in_detection_region",0)}')
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No decayed ALPs', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'$m_a$ = {mass:.3f} GeV')

        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
        plt.suptitle(fr'ALP Distance from IP ($C_{{a\phi}}$ = {caphi_str}) - Channel: {channel}', fontsize=14)
        plt.tight_layout()
        out_name = f'alp_distance_ip_individual_channel_{sanitize_label(channel)}.pdf'
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-mass individuals for channel {channel} to: {out_path}")
        plt.close()


def plot_overlay_channels(channel_to_data, output_dir, caphi, use_log=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    channels = sorted(channel_to_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(channels))))

    # Determine global max for bins
    max_d = 0.0
    for ch, data_list in channel_to_data.items():
        for d in data_list:
            if len(d['distances']) > 0:
                max_d = max(max_d, d['distances'].max())

    if max_d <= 0:
        print('No decays to plot in overlay across channels.')
        return

    bins = np.logspace(-3, np.log10(max_d), 80) if use_log else 80

    for idx, ch in enumerate(channels):
        # aggregate distances across masses for this channel
        all_dist = np.concatenate([d['distances'] for d in channel_to_data[ch] if len(d['distances']) > 0]) if any(len(d['distances'])>0 for d in channel_to_data[ch]) else np.array([])
        if len(all_dist) == 0:
            continue
        ax.hist(all_dist, bins=bins, alpha=0.6, edgecolor='black', linewidth=0.5, label=f'{ch}', color=colors[idx % len(colors)])

    ax.axvline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label=f'ATLAS ~{ATLAS_TRACKING_RADIUS} m')
    ax.axvline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', linewidth=2.5, alpha=0.8, label=f'Cavern ~{CAVERN_DETECTION_RADIUS} m')
    ax.axvspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green')

    if use_log:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance from IP (m)')
    ax.set_ylabel('Number of ALPs')
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    ax.set_title(fr'ALP Distance from IP - Channels (CaPhi = {caphi_str})')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    out_path = os.path.join(output_dir, 'alp_distance_ip_overlay_channels.pdf')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved channel overlay to: {out_path}")
    plt.close()


def plot_stats_vs_mass_channel(channel_to_data, output_dir, caphi):
    fig, ax = plt.subplots(figsize=(12, 8))
    channels = sorted(channel_to_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(channels))))

    for idx, ch in enumerate(channels):
        masses = []
        medians = []
        means = []
        p25 = []
        p75 = []
        for d in channel_to_data[ch]:
            if len(d['distances']) > 0 and d['mass'] is not None:
                masses.append(d['mass'])
                medians.append(np.median(d['distances']))
                means.append(np.mean(d['distances']))
                p25.append(np.percentile(d['distances'], 25))
                p75.append(np.percentile(d['distances'], 75))

        if not masses:
            continue

        masses = np.array(masses)
        sort_idx = np.argsort(masses)
        masses = masses[sort_idx]
        medians = np.array(medians)[sort_idx]
        means = np.array(means)[sort_idx]
        p25 = np.array(p25)[sort_idx]
        p75 = np.array(p75)[sort_idx]

        ax.plot(masses, medians, 'o-', color=colors[idx % len(colors)], label=f'{ch} median')
        ax.plot(masses, means, 's--', color=colors[idx % len(colors)], alpha=0.7, label=f'{ch} mean')
        ax.fill_between(masses, p25, p75, alpha=0.15, color=colors[idx % len(colors)])

    ax.axhline(ATLAS_TRACKING_RADIUS, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'ATLAS ~{ATLAS_TRACKING_RADIUS} m')
    ax.axhline(CAVERN_DETECTION_RADIUS, color='brown', linestyle='--', linewidth=2, alpha=0.8, label=f'Cavern ~{CAVERN_DETECTION_RADIUS} m')
    ax.axhspan(ATLAS_TRACKING_RADIUS, CAVERN_DETECTION_RADIUS, alpha=0.15, color='green')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass $m_a$ (GeV)')
    ax.set_ylabel('Distance from IP (m)')
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    ax.set_title(fr'ALP Distance from IP Statistics vs Mass (CaPhi = {caphi_str})')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    out_path = os.path.join(output_dir, 'alp_distance_ip_vs_mass_channels.pdf')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistics vs mass per channel to: {out_path}")
    plt.close()


def print_summary_table(channel_to_data, caphi):
    print('\n' + '='*120)
    print('ALP DISTANCE FROM IP SUMMARY BY CHANNEL')
    caphi_str = f"{caphi:.6f}" if caphi < 0.01 else f"{caphi:.3f}"
    print(f'CaPhi = {caphi_str}')
    print('='*120)
    print(f"{'Channel':<20} {'Mass (GeV)':<12} {'N_total':<10} {'N_decay':<10} {'In Region':<12} {'Median (m)':<15} {'Mean (m)':<15}")
    print('-'*120)
    for ch, lst in channel_to_data.items():
        for d in lst:
            mass = d['mass'] if d['mass'] is not None else 0.0
            n_total = d['n_total']
            n_decay = d['n_decayed']
            in_region = d.get('n_in_detection_region', 0)
            if len(d['distances']) > 0:
                median = np.median(d['distances'])
                mean = np.mean(d['distances'])
                print(f"{ch:<20} {mass:<12.4f} {n_total:<10} {n_decay:<10} {in_region:<12} {median:<15.3e} {mean:<15.3e}")
            else:
                print(f"{ch:<20} {mass:<12.4f} {n_total:<10} {n_decay:<10} {in_region:<12} {'N/A':<15} {'N/A':<15}")
    print('='*120)


def main():
    parser = argparse.ArgumentParser(description='Plot ALP distance from IP grouped by decay channel')
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--caphi', type=float, default=DEFAULT_CAPHI)
    parser.add_argument('--max-distance', type=float, default=None)
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()

    print(f"Searching {args.data_path} for CaPhi = {args.caphi}")
    data_info = find_available_data_by_caphi_and_channel(args.data_path, args.caphi)
    if not data_info:
        print(f"No data found for CaPhi = {args.caphi} in {args.data_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    channel_to_data = {}

    print('\nLoading data...')
    for scan_num, run_num, pkl_file, mass, caphi_val, params_path in data_info:
        print(f'  Loading Scan {scan_num} Run {run_num} -> {os.path.basename(pkl_file)}')
        d = load_alp_decay_distances_and_channel(pkl_file, params_path=params_path, distance_cut=args.max_distance)
        # fall back to mass from params if df didn't have it
        if (d['mass'] is None or d['mass'] == 0.0) and mass is not None:
            d['mass'] = mass

        ch = d.get('channel', 'unknown')
        channel_to_data.setdefault(ch, []).append(d)
        print(f"    Channel: {ch}, Decayed: {d['n_decayed']}, In Region: {d.get('n_in_detection_region',0)}")

    print_summary_table(channel_to_data, args.caphi)

    use_log = not args.linear
    plot_per_channel_individuals(channel_to_data, args.output_dir, args.caphi, use_log=use_log)
    plot_overlay_channels(channel_to_data, args.output_dir, args.caphi, use_log=use_log)
    plot_stats_vs_mass_channel(channel_to_data, args.output_dir, args.caphi)

    print('\nAll plots saved to: ' + args.output_dir)


if __name__ == '__main__':
    main()
