#!/usr/bin/env python3
"""Plot fraction of LLPs surviving each cut stage for a given mass and CaPhi.

Saves a PNG into a `Plots/` folder next to this script.
"""
import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_WEIGHTED_COLS = [
    'nLLP_original_weighted',
    'nLLP_InCavern_weighted',
    'nLLP_NotInATLAS_weighted',
    'nLLP_Geometry_weighted',
    'nLLP_Tracker_weighted',
    'nLLP_MET_weighted',
    'nLLP_IsoJet_weighted',
    'nLLP_IsoCharged_weighted',
    'nLLP_IsoAll_weighted',
    'nLLP_Final_weighted',
]

STAGE_LABELS = [
    'Original', 'Within Cavern', 'ATLAS Veto', 'ANUBIS\nIntersections', 'Track Requirement\n+ Momenta', r'$E_\text{T}^\text{miss}$', 'LLP-Jet Isolation', 'LLP-Charged\nParticle Isolation', 'Combined\nTopology Veto', 'Final'
]


def plot_fraction_surviving(mass: float = 0.0562,
                            caphi: float = 0.1,
                            csv_path: Optional[str] = None,
                            outdir: str = 'Plots',
                            use_weighted: bool = True,
                            outfile: Optional[str] = None,
                            outfmt: str = 'pdf') -> Optional[str]:
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'selection_cutflow_ee_decay_channel.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)

    # Use isclose to avoid float-equality issues
    mask = np.isclose(df['mass'].astype(float), float(mass)) & np.isclose(df['CaPhi'].astype(float), float(caphi))
    df_target = df[mask]
    if df_target.empty:
        print(f'No rows found for mass={mass}, CaPhi={caphi}')
        return None

    cols = DEFAULT_WEIGHTED_COLS if use_weighted else [c.replace('_weighted', '') for c in DEFAULT_WEIGHTED_COLS]
    sums = df_target[cols].sum()
    denom = float(sums[cols[0]])
    if denom == 0:
        print('Original count (denominator) is zero; cannot compute fractions')
        return None

    fractions = (sums / denom).values

    outdir_abs = os.path.join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir_abs, exist_ok=True)
    if outfile is None:
        safe_mass = str(mass).replace('.', 'p')
        safe_caphi = str(caphi).replace('.', 'p')
        outfile = f'fraction_CaPhi{safe_caphi}_mass{safe_mass}.{outfmt}'
    outpath = os.path.join(outdir_abs, outfile)

    # Draw a step-like trace of surviving fractions but only show
    # the vertical parts where a bar's right edge rises above the
    # next bar. We draw horizontal tops and then partial vertical
    # segments at the boundaries to avoid double/thick overlapping
    # edges while keeping the visual step appearance.
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(STAGE_LABELS))
    # Use bar width 1.0 and center alignment so adjacent bars touch (step-like appearance)
    bar_width = 1.0

    # Prepare a small positive floor for plotting on a log scale so
    # zero-values don't break the axis. Use the same logic later for
    # axis limits.
    pos = fractions[fractions > 0]
    if pos.size > 0:
        ymin = float(pos.min()) / 10.0
    else:
        ymin = 1e-12

    # Clip plotting heights to the positive floor so lines render on
    # a log scale even when some stages have zero surviving events.
    y_plot = np.maximum(fractions, ymin)

    # Draw horizontal tops for each bar using Line2D so we can set
    # capstyle and antialiasing to avoid tiny gaps when joining with
    # vertical segments. We add a tiny x-overlap and extend vertical
    # endpoints slightly to ensure seamless joins on a log scale.
    line_kwargs_h = dict(color='C0', linewidth=2.5, zorder=3, solid_capstyle='projecting', antialiased=False)
    line_kwargs_v = dict(color='C0', linewidth=2.5, zorder=4, solid_capstyle='projecting', antialiased=False)
    x_eps = 1e-6
    y_rel_eps = 1e-9

    for xi, yi in zip(x, y_plot):
        ax.plot([xi - bar_width / 2 - x_eps, xi + bar_width / 2 + x_eps], [yi, yi], **line_kwargs_h)

    # Draw partial vertical segments at each boundary x = j + 0.5.
    # For the right edge of bar j show only the portion above bar j+1.
    if len(x) > 0:
        # leftmost full edge
        y0_low = ymin * (1.0 - y_rel_eps)
        y0_high = y_plot[0] * (1.0 + y_rel_eps)
        ax.plot([x[0] - bar_width / 2, x[0] - bar_width / 2], [y0_low, y0_high], **line_kwargs_v)

    for j in range(len(x) - 1):
        x_vert = j + 0.5
        h_left = float(fractions[j])
        h_right = float(fractions[j + 1])
        if h_left > h_right:
            y1 = max(h_right, ymin) * (1.0 - y_rel_eps)
            y2 = max(h_left, ymin) * (1.0 + y_rel_eps)
            ax.plot([x_vert, x_vert], [y1, y2], **line_kwargs_v)

    # rightmost full edge
    if len(x) > 0:
        yR_low = ymin * (1.0 - y_rel_eps)
        yR_high = y_plot[-1] * (1.0 + y_rel_eps)
        ax.plot([x[-1] + bar_width / 2, x[-1] + bar_width / 2], [yR_low, yR_high], **line_kwargs_v)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('Fraction of Surviving Events', fontsize=16)
    # Configure log scale on y-axis and handle zero values
    pos = fractions[fractions > 0]
    if pos.size > 0:
        ymin = float(pos.min()) / 10.0
    else:
        ymin = 1e-12
    ymax = max(1.05, float(fractions.max()) * 1.4)
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)
    # Expand x-limits so the first/last bars are fully visible
    ax.set_xlim(-0.5, len(STAGE_LABELS) - 0.5)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, format=outfmt, bbox_inches='tight')

    print('Saved:', outpath)
    print('Fractions per stage:')
    for s, f in zip(STAGE_LABELS, fractions):
        print(f'{s}: {f}')

    return outpath


def main():
    parser = argparse.ArgumentParser(description='Plot fraction of LLPs surviving stages.')
    parser.add_argument('--mass', type=float, default=0.0562, help='mass value to filter (default 0.0562)')
    parser.add_argument('--caphi', type=float, default=0.1, help='CaPhi value to filter (default 0.1)')
    parser.add_argument('--csv', type=str, default=None, help='path to selection_cutflow CSV (default in same folder)')
    parser.add_argument('--outdir', type=str, default='Plots', help='output directory inside script folder')
    parser.add_argument('--no-weighted', dest='use_weighted', action='store_false', help='use unweighted counts instead of weighted')
    parser.add_argument('--outfile', type=str, default=None, help='output filename (optional)')
    parser.add_argument('--format', type=str, default='pdf', choices=['png', 'pdf', 'svg', 'eps'], help='output file format (default: pdf)')
    args = parser.parse_args()

    plot_fraction_surviving(mass=args.mass, caphi=args.caphi, csv_path=args.csv, outdir=args.outdir,
                            use_weighted=args.use_weighted, outfile=args.outfile)


if __name__ == '__main__':
    main()
