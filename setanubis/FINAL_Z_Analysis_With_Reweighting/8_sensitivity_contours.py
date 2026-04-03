"""
Plot sensitivity contours from an extended Higgs signal-events CSV.

Reads `Plots/higgs_signal_events_data.csv` (produced by
`5_plot_signal_events_heatmap.py`) and draws a heatmap of expected signal
events with contour lines at specified event counts. One contour is
highlighted at 4 events by default.

Usage:
    python 6_sensitivity_contours.py --csv /path/to/higgs_signal_events_data.csv

"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
try:
    # Use LinearNDInterpolator to prevent cubic "overshoot" loops and artifacts
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull
except Exception:
    LinearNDInterpolator = None
    gaussian_filter = None
    ConvexHull = None


def prepare_grid_from_csv(csv_path: str, value_column: str = 'N_signal'):
    """Load a CSV and return grid arrays for heatmap plotting.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing columns ``mass`` and ``CaPhi`` and
        the value column to place on the grid.
    value_column : str, optional
        Name of the column to use for grid values (default ``'N_signal'``).

    Returns
    -------
    mass_vals : numpy.ndarray
        Sorted unique mass values (1D).
    caphi_vals : numpy.ndarray
        Sorted unique CaPhi values (1D).
    heat : numpy.ndarray
        2D array with shape ``(len(caphi_vals), len(mass_vals))`` containing
        the requested values or ``NaN`` for missing points. Indexed as
        ``heat[caphi_index, mass_index]``.

    Notes
    -----
    The function expects exact matches in the CSV for mass and CaPhi grid
    points; rows with unknown coordinates are ignored.
    """
    df = pd.read_csv(csv_path)
    if 'mass' not in df.columns or 'CaPhi' not in df.columns:
        raise RuntimeError(f"CSV {csv_path} must contain 'mass' and 'CaPhi' columns")

    mass_vals = np.sort(df['mass'].unique())
    caphi_vals = np.sort(df['CaPhi'].unique())

    heat = np.full((len(caphi_vals), len(mass_vals)), np.nan)
    # build index maps for robust filling
    mass_to_idx = {m: i for i, m in enumerate(mass_vals)}
    caphi_to_idx = {c: i for i, c in enumerate(caphi_vals)}

    for _, row in df.iterrows():
        m = float(row['mass'])
        c = float(row['CaPhi'])
        val = float(row.get(value_column, np.nan))
        i = caphi_to_idx.get(c, None)
        j = mass_to_idx.get(m, None)
        if i is None or j is None:
            continue
        heat[i, j] = val

    return mass_vals, caphi_vals, heat


def plot_sensitivity_contours(mass_vals, caphi_vals, heat, levels, output_path,
                              title=None, cmap_name='viridis', use_log_scale=True, dpi=300,
                              smooth_sigma=0.0):
    """Plot sensitivity contours from a grid of expected signal events.

    Parameters
    ----------
    mass_vals : array-like
        1D sorted array of mass grid points.
    caphi_vals : array-like
        1D sorted array of coupling grid points.
    heat : array-like
        2D array with shape ``(len(caphi_vals), len(mass_vals))`` containing
        event counts or ``NaN`` for missing points.
    levels : sequence of float
        Contour levels (event counts) to draw.
    output_path : str or pathlib.Path
        Path where the plot PNG (and PDF if possible) will be saved.
    title : str, optional
        Plot title.
    cmap_name : str, optional
        Name of the Matplotlib colormap to use (default ``'viridis'``).
    use_log_scale : bool, optional
        Use logarithmic color normalization for the colormap.
    dpi : int, optional
        Output resolution in DPI.
    smooth_sigma : float, optional
        Gaussian smoothing sigma applied to the interpolated log-space grid
        (set to 0 to disable smoothing).

    Notes
    -----
    The function saves figures to ``output_path`` and does not return a value.
    """

    X, Y = np.meshgrid(mass_vals, caphi_vals)
    Z = np.array(heat, dtype=float)

    # mask invalid / negative values for plotting
    Z_masked = np.ma.masked_invalid(Z)

    # find positive range for log normalization
    positive = Z_masked.compressed()[Z_masked.compressed() > 0] if Z_masked.count() > 0 else np.array([])
    if positive.size > 0:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
    else:
        vmin = np.finfo(float).tiny
        vmax = 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    # color for masked cells (e.g., kinematically forbidden or true NaNs)
    try:
        cmap.set_bad('lightgrey')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 8))

    if use_log_scale and vmax > 0:
        vmin_safe = max(vmin, np.finfo(float).tiny)
        norm = colors.LogNorm(vmin=vmin_safe, vmax=vmax)
    else:
        norm = None

    # Recreate discrete grid-style plotting: empty circles for all grid
    # points, filled colored circles for positive N_signal values.
    points = []
    for i, cval in enumerate(caphi_vals):
        for j, mval in enumerate(mass_vals):
            v = Z[i, j]
            points.append((mval, cval, float(v) if not np.isnan(v) else np.nan))

    arr = np.array(points, dtype=object)
    mass_arr = arr[:, 0].astype(float)
    caphi_arr = arr[:, 1].astype(float)
    vals_arr = arr[:, 2].astype(float)

    # empty circles for the full grid layout
    ax.scatter(mass_arr, caphi_arr, facecolors='none', edgecolors='lightgrey', s=120, linewidths=0.8, zorder=2)

    # filled circles for positive actual MC entries
    pos_mask = ~np.isnan(vals_arr) & (vals_arr > 0)
    if np.any(pos_mask):
        # avoid strong black edges which can obscure contour lines
        sc = ax.scatter(mass_arr[pos_mask], caphi_arr[pos_mask], c=vals_arr[pos_mask], cmap=cmap, norm=norm,
                        marker='o', s=140, edgecolors='none', alpha=0.95, zorder=3)
    else:
        sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

    # Use geometric bin edges (same as pcolormesh/heatmap) for exact
    # Matplotlib-style padding.
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

    # Use SciPy's LinearNDInterpolator on scattered log-space data
    if LinearNDInterpolator is None:
        raise RuntimeError('scipy.interpolate.LinearNDInterpolator is required; please install scipy.')

    # Require complete rectangular grid (no NaNs)
    if np.isnan(Z).any():
        raise RuntimeError('Data grid contains NaNs; fill missing values before using structured interpolation.')

    x = np.asarray(mass_vals, dtype=float)
    y = np.asarray(caphi_vals, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise RuntimeError('Mass and CaPhi must be positive to perform log-space interpolation.')

    # Prepare log-space arrays
    xlog = np.log10(x)
    ylog = np.log10(y)

    Z2 = np.array(Z, dtype=float)  # shape (ny, nx) where ny=len(caphi), nx=len(mass)
    positive_mask = Z2 > 0
    
    if np.any(positive_mask):
        min_pos = float(np.min(Z2[positive_mask]))
        max_pos = float(np.max(Z2[positive_mask]))
        # Dynamically scale eps safely below the minimum data point for zero bins
        # This gives a "cliff" for the linear interpolator to slope down towards
        eps = min_pos * 1e-3
    else:
        # Failsafe for entirely empty plots
        eps = 1e-15
        max_pos = eps
        
    Z_safe = np.where(Z2 > 0, Z2, eps)
    zlog = np.log10(Z_safe)

    # Flatten the grid into scattered points for smooth SciPy interpolation
    XX, YY = np.meshgrid(xlog, ylog)
    pts = np.column_stack((XX.ravel(), YY.ravel()))  # (xlog, ylog) pairs
    vals = zlog.ravel()

    # Keep only finite entries
    finite_mask_pts = np.isfinite(vals)
    pts_f = pts[finite_mask_pts]
    vals_f = vals[finite_mask_pts]

    if pts_f.shape[0] < 3:
        raise RuntimeError('Not enough points for interpolation (need >=3).')

    # Using Linear instead of Cubic to strictly prevent artificial loops/islands
    interp = LinearNDInterpolator(pts_f, vals_f)

    # Evaluation grid in log-space
    nx_grid, ny_grid = 2000, 2000
    LOGX = np.linspace(xlog.min(), xlog.max(), nx_grid)
    LOGY = np.linspace(ylog.min(), ylog.max(), ny_grid)
    LOGGX, LOGGY = np.meshgrid(LOGX, LOGY)
    eval_pts = np.column_stack((LOGGX.ravel(), LOGGY.ravel()))
    GZ_log = interp(eval_pts).reshape(LOGGX.shape)

    # Mask points outside convex hull to avoid extrapolation artifacts
    try:
        if ConvexHull is not None and pts_f.shape[0] >= 3:
            hull = ConvexHull(pts_f)
            hull_path = MplPath(pts_f[hull.vertices])
            inside = hull_path.contains_points(eval_pts)
            GZ_log_flat = GZ_log.ravel()
            GZ_log_flat[~inside] = np.nan
            GZ_log = GZ_log_flat.reshape(GZ_log.shape)
    except Exception:
        # If hull computation fails, continue without masking
        pass

    # Optional Gaussian smoothing in log-space (highly recommended for linear interp to round corners)
    if gaussian_filter is not None and smooth_sigma is not None and smooth_sigma > 0:
        try:
            # Fill NaNs with median for smoothing, then re-mask
            med = np.nanmedian(GZ_log)
            if np.isfinite(med):
                filled = np.where(np.isfinite(GZ_log), GZ_log, med)
            else:
                filled = np.where(np.isfinite(GZ_log), GZ_log, 0.0)
            smoothed = gaussian_filter(filled, sigma=smooth_sigma, mode='nearest')
            smoothed[~np.isfinite(GZ_log)] = np.nan
            GZ_log = smoothed
        except Exception:
            pass

    GZ = np.where(np.isfinite(GZ_log), 10.0 ** GZ_log, np.nan)
    GX, GY = np.meshgrid(10.0 ** LOGX, 10.0 ** LOGY)

    # If smoothing was applied, clip to original data range to avoid artificial peaks
    if smooth_sigma is not None and smooth_sigma > 0 and np.any(positive_mask):
        try:
            GZ = np.clip(GZ, eps, max_pos)
        except Exception:
            pass

    # Sort requested levels (user inputs via terminal)
    levels = sorted(set(float(l) for l in levels))
    
    # DO NOT forcefully filter out the user's requested levels (e.g., 1e-8 or 4). 
    # Matplotlib will safely ignore levels that don't cross the data, but strict 
    # Python limits can drop contours due to tiny floating point precision errors.
    levels_to_plot = list(levels)

    # determine data min/max from interpolated grid if available,
    # otherwise from the raw masked values
    try:
        if GZ is not None:
            gvals = np.array(GZ).flatten()
            gvals = gvals[np.isfinite(gvals)]
        else:
            gvals = Z_masked.compressed() if Z_masked.count() > 0 else np.array([])
    except Exception:
        gvals = np.array([])

    if gvals.size > 0:
        data_min = float(np.nanmin(gvals))
        data_max = float(np.nanmax(gvals))
    else:
        data_min = data_max = 0.0

    if len(levels_to_plot) == 0:
        print('No valid contour levels to plot.')
    else:
        # Separate the 4.0 level from the rest for highlighting
        levels_non4 = [lv for lv in levels_to_plot if not np.isclose(lv, 4.0)]
        has_4 = any(np.isclose(lv, 4.0) for lv in levels_to_plot)
        
        cs_main = None
        cs4 = None
        try:
            if GZ is not None:
                if len(levels_non4) > 0:
                    cs_main = ax.contour(GX, GY, GZ, levels=levels_non4, colors='black', linewidths=1.6, zorder=60)
                if has_4:
                    # Shade the area >= 4.0 events
                    ax.contourf(GX, GY, GZ, levels=[4.0, np.inf], colors=['red'], alpha=0.15, zorder=50)                    
                    cs4 = ax.contour(GX, GY, GZ, levels=[4.0], colors='red', linewidths=3.0, zorder=70)
            else:
                if len(levels_non4) > 0:
                    cs_main = ax.contour(X, Y, Z_masked, levels=levels_non4, colors='black', linewidths=1.6, zorder=60)
                if has_4:
                    # Shade the area >= 4.0 events (fallback block)
                    ax.contourf(X, Y, Z_masked, levels=[4.0, np.inf], colors=['red'], alpha=0.15, zorder=50)
                    cs4 = ax.contour(X, Y, Z_masked, levels=[4.0], colors='red', linewidths=3.0, zorder=70)
        except Exception as e:
            print(f'Warning: failed to draw contours: {e}')

        # Add white strokes around the contours so they stand out against the colored dots
        def _apply_to_contour(cs_obj):
            if cs_obj is None:
                return
            
            # Handling differences between Matplotlib >= 3.8 and older versions
            if not hasattr(cs_obj, 'collections'):
                try:
                    cs_obj.set_path_effects([path_effects.withStroke(linewidth=cs_obj.get_linewidths()[0] + 3, foreground='white')])
                except Exception:
                    pass
            else:
                for coll in cs_obj.collections:
                    coll.set_path_effects([path_effects.withStroke(linewidth=coll.get_linewidth() + 3, foreground='white')])
            
            # REMOVED INLINE CLABEL:
            # try:
            #     ax.clabel(cs_obj, fmt='%g', inline=True, fontsize=8, colors='k')
            # except Exception:
            #     pass

        _apply_to_contour(cs_main)
        _apply_to_contour(cs4)

        # ---------------------------------------------------------
        # Build a dynamic legend instead of inline contour labels
        # ---------------------------------------------------------
        legend_handles = []
        
        # Add handles for standard contour levels
        if cs_main is not None and hasattr(cs_main, 'levels'):
            for lvl in cs_main.levels:
                legend_handles.append(Line2D([0], [0], color='black', lw=1.6, label=f'{lvl:g} events'))
                
        # Add handles for highlighted contour levels
        if cs4 is not None and hasattr(cs4, 'levels'):
            for lvl in cs4.levels:
                # Add the solid line for the contour border
                legend_handles.append(Line2D([0], [0], color='red', lw=3.0, label=f'{lvl:g} events (highlight)'))

        if legend_handles:
            # Draw the legend and place it in the bottom left
            legend = ax.legend(handles=legend_handles, loc='lower left', title='Contour Levels', 
                               framealpha=0.9, edgecolor='grey')
            legend.set_zorder(100) # Keep the legend above the scatter plot

    cbar = fig.colorbar(sc, ax=ax, label='Expected Signal Events', pad=0.02)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALP Mass [GeV]', fontsize=12)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=12)
    if title is None:
        title = 'Sensitivity Contours: Expected Signal Events'
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    out_path = Path(output_path)
    os.makedirs(out_path.parent or '.', exist_ok=True)
    # Save PNG (raster) and PDF (vector) for publication
    plt.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    try:
        pdf_path = out_path.with_suffix('.pdf')
        plt.savefig(str(pdf_path), bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def main():
    default_csv = os.path.join(os.path.dirname(__file__), 'Plots', 'combined_signal_events_data.csv')
    parser = argparse.ArgumentParser(description='Plot sensitivity contours from combined signal-events CSV')
    parser.add_argument('--csv', type=str, default=default_csv, help='Path to combined_signal_events_data.csv')
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(__file__), 'Plots', 'sensitivity_contours.png'), help='Output PNG path')
    parser.add_argument('--levels', type=str, default='1e-8,4', help='Comma-separated contour levels (event counts)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing sigma (log-grid units). Use 0 to disable smoothing, use 1 for final plot')
    parser.add_argument('--no-log', dest='use_log', action='store_false', help='Do not use log normalization/scales')
    parser.set_defaults(use_log=True)
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    mass_vals, caphi_vals, heat = prepare_grid_from_csv(csv_path, 'N_signal')
    levels = [float(x) for x in args.levels.split(',') if x.strip()]

    print(f'Loaded {len(mass_vals)} mass points × {len(caphi_vals)} coupling points from {csv_path}')
    print(f'Plotting contours at levels: {levels} -> saving to {args.output}')

    plot_sensitivity_contours(mass_vals, caphi_vals, heat, levels, args.output,
                              title=f'Expected Signal Events for Fermion-Coupled ALPs (Higgs and pp Production)',
                              use_log_scale=args.use_log, smooth_sigma=args.sigma)


if __name__ == '__main__':
    main()