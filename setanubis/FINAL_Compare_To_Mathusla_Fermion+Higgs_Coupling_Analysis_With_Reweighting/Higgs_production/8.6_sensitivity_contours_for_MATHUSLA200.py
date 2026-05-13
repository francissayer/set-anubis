"""
Plot sensitivity contours from an extended Higgs signal-events CSV for MATHUSLA200.

Reads `Plots/Simulated_MATHUSLA200_higgs_signal_events_data.csv` (produced by
`7.6_plot_signal_events_heatmap_for_MATHUSLA200.py`) and draws a heatmap of expected signal
events with contour lines at specified event counts. One contour is
highlighted at 4 events by default.

Usage:
    python 8.6_sensitivity_contours_for_MATHUSLA200.py --csv /path/to/Simulated_MATHUSLA200_higgs_signal_events_data.csv

"""
import os
import glob
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
        Path to a CSV file containing columns ``C_Zh`` and ``CaPhi`` and
        the value column to place on the grid.
    value_column : str, optional
        Name of the column to use for grid values (default ``'N_signal'``).

    Returns
    -------
    x_vals : numpy.ndarray
        Sorted unique C_Zh values (1D).
    caphi_vals : numpy.ndarray
        Sorted unique CaPhi values (1D).
    heat : numpy.ndarray
        2D array with shape ``(len(caphi_vals), len(x_vals))`` containing
        the requested values or ``NaN`` for missing points. Indexed as
        ``heat[caphi_index, x_index]``.

    Notes
    -----
    The function expects exact matches in the CSV for C_Zh and CaPhi grid
    points; rows with unknown coordinates are ignored.
    """
    df = pd.read_csv(csv_path)
    # [FIXED]: Strictly use C_Zh as the x-axis to match 5.py output[cite: 3]
    if 'C_Zh' not in df.columns or 'CaPhi' not in df.columns:
        raise RuntimeError(f"CSV {csv_path} must contain 'C_Zh' and 'CaPhi' columns")

    x_vals = np.sort(df['C_Zh'].unique())
    caphi_vals = np.sort(df['CaPhi'].unique())

    heat = np.full((len(caphi_vals), len(x_vals)), np.nan)
    # build index maps for robust filling
    x_to_idx = {m: i for i, m in enumerate(x_vals)}
    caphi_to_idx = {c: i for i, c in enumerate(caphi_vals)}

    for _, row in df.iterrows():
        m = float(row['C_Zh'])
        c = float(row['CaPhi'])
        val = float(row.get(value_column, np.nan))
        i = caphi_to_idx.get(c, None)
        j = x_to_idx.get(m, None)
        if i is None or j is None:
            continue
        heat[i, j] = val

    return x_vals, caphi_vals, heat


def _tab10_color_by_index(idx: int):
    """Return a reproducible color from the `tab10` colormap by integer index.

    Uses the ListedColormap `.colors` attribute when available to avoid
    calling the colormap with out-of-range floats which can produce
    inconsistent results.
    """
    cmap = plt.get_cmap('tab10')
    try:
        colors_list = cmap.colors
    except Exception:
        N = getattr(cmap, 'N', 10)
        colors_list = [cmap(i / float(max(1, N - 1))) for i in range(N)]
    return colors_list[int(idx) % len(colors_list)]


def _extract_br_value(fp, df=None):
    """Attempt to extract a single BR value from a DataFrame or filename.

    Preference order:
    1. Look for any column name containing 'br' (case-insensitive) and
       accept it if it contains a single unique numeric value.
    2. Fall back to parsing the filename for a `BR_...` token.
    Returns a float or None.
    """
    # Try from DataFrame columns first
    try:
        if df is not None:
            for col in df.columns:
                if __import__('re').search(r'br', str(col), flags=__import__('re').IGNORECASE):
                    try:
                        numeric = pd.to_numeric(df[col], errors='coerce').dropna().astype(float).values
                        if numeric.size == 0:
                            continue
                        # use median/mode heuristics to accept nearly-constant columns
                        m = float(np.nanmedian(numeric))
                        frac_close = np.count_nonzero(np.isclose(numeric, m, rtol=1e-3, atol=1e-12)) / float(numeric.size)
                        if frac_close >= 0.9:
                            return m
                        # if most entries equal 1.0, treat as BR==1
                        frac_one = np.count_nonzero(np.isclose(numeric, 1.0, rtol=1e-6, atol=1e-12)) / float(numeric.size)
                        if frac_one >= 0.9:
                            return 1.0
                        # fallback: if unique rounded values collapse to one, return that
                        try:
                            u = np.unique(np.round(numeric, decimals=6))
                            if u.size == 1:
                                return float(u[0])
                        except Exception:
                            pass
                    except Exception:
                        continue
    except Exception:
        pass

    # Fallback to filename parsing (operate on the stem to avoid extensions)
    try:
        stem = Path(fp).stem
        m = __import__('re').search(r'BR[_-]?([0-9Ee.+-p]+)', stem, flags=__import__('re').IGNORECASE)
        if m:
            token = m.group(1).replace('p', '.')
            return float(token)
        else:
            # Try last underscore-separated token from the stem
            token = stem.split('_')[-1].replace('p', '.')
            return float(token)
    except Exception:
        return None


def plot_sensitivity_contours(mass_vals, caphi_vals, heat, levels, output_path,
                              title=None, cmap_name='viridis', use_log_scale=True, dpi=300,
                              smooth_sigma=0.0):
    """Plot sensitivity contours from a grid of expected signal events.

    Parameters
    ----------
    mass_vals : array-like
        1D sorted array of x-axis grid points (C_Zh).
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
        raise RuntimeError('X-axis and CaPhi must be positive to perform log-space interpolation.')

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
    # [FIXED]: Explicitly label the x-axis for C_Zh scans[cite: 3, 4]
    ax.set_xlabel(r'Effective $C_{Zh}$', fontsize=12)
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


def plot_sensitivity_contours_overlay(csv_paths, levels, output_path,
                                      title=None, cmap_name='tab10', use_log_scale=True,
                                      dpi=300, smooth_sigma=0.0, nx_grid=1000, ny_grid=1000,
                                      envelope=True, draw_heatmap=False):
    """
    Overlay contour lines from multiple CSV grids on the same C_Zh (x) vs CaPhi (y)
    plot. Each CSV is expected to contain columns `C_Zh`, `CaPhi`, and
    `N_signal` forming a complete rectangular grid. The function interpolates in
    log-log space (using SciPy's LinearNDInterpolator) and draws contours for the
    requested `levels` for each CSV, labeling them by BR (extracted from the
    CSV's `BR_mu` column if present, otherwise by filename).
    """
    if LinearNDInterpolator is None:
        raise RuntimeError('scipy.interpolate.LinearNDInterpolator is required; please install scipy.')

    datasets = []
    for p in csv_paths:
        mass_vals, caphi_vals, heat = prepare_grid_from_csv(p, 'N_signal')
        datasets.append({'path': p, 'mass_vals': mass_vals, 'caphi_vals': caphi_vals, 'heat': heat})

    # Build union grids for display (layout only) and evaluation grid in log-space
    union_mass = np.sort(np.unique(np.concatenate([d['mass_vals'] for d in datasets])))
    union_caphi = np.sort(np.unique(np.concatenate([d['caphi_vals'] for d in datasets])))

    # create evaluation grid in log-space spanning union range
    if np.any(union_mass <= 0) or np.any(union_caphi <= 0):
        raise RuntimeError('C_Zh and CaPhi must be positive for log-space interpolation.')

    LOGX = np.linspace(np.log10(float(union_mass.min())), np.log10(float(union_mass.max())), nx_grid)
    LOGY = np.linspace(np.log10(float(union_caphi.min())), np.log10(float(union_caphi.max())), ny_grid)
    LOGGX, LOGGY = np.meshgrid(LOGX, LOGY)
    eval_pts = np.column_stack((LOGGX.ravel(), LOGGY.ravel()))
    GX, GY = np.meshgrid(10.0 ** LOGX, 10.0 ** LOGY)

    fig, ax = plt.subplots(figsize=(10, 8))

    # base empty grid points for layout (do not color by N_signal when overlaying contours)
    pts = []
    for i, c in enumerate(union_caphi):
        for j, m in enumerate(union_mass):
            pts.append((m, c))
    arr = np.array(pts, dtype=object)
    mass_arr = arr[:, 0].astype(float)
    caphi_arr = arr[:, 1].astype(float)
    ax.scatter(mass_arr, caphi_arr, facecolors='none', edgecolors='lightgrey', s=120, linewidths=0.8, zorder=2)

    # choose colors for overlay contours and prepare legend handles
    color_map = plt.get_cmap('tab10')
    legend_handles = []

    # Precompute interpolated grids for each dataset so we can operate
    # group-wise and form BR envelopes identical to 8.py's behavior.
    gz_list = []
    for d in datasets:
        try:
            x = np.asarray(d['mass_vals'], dtype=float)
            y = np.asarray(d['caphi_vals'], dtype=float)
            Z2 = np.array(d['heat'], dtype=float)

            positive_mask = Z2 > 0
            if np.any(positive_mask):
                min_pos = float(np.min(Z2[positive_mask]))
                max_pos = float(np.max(Z2[positive_mask]))
                eps = max(min_pos * 1e-3, 1e-20)
            else:
                eps = 1e-15
                max_pos = eps

            Z_safe = np.where(Z2 > 0, Z2, eps)
            zlog = np.log10(Z_safe)

            # build scattered points in log-space for interpolation
            xlog = np.log10(x)
            ylog = np.log10(y)
            XX_loc, YY_loc = np.meshgrid(xlog, ylog)
            pts_loc = np.column_stack((XX_loc.ravel(), YY_loc.ravel()))
            vals_loc = zlog.ravel()
            finite_mask = np.isfinite(vals_loc)
            pts_f = pts_loc[finite_mask]
            vals_f = vals_loc[finite_mask]
            if pts_f.shape[0] < 3:
                gz_list.append(None)
                continue

            interp = LinearNDInterpolator(pts_f, vals_f)
            GZ_log = interp(eval_pts).reshape(LOGGX.shape)

            # mask outside convex hull
            try:
                if ConvexHull is not None and pts_f.shape[0] >= 3:
                    hull = ConvexHull(pts_f)
                    hull_path = MplPath(pts_f[hull.vertices])
                    inside = hull_path.contains_points(eval_pts)
                    GZ_log_flat = GZ_log.ravel()
                    GZ_log_flat[~inside] = np.nan
                    GZ_log = GZ_log_flat.reshape(GZ_log.shape)
            except Exception:
                pass

            # optional smoothing
            if gaussian_filter is not None and smooth_sigma is not None and smooth_sigma > 0:
                try:
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
            if smooth_sigma is not None and smooth_sigma > 0 and np.any(positive_mask):
                try:
                    GZ = np.clip(GZ, eps, max_pos)
                except Exception:
                    pass

            gz_list.append(GZ)
        except Exception:
            gz_list.append(None)

    # Group datasets by their filename prefix before the `_BR_` token so
    # different BR scans of the same dataset share a color and can be
    # envelope-merged correctly.
    group_keys = [os.path.basename(d['path']).split('_BR_')[0] for d in datasets]
    unique_groups = []
    for g in group_keys:
        if g not in unique_groups:
            unique_groups.append(g)

    group_to_color_idx = {}
    next_auto_idx = 0
    for g in unique_groups:
        lg = g.lower()
        if 'anubis' in lg or ('mumu' in lg and 'higgs_signal_events' in lg):
            group_to_color_idx[g] = 2
        elif 'mathusla' in lg:
            group_to_color_idx[g] = 1
        elif 'lhc' in lg or 'atlas' in lg or 'cms' in lg:
            group_to_color_idx[g] = 0
        else:
            group_to_color_idx[g] = next_auto_idx
            next_auto_idx += 1

    # PHASE 1: build processed_items with metadata
    processed_items = []
    for idx, d in enumerate(datasets):
        GZ = gz_list[idx]
        if GZ is None:
            # skip datasets that failed interpolation
            continue

        base_name = os.path.basename(d['path'])
        grp = base_name.split('_BR_')[0]
        grp_lower = grp.lower() if isinstance(grp, str) else ''

        # choose a color
        try:
            if 'higgs_signal_events_data' in grp_lower:
                color = 'blue'
            elif grp_lower == 'mathusla40':
                color = 'red'
            elif grp_lower == 'mathusla' or grp_lower == 'mathusla200':
                color = 'orange'
            else:
                cidx = group_to_color_idx.get(grp, idx)
                color = _tab10_color_by_index(cidx)
        except Exception:
            color = _tab10_color_by_index(group_to_color_idx.get(grp, idx))

        # determine BR robustly
        br_val = None
        try:
            df_tmp2 = pd.read_csv(d['path'])
            br_val = _extract_br_value(d['path'], df=df_tmp2)
        except Exception:
            br_val = _extract_br_value(d['path'], df=None)

        linestyle = ':' if (br_val is not None and not np.isclose(br_val, 1.0)) else '-'

        # compute z-orders: ANUBIS/higgs highest, then MATHUSLA40, then MATHUSLA, then LHC
        try:
            if ('anubis' in grp_lower) or ('higgs_signal' in grp_lower) or ('higgs_signal_events' in grp_lower):
                z_cs = 260 + idx
                z_cs4 = 270 + idx
            elif 'mathusla40' in grp_lower:
                z_cs = 240 + idx
                z_cs4 = 250 + idx
            elif 'mathusla' in grp_lower:
                z_cs = 220 + idx
                z_cs4 = 230 + idx
            elif 'lhc' in grp_lower or 'atlas' in grp_lower or 'cms' in grp_lower:
                z_cs = 200 + idx
                z_cs4 = 210 + idx
            else:
                z_cs = 60 + idx
                z_cs4 = 70 + idx
        except Exception:
            z_cs = 60 + idx
            z_cs4 = 70 + idx

        # group display (prefer explicit MATHUSLA200 simulated label when present)
        try:
            if 'mathusla40' in grp_lower:
                grp_display = 'MATHUSLA40 (simulated)'
            elif 'mathusla200' in grp_lower or 'mathusla' in grp_lower:
                grp_display = 'MATHUSLA200 (simulated)'
            elif 'higgs_signal_events_data' in grp_lower:
                grp_display = 'ANUBIS'
            else:
                grp_display = grp
            legend_label = grp_display
        except Exception:
            grp_display = grp
            legend_label = base_name

        processed_items.append({
            'idx': idx, 'd': d, 'GZ': GZ, 'grp': grp, 'color': color,
            'linestyle': linestyle, 'label': base_name, 'br_val': br_val,
            'z_cs': z_cs, 'z_cs4': z_cs4, 'legend_label': legend_label,
            'grp_display': grp_display
        })

    # Assign distinct line colors per BR within each group so individual
    # BR contours are visually distinguishable (fills still use group color).
    try:
        cmap_lines = plt.get_cmap('tab10')
        groups_for_colors = {}
        for item in processed_items:
            groups_for_colors.setdefault(item['grp'], []).append(item)
        for grp, items in groups_for_colors.items():
            # sort by BR ascending for consistent color assignment
            items_sorted = sorted(items, key=lambda it: float(it['br_val']) if it['br_val'] is not None else 0.0)
            for k, it in enumerate(items_sorted):
                it['line_color'] = cmap_lines(k % cmap_lines.N)
    except Exception:
        for item in processed_items:
            item['line_color'] = item.get('color')

    # PHASE 2: apply BR envelope if requested
    if envelope:
        groups = {}
        for item in processed_items:
            groups.setdefault(item['grp'], []).append(item)

        for g_name, items in groups.items():
            items.sort(key=lambda x: float(x['br_val']) if x['br_val'] is not None else 0.0)
            running_gz = None
            for item in items:
                if item['GZ'] is None:
                    continue
                if running_gz is None:
                    running_gz = item['GZ'].copy()
                else:
                    running_gz = np.fmax(running_gz, item['GZ'])
                    item['GZ'] = running_gz.copy()

        # restore original plotting order
        processed_items.sort(key=lambda x: x['idx'])

    # PHASE 3: draw contours and accumulate 4-event masks per group
    group_masks = {}
    group_colors = {}
    group_zords = {}

    for item in processed_items:
        idx = item['idx']
        GZ = item['GZ']
        if GZ is None:
            continue

        try:
            levels_non4 = [lv for lv in levels if not np.isclose(lv, 4.0)]
            has_4 = any(np.isclose(levels, 4.0))

            if len(levels_non4) > 0:
                ax.contour(GX, GY, GZ, levels=levels_non4, colors=[item.get('line_color', item['color'])], linewidths=1.8, linestyles=item['linestyle'], zorder=item['z_cs'])

            if has_4:
                try:
                    cs4 = ax.contour(GX, GY, GZ, levels=[4.0], colors=[item.get('line_color', item['color'])], linewidths=3.2, linestyles=item['linestyle'], zorder=item['z_cs4'])
                    mask = np.isfinite(GZ) & (GZ >= 4.0)
                    if np.any(mask):
                        grp = item['grp']
                        if grp not in group_masks:
                            group_masks[grp] = mask
                            group_colors[grp] = item['color']
                            group_zords[grp] = item['z_cs4'] - 5
                        else:
                            group_masks[grp] = group_masks[grp] | mask
                except Exception:
                    pass
        except Exception as e:
            print(f'Warning: failed to draw contours for {item["d"]["path"]}: {e}')
            continue

        # legend handles: add per-BR entries so lines map to BR values
        try:
            if item.get('br_val') is not None:
                try:
                    brf = float(item['br_val'])
                    label = f"{item['grp_display']} (BR={brf:g})"
                except Exception:
                    label = f"{item['grp_display']} (BR={item['br_val']})"
            else:
                label = item.get('label', item.get('legend_label', ''))
            legend_handles.append(Line2D([0], [0], color=item.get('line_color', item['color']), lw=2.4, linestyle=item['linestyle'], label=label))
        except Exception:
            pass

    # Draw unified fills for groups
    for grp_fill, mask_fill in group_masks.items():
        try:
            mask_float = np.where(mask_fill, 1.0, np.nan)
            ax.contourf(GX, GY, mask_float, levels=[0.5, 1.5], colors=[group_colors[grp_fill]], alpha=0.15, zorder=group_zords[grp_fill])
        except Exception as e:
            print(f"Warning: failed to draw filled contour for group {grp_fill}: {e}")

    # draw legend
    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc='lower left', title='Datasets (BR)', framealpha=0.9, edgecolor='grey')
        legend.set_zorder(100)

    # draw legend (no unified colorbar — contours convey the 4-event boundary per BR)
    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc='lower left', title='Datasets (BR)', framealpha=0.9, edgecolor='grey')
        legend.set_zorder(100)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Effective $C_{Zh}$', fontsize=12)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=12)
    if title is None:
        title = 'Sensitivity Contours (overlaid BR scans)'
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    out_path = Path(output_path)
    os.makedirs(out_path.parent or '.', exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    try:
        pdf_path = out_path.with_suffix('.pdf')
        plt.savefig(str(pdf_path), bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def main():
    default_csv = os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA200_higgs_signal_events_data.csv')
    parser = argparse.ArgumentParser(description='Plot sensitivity contours from Higgs signal-events CSV')
    parser.add_argument('--csv', type=str, default=default_csv, help='Path to Simulated_MATHUSLA200_higgs_signal_events_data.csv')
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA200_sensitivity_contours.png'), help='Output PNG path')
    parser.add_argument('--levels', type=str, default='4', help='Comma-separated contour levels (event counts)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Gaussian smoothing sigma (log-grid units). Use 0 to disable smoothing, use 1 for final plot')
    parser.add_argument('--no-log', dest='use_log', action='store_false', help='Do not use log normalization/scales')
    parser.add_argument('--envelope', dest='envelope', action='store_true', help='When overlaying BR scans, include data from BRs <= current BR to form an envelope')
    parser.set_defaults(use_log=True, envelope=False)
    args = parser.parse_args()

    csv_path = args.csv
    # Expand potential glob patterns or detect per-BR CSVs in the Plots folder
    csv_list = []
    # If user provided a glob pattern, expand it
    if '*' in csv_path or '?' in csv_path:
        csv_list = sorted(glob.glob(csv_path))
    elif os.path.isdir(csv_path):
        csv_list = sorted(glob.glob(os.path.join(csv_path, 'Simulated_MATHUSLA200_higgs_signal_events_data_mumu_BR_*.csv')))
    else:
        # If using default csv and per-BR mumu CSVs exist, prefer those for overlay
        if os.path.abspath(csv_path) == os.path.abspath(default_csv):
            pb_pattern = os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA200_higgs_signal_events_data_mumu_BR_*.csv')
            pb_files = sorted(glob.glob(pb_pattern))
            if pb_files:
                csv_list = pb_files
            else:
                csv_list = [csv_path]
        else:
            csv_list = [csv_path]

    csv_list = [p for p in csv_list if p and os.path.exists(p)]
    if not csv_list:
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    levels = [float(x) for x in args.levels.split(',') if x.strip()]

    if len(csv_list) > 1:
        print(f'Found multiple CSVs: {csv_list} -> overlaying BR contours on same plot')
        # [FIXED]: Metadata parsing using 'C_Zh' verbatim[cite: 3, 4]
        czh_text = ''
        try:
            df_full = pd.read_csv(csv_list[0])
            if 'C_Zh' in df_full.columns:
                unique_vals = pd.unique(df_full['C_Zh'])
                try:
                    fv = sorted([float(v) for v in unique_vals])
                except Exception:
                    fv = list(unique_vals)
                if len(fv) == 1:
                    mass_text = ', $m_a=' + f"{mv[0]:g}" + '\\,\\mathrm{GeV}$'
                else:
                    try:
                        czh_text = ', $C_{Zh}^{eff}\\in[' + f"{min(fv):g}" + ',' + f"{max(fv):g}" + ']$'
                    except Exception:
                        czh_text = ', $C_{Zh}^{eff}=\\mathrm{various}$'
        except Exception:
            pass

        title_base = 'Simulated MATHUSLA200 Expected Signal Events for Fermion-Coupled ALPs, $pp\\to H \\to Z a$'
        # Try to add ALP mass to the title if present in the CSVs
        mass_text = ''
        try:
            dfm = pd.read_csv(csv_list[0])
            if 'mass' in dfm.columns:
                uniqm = pd.unique(dfm['mass'])
                try:
                    mv = sorted([float(v) for v in uniqm])
                except Exception:
                    mv = list(uniqm)
                if len(mv) == 1:
                    mass_text = f', $m_a={mv[0]:g}\\,\\mathrm{{GeV}}$'
                else:
                    try:
                        mass_text = f', $m_a\\in[{min(mv):g},{max(mv):g}]$'
                    except Exception:
                        mass_text = ''
        except Exception:
            pass

        title = title_base + czh_text + mass_text
        # Use 4-event contours for each BR overlay as requested
        levels_overlay = [4.0]
        plot_sensitivity_contours_overlay(csv_list, levels_overlay, args.output, title=title,
                          use_log_scale=args.use_log, smooth_sigma=args.sigma, envelope=args.envelope)
    else:
        # [FIXED]: Mapping based on C_Zh column[cite: 3]
        mass_vals, caphi_vals, heat = prepare_grid_from_csv(csv_list[0], 'N_signal')
        print(f'Loaded {len(mass_vals)} C_Zh points × {len(caphi_vals)} coupling points from {csv_list[0]}')
        print(f'Plotting contours at levels: {levels} -> saving to {args.output}')

        # Try to extract the Higgs coupling `C_Zh` from the CSV and append to title
        czh_text = ''
        try:
            df_full = pd.read_csv(csv_list[0])
            if 'C_Zh' in df_full.columns:
                unique_vals = pd.unique(df_full['C_Zh'])
                try:
                    fv = sorted([float(v) for v in unique_vals])
                except Exception:
                    fv = list(unique_vals)
                if len(fv) == 1:
                    czh_text = ', $C_{Zh}^{eff}=' + f"{fv[0]:g}" + '$'
                else:
                    try:
                        czh_text = ', $C_{Zh}^{eff}\\in[' + f"{min(fv):g}" + ',' + f"{max(fv):g}" + ']$'
                    except Exception:
                        czh_text = ', $C_{Zh}^{eff}=\\mathrm{various}$'
        except Exception:
            # ignore errors and leave title without coupling
            pass

        title_base = 'Simulated MATHUSLA200 Expected Signal Events for Fermion-Coupled ALPs, $pp\\to H \\to Z a$'
        # Append ALP mass to single-file title if available
        mass_text = ''
        try:
            dfm = pd.read_csv(csv_list[0])
            if 'mass' in dfm.columns:
                uniqm = pd.unique(dfm['mass'])
                try:
                    mv = sorted([float(v) for v in uniqm])
                except Exception:
                    mv = list(uniqm)
                if len(mv) == 1:
                    mass_text = f', $m_a={mv[0]:g}\\,\\mathrm{{GeV}}$'
                else:
                    try:
                        mass_text = f', $m_a\\in[{min(mv):g},{max(mv):g}]$'
                    except Exception:
                        mass_text = ''
        except Exception:
            pass

        title = title_base + czh_text + mass_text

        plot_sensitivity_contours(mass_vals, caphi_vals, heat, levels, args.output,
                                  title=title,
                                  use_log_scale=args.use_log, smooth_sigma=args.sigma)


if __name__ == '__main__':
    main()