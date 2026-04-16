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
import glob
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
import re
try:
    # Use LinearNDInterpolator to prevent cubic "overshoot" loops and artifacts
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull
except Exception:
    LinearNDInterpolator = None
    gaussian_filter = None
    ConvexHull = None


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


# NOTE: Branching-ratio display helpers removed — labels are handled
# in post-production to keep this script minimal.


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
                if re.search(r'br', str(col), flags=re.IGNORECASE):
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
        m = re.search(r'BR[_-]?([0-9Ee.+-p]+)', stem, flags=re.IGNORECASE)
        if m:
            token = m.group(1).replace('p', '.')
            return float(token)
        else:
            # Try last underscore-separated token from the stem
            token = stem.split('_')[-1].replace('p', '.')
            return float(token)
    except Exception:
        return None


def _place_label_on_contour(ax, cs, text, color='k', fontsize=14, y_factor=1.0):
    """Place a single text label on the longest path of the given QuadContourSet.

    This chooses the longest path from the contour collection and places a
    text label at its midpoint. Fails silently on any error.
    """
    try:
        if cs is None:
            return
        for coll in cs.collections:
            paths = coll.get_paths()
            if not paths:
                continue
            longest = max(paths, key=lambda p: p.vertices.shape[0])
            verts = longest.vertices
            if verts is None or verts.size == 0:
                continue
            mid = len(verts) // 2
            xm, ym = verts[mid]
            try:
                ym = float(ym) * float(y_factor)
            except Exception:
                ym = float(ym)
            ax.text(xm, ym, text, color=color, fontsize=fontsize,
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='none'),
                    zorder=200)
            return
    except Exception:
        pass

def _place_label_on_line(ax, x, y, text, color='k', fontsize=14, y_factor=1.0, anchor='mid'):
    """Place a small label approximately at the midpoint of a polyline.

    Used for experimental polylines (limits). Fails silently on any error.
    """
    try:
        if len(x) == 0 or len(y) == 0:
            return
        # choose index based on requested anchor; default is midpoint
        if anchor == 'mid' or anchor is None:
            idx = len(x) // 2
        elif anchor == 'last' or anchor == 'end':
            idx = None
            for k in range(len(x) - 1, -1, -1):
                try:
                    xm_test = float(x[k])
                    ym_test = float(y[k])
                    if np.isfinite(xm_test) and np.isfinite(ym_test):
                        idx = k
                        break
                except Exception:
                    continue
            if idx is None:
                return
        elif anchor == 'first':
            idx = None
            for k in range(len(x)):
                try:
                    xm_test = float(x[k])
                    ym_test = float(y[k])
                    if np.isfinite(xm_test) and np.isfinite(ym_test):
                        idx = k
                        break
                except Exception:
                    continue
            if idx is None:
                return
        else:
            # unknown anchor -> midpoint
            idx = len(x) // 2

        xm = float(x[idx])
        ym = float(y[idx])
        # Apply multiplicative vertical offset (safe on log-scaled axis)
        try:
            ym = float(ym) * float(y_factor)
        except Exception:
            ym = float(ym)
        ax.text(xm, ym, text, color=color, fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='none'),
                zorder=200)
    except Exception:
        pass


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


def _collect_experiment_contour_handles(ax, base_dir=None, patterns=None):
    """
    Find and plot experimental contour polylines (CSV pairs of x,y) for
    LHC and MATHUSLA. Returns a list of Line2D legend handles.

    Files are expected to be named like `LHC_BR_1.csv` or `MATHUSLA_BR_0.1.csv`.
    BR==1 entries are drawn solid and thicker; BR<1 entries are dotted.
    """
    if base_dir is None:
        base_dir = Path(os.path.dirname(__file__))
    else:
        base_dir = Path(base_dir)

    if patterns is None:
        # third element is an index into the `tab10` colormap for consistent colors
        patterns = [
            ('LHC', 'LHC_BR_*.csv', 0),
            ('MATHUSLA', 'MATHUSLA_BR_*.csv', 1),
            ('ANUBIS', 'ANUBIS_BR_*.csv', 2),
        ]

    color_map = plt.get_cmap('tab10')

    handles = []
    # (no explicit label placement by default)
    for dataset, pattern, color in patterns:
        # search both the base_dir and its 'Plots' subdirectory for experiment CSVs
        candidates = []
        try:
            candidates.extend(sorted(glob.glob(str(base_dir / pattern))))
        except Exception:
            pass
        try:
            candidates.extend(sorted(glob.glob(str(base_dir / 'Plots' / pattern))))
        except Exception:
            pass

        candidates = sorted(set(candidates))
        if not candidates:
            # debugging: show none found for this pattern
            try:
                print(f"[DEBUG_EXP_SEARCH] pattern={pattern} base_dir={base_dir} found=0")
            except Exception:
                pass

        for fp in candidates:
            # Read the CSV first (we may be able to extract BR from a column)
            try:
                df = pd.read_csv(fp, comment='#', header=0)
                x = df.iloc[:, 0].astype(float).values
                y = df.iloc[:, 1].astype(float).values
            except Exception:
                continue

            # Extract BR preferentially from the CSV, falling back to filename
            try:
                br = _extract_br_value(fp, df=df)
            except Exception:
                br = None

            # For BR<1 contours (ANUBIS, LHC, MATHUSLA), avoid modifying
            # files. Instead, when placing inline labels anchored to the
            # 'last' point, prefer the last point that lies within the
            # current axes limits (if available). We'll compute a trimmed
            # x/y view used only for label anchoring below.
            try:
                # compute a default trimmed view equal to full arrays
                x_for_label = x
                y_for_label = y
                d_up_tmp = str(dataset).upper() if dataset is not None else ''
                is_br1_tmp = (br is not None and np.isclose(br, 1.0))
                if (br is not None) and (not is_br1_tmp):
                    try:
                        # Try to fetch current axis view limits; fall back
                        # to accepting the full arrays if unavailable.
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        # widen the allowed window slightly to account for
                        # plotting margins
                        x_margin = (abs(xlim[1] - xlim[0]) * 0.05) if (xlim[1] != xlim[0]) else 0.0
                        y_margin = (abs(ylim[1] - ylim[0]) * 0.05) if (ylim[1] != ylim[0]) else 0.0
                        last_idx_in_view = None
                        for k in range(len(x) - 1, -1, -1):
                            try:
                                xv = float(x[k]); yv = float(y[k])
                                if not (np.isfinite(xv) and np.isfinite(yv)):
                                    continue
                                if (xv >= (xlim[0] - x_margin) and xv <= (xlim[1] + x_margin) and
                                        yv >= (ylim[0] - y_margin) and yv <= (ylim[1] + y_margin)):
                                    last_idx_in_view = k
                                    break
                            except Exception:
                                continue
                        if last_idx_in_view is not None:
                            x_for_label = x[:last_idx_in_view + 1]
                            y_for_label = y[:last_idx_in_view + 1]
                    except Exception:
                        # If fetching axis limits or trimming fails, ignore
                        x_for_label = x
                        y_for_label = y
            except Exception:
                x_for_label = x
                y_for_label = y

            is_br1 = (br is not None and np.isclose(br, 1.0))
            linestyle = '-' if is_br1 else ':'
            lw = 2.8 if is_br1 else 1.6
            # Legend label: use dataset name only (no BR suffix).
            label = dataset
            # For ANUBIS BR==1, prefer a simple lowercase legend label 'anubis'
            try:
                if dataset is not None and str(dataset).upper().startswith('ANUBIS') and is_br1:
                    label = 'anubis'
            except Exception:
                pass
            # resolve color specification: prefer explicit dataset colors
            d_up = str(dataset).upper() if dataset is not None else ''
            if d_up.startswith('LHC'):
                plot_color = 'green'
            elif d_up.startswith('MATHUSLA'):
                plot_color = 'red'
            elif d_up.startswith('ANUBIS'):
                plot_color = 'blue'
            else:
                try:
                    color_idx = int(color)
                    plot_color = _tab10_color_by_index(color_idx)
                except Exception:
                    if isinstance(color, str):
                        plot_color = color
                    else:
                        plot_color = _tab10_color_by_index(0)

            ax.plot(x, y, color=plot_color, lw=lw, linestyle=linestyle, zorder=120)
            # For BR==1 (solid) keep a legend handle but do NOT append
            # any branching-ratio suffix; for BR<1 do not place inline labels.
            if is_br1:
                handles.append(Line2D([0], [0], color=plot_color, lw=lw, linestyle=linestyle, label=label))
            # For BR<1: intentionally do not place inline labels or use
            # explicit label_positions — user will add BR labels later.

    return handles


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

    fig, ax = plt.subplots(figsize=(8, 12))

    if use_log_scale and vmax > 0:
        vmin_safe = max(vmin, np.finfo(float).tiny)
        norm = colors.LogNorm(vmin=vmin_safe, vmax=vmax)
    else:
        norm = None

    # Do not plot discrete data-point markers here; contours alone
    # represent the interpolated results. No colorbar will be drawn.

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
            # Draw interpolated filled heatmap first so contours overlay cleanly.
            try:
                # Prefer a filled contour map for smooth, gap-free display on
                # log-scaled axes (contourf handles transformed coordinates
                # more robustly than pcolormesh in some Matplotlib versions).
                if GZ is not None:
                    data_heat = np.ma.masked_invalid(GZ)
                    X_plot, Y_plot = GX, GY
                else:
                    data_heat = np.ma.masked_invalid(Z_masked)
                    X_plot, Y_plot = X, Y

                try:
                    # Choose many levels in log-space for a smooth appearance
                    if use_log_scale and vmin > 0 and np.isfinite(vmax) and vmax > vmin:
                        low = max(vmin, np.finfo(float).tiny)
                        levels_heat = np.logspace(np.log10(low), np.log10(vmax), 200)
                    else:
                        finite_vals = data_heat[np.isfinite(data_heat)]
                        if finite_vals.size:
                            levels_heat = np.linspace(float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals)), 200)
                        else:
                            levels_heat = np.linspace(0.0, 1.0, 2)

                    cf = ax.contourf(X_plot, Y_plot, data_heat, levels=levels_heat, cmap=cmap, norm=norm, zorder=20)
                    try:
                        for coll in cf.collections:
                            try:
                                coll.set_edgecolor('face')
                            except Exception:
                                pass
                            try:
                                coll.set_linewidth(0.0)
                            except Exception:
                                pass
                            try:
                                coll.set_antialiased(False)
                            except Exception:
                                pass
                            try:
                                coll.set_rasterized(True)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    # Fallback to pcolormesh using computed edges when contourf fails
                    try:
                        if 'LOGX' in locals() and 'LOGY' in locals() and LOGX.size > 1 and LOGY.size > 1:
                            dx = LOGX[1] - LOGX[0]
                            dy = LOGY[1] - LOGY[0]
                            LOGX_edges = np.concatenate(([LOGX[0] - dx/2.0], (LOGX[:-1] + LOGX[1:]) / 2.0, [LOGX[-1] + dx/2.0]))
                            LOGY_edges = np.concatenate(([LOGY[0] - dy/2.0], (LOGY[:-1] + LOGY[1:]) / 2.0, [LOGY[-1] + dy/2.0]))
                            X_edges = 10.0 ** LOGX_edges
                            Y_edges = 10.0 ** LOGY_edges
                            Xedges2D, Yedges2D = np.meshgrid(X_edges, Y_edges)
                            try:
                                pcm = ax.pcolormesh(Xedges2D, Yedges2D, data_heat, cmap=cmap, norm=norm, shading='auto', antialiased=False, zorder=20)
                            except Exception:
                                pcm = ax.pcolormesh(Xedges2D, Yedges2D, data_heat, shading='auto', zorder=20)
                        else:
                            try:
                                pcm = ax.pcolormesh(X_plot, Y_plot, data_heat, cmap=cmap, norm=norm, shading='auto', antialiased=False, zorder=20)
                            except Exception:
                                ax.pcolormesh(X_plot, Y_plot, data_heat, shading='auto', zorder=20)
                    except Exception:
                        pass
            except Exception:
                pass

            if GZ is not None:
                if len(levels_non4) > 0:
                    cs_main = ax.contour(GX, GY, GZ, levels=levels_non4, colors='black', linewidths=1.6, zorder=60)
                if has_4:
                    # Shade the area >= 4.0 events (highlight) on top of the heatmap
                    try:
                        mask = np.isfinite(GZ) & (GZ >= 4.0)
                        if np.any(mask):
                            mask_float = np.where(mask, 1.0, np.nan)
                            ax.contourf(GX, GY, mask_float, levels=[0.5, 1.5], colors=['red'], alpha=0.15, zorder=50)
                    except Exception:
                        try:
                            ax.contourf(GX, GY, GZ, levels=[4.0, float(np.nanmax(GZ)) * 10.0], colors=['red'], alpha=0.15, zorder=50)
                        except Exception:
                            pass
                    cs4 = ax.contour(GX, GY, GZ, levels=[4.0], colors='red', linewidths=3.0, zorder=70)
            else:
                if len(levels_non4) > 0:
                    cs_main = ax.contour(X, Y, Z_masked, levels=levels_non4, colors='black', linewidths=1.6, zorder=60)
                if has_4:
                    try:
                        mask = np.isfinite(Z_masked) & (Z_masked >= 4.0)
                        if np.any(mask):
                            mask_float = np.where(mask, 1.0, np.nan)
                            ax.contourf(X, Y, mask_float, levels=[0.5, 1.5], colors=['red'], alpha=0.15, zorder=50)
                    except Exception:
                        try:
                            ax.contourf(X, Y, Z_masked, levels=[4.0, float(np.nanmax(Z_masked)) * 10.0], colors=['red'], alpha=0.15, zorder=50)
                        except Exception:
                            pass
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
                               framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
            legend.set_zorder(100) # Keep the legend above the scatter plot

        # ---------------------------------------------------------
        # Overlay experimental contour polylines (LHC, MATHUSLA, ANUBIS)
        # - BR==1: solid, thicker; BR<1: dotted
        # ---------------------------------------------------------
        try:
            exp_handles = _collect_experiment_contour_handles(ax, base_dir=os.path.dirname(__file__))
            if exp_handles:
                # sanitize experimental handle labels to avoid raw filenames leaking
                try:
                    for h in exp_handles:
                        try:
                            lab = h.get_label()
                            if isinstance(lab, str) and 'higgs_signal_events_data' in lab.lower():
                                # preserve any BR suffix in parentheses
                                if '(' in lab:
                                    suffix = lab[lab.find('('):]
                                    h.set_label('ANUBIS ' + suffix)
                                else:
                                    h.set_label('ANUBIS')
                        except Exception:
                            pass
                except Exception:
                    pass

                if 'legend' in locals():
                    # extend existing legend handles list by creating a new legend merging both
                    all_handles = legend_handles + exp_handles
                    legend = ax.legend(handles=all_handles, loc='lower left', title='Contour Levels / Limits', 
                                       framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
                    legend.set_zorder(100)
                else:
                    legend = ax.legend(handles=exp_handles, loc='lower left', title='Limits', framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
                    legend.set_zorder(100)
        except Exception:
            pass

    # No colorbar requested — contours/legends suffice.

    ax.set_xscale('log')
    ax.set_yscale('log')
    # Force y-axis range as requested
    try:
        ax.set_ylim(1e-7, 100.0)
    except Exception:
        pass
    ax.set_xlabel('ALP Mass [GeV]', fontsize=16)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=16)
    if title is None:
        title = 'Sensitivity Contours: Expected Signal Events'
    ax.set_title(title, fontsize=18)
    # Increase tick label sizes and enable minor ticks for log axes
    try:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.minorticks_on()
    except Exception:
        pass
    # Show major and minor grid lines on both axes (including y log-grid)
    ax.grid(which='major', axis='both', alpha=0.45, linestyle='--', linewidth=0.6)
    ax.grid(which='minor', axis='both', alpha=0.25, linestyle=':', linewidth=0.4)
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
                                      dpi=300, smooth_sigma=0.0, draw_heatmap=False, nx_grid=1000, ny_grid=1000):
    """
    Overlay contour lines from multiple CSV grids on the same C_Zh (x) vs CaPhi (y)
    plot. Each CSV is expected to contain columns `mass` (here: C_Zh), `CaPhi`, and
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
        raise RuntimeError('Mass (C_Zh) and CaPhi must be positive for log-space interpolation.')

    LOGX = np.linspace(np.log10(float(union_mass.min())), np.log10(float(union_mass.max())), nx_grid)
    LOGY = np.linspace(np.log10(float(union_caphi.min())), np.log10(float(union_caphi.max())), ny_grid)
    LOGGX, LOGGY = np.meshgrid(LOGX, LOGY)
    eval_pts = np.column_stack((LOGGX.ravel(), LOGGY.ravel()))
    GX, GY = np.meshgrid(10.0 ** LOGX, 10.0 ** LOGY)

    fig, ax = plt.subplots(figsize=(8, 12))
    # Removed plotting of discrete grid markers in overlay mode; only contours are drawn.

    # Use log scales and set y-limits early so experiment fills reach the top
    try:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-7, 100.0)
    except Exception:
        pass

    # choose colors for overlay contours and prepare legend handles
    color_map = plt.get_cmap('tab10')
    legend_handles = []

    # Precompute interpolated grids for each dataset so we can draw a
    # combined background heatmap (use nanmax across datasets).
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

    # Combine per-dataset grids (use nanmax so the brightest features show)
    valid_gzs = [g for g in gz_list if g is not None]
    combined_GZ = None
    if valid_gzs:
        try:
            stack = np.stack(valid_gzs, axis=0)
            combined_GZ = np.nanmax(stack, axis=0)
        except Exception:
            combined_GZ = None
        # Post-process combined grid to reduce interpolation/masking
        # artifacts (horizontal banding). Smooth in log-space with an
        # anisotropic kernel that smooths more in the vertical (y)
        # direction to remove horizontal stripes while preserving
        # sharp features along x.
        try:
            if combined_GZ is not None and gaussian_filter is not None:
                # Work in log10 space for dynamic range stability
                pos_mask = np.isfinite(combined_GZ) & (combined_GZ > 0)
                if np.any(pos_mask):
                    logG = np.full_like(combined_GZ, np.nan, dtype=float)
                    logG[pos_mask] = np.log10(combined_GZ[pos_mask])
                    med_log = np.nanmedian(logG)
                    if np.isfinite(med_log):
                        filled = np.where(np.isfinite(logG), logG, med_log)
                    else:
                        filled = np.where(np.isfinite(logG), logG, 0.0)
                    # stronger smoothing along axis=0 (vertical) to remove
                    # horizontal banding; preserve x-detail with smaller sigma
                    try:
                        smoothed_log = gaussian_filter(filled, sigma=(2.6, 0.8), mode='nearest')
                        # restore NaNs outside original valid region
                        smoothed_log[~np.isfinite(logG)] = np.nan
                        combined_GZ = np.where(np.isfinite(smoothed_log), 10.0 ** smoothed_log, np.nan)
                    except Exception:
                        pass
        except Exception:
            pass

    # Draw a combined heatmap underneath contours for context (optional)
    if draw_heatmap and combined_GZ is not None:
        try:
            # Apply a small Gaussian smoothing to reduce interpolation /
            # masking banding artifacts when stacking per-dataset grids.
            # We fill NaNs with the median temporarily, smooth, then
            # restore NaNs outside the original valid region so we don't
            # extrapolate into empty space.
            try:
                if gaussian_filter is not None:
                    med_cgz = np.nanmedian(combined_GZ)
                    if np.isfinite(med_cgz):
                        filled_cgz = np.where(np.isfinite(combined_GZ), combined_GZ, med_cgz)
                    else:
                        filled_cgz = np.where(np.isfinite(combined_GZ), combined_GZ, 0.0)
                    # small sigma to remove narrow bands but preserve features
                    try:
                        smoothed_cgz = gaussian_filter(filled_cgz, sigma=1.2, mode='nearest')
                        smoothed_cgz[~np.isfinite(combined_GZ)] = np.nan
                        combined_GZ = smoothed_cgz
                    except Exception:
                        pass
            except Exception:
                pass

            data_heat = np.ma.masked_invalid(combined_GZ)
            positive_vals = data_heat.compressed()[data_heat.compressed() > 0] if data_heat.count() > 0 else np.array([])
            if positive_vals.size > 0:
                h_vmin = float(np.min(positive_vals))
                h_vmax = float(np.max(positive_vals))
            else:
                h_vmin = np.finfo(float).tiny
                h_vmax = 1.0

            # choose many log-spaced levels for a smooth colorfield
            if use_log_scale and h_vmin > 0 and np.isfinite(h_vmax) and h_vmax > h_vmin:
                heat_levels = np.logspace(np.log10(max(h_vmin, np.finfo(float).tiny)), np.log10(h_vmax), 200)
                heat_norm = colors.LogNorm(vmin=max(h_vmin, np.finfo(float).tiny), vmax=h_vmax)
            else:
                finite_vals = data_heat[np.isfinite(data_heat)]
                if finite_vals.size:
                    heat_levels = np.linspace(float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals)), 200)
                    heat_norm = None
                else:
                    heat_levels = np.linspace(0.0, 1.0, 2)
                    heat_norm = None

            try:
                # contourf is more robust on transformed axes. Configure the
                # returned collections to avoid edge-drawing banding and to
                # rasterize the filled polygons for cleaner output.
                cf = ax.contourf(GX, GY, data_heat, levels=heat_levels, cmap='viridis', norm=heat_norm, zorder=10)
                try:
                    for coll in cf.collections:
                        try:
                            coll.set_edgecolor('face')
                        except Exception:
                            pass
                        try:
                            coll.set_linewidth(0.0)
                        except Exception:
                            pass
                        try:
                            coll.set_antialiased(False)
                        except Exception:
                            pass
                        try:
                            coll.set_rasterized(True)
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                # fallback to pcolormesh with edges
                try:
                    dx = LOGX[1] - LOGX[0]
                    dy = LOGY[1] - LOGY[0]
                    LOGX_edges = np.concatenate(([LOGX[0] - dx/2.0], (LOGX[:-1] + LOGX[1:]) / 2.0, [LOGX[-1] + dx/2.0]))
                    LOGY_edges = np.concatenate(([LOGY[0] - dy/2.0], (LOGY[:-1] + LOGY[1:]) / 2.0, [LOGY[-1] + dy/2.0]))
                    X_edges = 10.0 ** LOGX_edges
                    Y_edges = 10.0 ** LOGY_edges
                    Xedges2D, Yedges2D = np.meshgrid(X_edges, Y_edges)
                    ax.pcolormesh(Xedges2D, Yedges2D, data_heat, cmap='viridis', norm=heat_norm, shading='auto', antialiased=False, zorder=10, rasterized=True)
                except Exception:
                    pass
        except Exception:
            pass

    # Group datasets by their filename prefix before the `_BR_` token so
    # different BR scans of the same dataset share a color. Map common
    # identifiers to experiment colors (ANUBIS->2, MATHUSLA->1, LHC->0).
    group_keys = [os.path.basename(d['path']).split('_BR_')[0] for d in datasets]
    unique_groups = []
    for g in group_keys:
        if g not in unique_groups:
            unique_groups.append(g)

    group_to_color_idx = {}
    next_auto_idx = 0
    for g in unique_groups:
        lg = g.lower()
        if 'anubis' in lg or 'mumu' in lg and 'higgs_signal_events_data' in lg:
            group_to_color_idx[g] = 2
        elif 'mathusla' in lg:
            group_to_color_idx[g] = 1
        elif 'lhc' in lg or 'atlas' in lg or 'cms' in lg:
            group_to_color_idx[g] = 0
        else:
            group_to_color_idx[g] = next_auto_idx
            next_auto_idx += 1

    for idx, d in enumerate(datasets):
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
            print(f'Warning: not enough points for interpolation in {d["path"]}; skipping')
            continue

        interp = LinearNDInterpolator(pts_f, vals_f)
        GZ_log = interp(eval_pts).reshape(LOGGX.shape)

        # mask outside convex hull to avoid extrapolation artifacts
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

        # choose a color based on the group prefix so BR variants match
        base_name = os.path.basename(d['path'])
        grp = base_name.split('_BR_')[0]
        # If this is the higgs signal dataset, treat it as ANUBIS (blue)
        try:
            if 'higgs_signal_events_data' in grp.lower():
                color = 'blue'
            else:
                cidx = group_to_color_idx.get(grp, idx)
                color = _tab10_color_by_index(cidx)
        except Exception:
            cidx = group_to_color_idx.get(grp, idx)
            color = _tab10_color_by_index(cidx)

        # determine BR from the CSV (or filename) to choose linestyle
        linestyle = '-'
        br_val = None
        try:
            # try to read once and extract BR robustly
            df_tmp2 = pd.read_csv(d['path'])
            br_val = _extract_br_value(d['path'], df=df_tmp2)
        except Exception:
            br_val = _extract_br_value(d['path'], df=None)

        if br_val is not None and not np.isclose(br_val, 1.0):
            linestyle = ':'

        # Compute a dataset label (use filename; do not include BR text)
        label = os.path.basename(d['path'])

        # Debug: log dataset BR detection and chosen label/color/linestyle
        try:
            print(f"[DEBUG_OVERLAY_PARSE] file={os.path.basename(d['path'])} br_val={br_val} label={label} linestyle={linestyle} color={color}")
        except Exception:
            pass

        # draw contours for this dataset (colored by BR/dataset)
        try:
            # separate the 4.0 highlight to avoid drawing the same level twice
            levels_non4 = [lv for lv in levels if not np.isclose(lv, 4.0)]
            has_4 = any(np.isclose(levels, 4.0))

            cs = None
            if len(levels_non4) > 0:
                cs = ax.contour(GX, GY, GZ, levels=levels_non4, colors=[color], linewidths=1.8, linestyles=linestyle, zorder=60 + idx)

            # highlight the 4.0 level if present (single thicker line)
            cs4 = None
            if has_4:
                try:
                    cs4 = ax.contour(GX, GY, GZ, levels=[4.0], colors=[color], linewidths=3.2, linestyles=linestyle, zorder=70 + idx)
                except Exception:
                    cs4 = None

            # Do not place branching-ratio inline labels here; labels will
            # be added in post-production if needed. The contours are drawn
            # above and legend entries will indicate dataset groups only.
        except Exception as e:
            print(f'Warning: failed to draw contours for {d["path"]}: {e}')
            continue
        except Exception as e:
            print(f'Warning: failed to draw contours for {d["path"]}: {e}')
            continue

        # Add legend handle for this dataset (show group prefix + BR when available)
        try:
            # Map raw higgs signal filenames to a cleaner ANUBIS group name
            grp_lower = grp.lower() if isinstance(grp, str) else ''
            if 'higgs_signal_events_data' in grp_lower:
                grp_display = 'ANUBIS'
            else:
                grp_display = grp

            # Use group display name only (do not append BR label)
            legend_label = grp_display
        except Exception:
            legend_label = label

        # Do NOT add ANUBIS BR<1 entries to the legend; label them inline on-plot instead
        try:
            if not (isinstance(grp_display, str) and grp_display.upper() == 'ANUBIS' and br_val is not None and float(br_val) < 1.0):
                legend_handles.append(Line2D([0], [0], color=color, lw=2.4, linestyle=linestyle, label=legend_label))
            else:
                # Ensure inline labels exist for ANUBIS BR<1 (they are drawn above when cs/cs4 exist)
                pass
        except Exception:
            try:
                legend_handles.append(Line2D([0], [0], color=color, lw=2.4, linestyle=linestyle, label=legend_label))
            except Exception:
                pass

    # draw legend (no unified colorbar — contours convey the 4-event boundary per BR)
    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc='lower left', title='Datasets (BR)', framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
        legend.set_zorder(100)

    # ---------------------------------------------------------
    # Overlay experimental contour polylines (LHC, MATHUSLA, ANUBIS)
    # - BR==1: solid, thicker; BR<1: dotted
    # ---------------------------------------------------------
    try:
        exp_handles = _collect_experiment_contour_handles(ax, base_dir=os.path.dirname(__file__))
        if exp_handles:
            # sanitize experimental handle labels to avoid raw filenames leaking
            try:
                for h in exp_handles:
                    try:
                        lab = h.get_label()
                        if isinstance(lab, str) and 'higgs_signal_events_data' in lab.lower():
                            if '(' in lab:
                                suffix = lab[lab.find('('):]
                                h.set_label('ANUBIS ' + suffix)
                            else:
                                h.set_label('ANUBIS')
                    except Exception:
                        pass
            except Exception:
                pass

            # merge with any existing legend
            if 'legend' in locals():
                all_handles = legend_handles + exp_handles
                legend = ax.legend(handles=all_handles, loc='lower left', title='Datasets / Limits', framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
            else:
                legend = ax.legend(handles=exp_handles, loc='lower left', title='Limits', framealpha=0.9, edgecolor='grey', prop={'size':14}, title_fontsize=14)
            legend.set_zorder(100)
    except Exception:
        pass

    ax.set_xscale('log')
    ax.set_yscale('log')
    # Force y-axis range as requested
    try:
        ax.set_ylim(1e-7, 100.0)
    except Exception:
        pass
    ax.set_xlabel('Effective $C_{Zh}$', fontsize=16)
    ax.set_ylabel(r'Coupling $C_{a\phi}$', fontsize=16)
    if title is None:
        title = 'Sensitivity Contours (overlaid BR scans)'
    ax.set_title(title, fontsize=18)
    # Increase tick label sizes and enable minor ticks for log axes
    try:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.minorticks_on()
    except Exception:
        pass
    # Show major and minor grid lines on both axes (including y log-grid)
    ax.grid(which='major', axis='both', alpha=0.45, linestyle='--', linewidth=0.6)
    ax.grid(which='minor', axis='both', alpha=0.25, linestyle=':', linewidth=0.4)
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
    default_csv = os.path.join(os.path.dirname(__file__), 'Plots', 'higgs_signal_events_data.csv')
    # Simplified non-interactive defaults (no CLI arguments)
    csv_path = default_csv
    output_path = os.path.join(os.path.dirname(__file__), 'Plots', 'sensitivity_contours_with_limits.png')
    # Contour levels to draw by default
    levels = [4.0]
    # Gaussian smoothing sigma (log-grid units)
    sigma = 0.0
    # Use log scales by default
    use_log = True
    # Expand potential glob patterns or detect per-BR CSVs in the Plots folder
    csv_list = []
    # If user provided a glob pattern, expand it
    if '*' in csv_path or '?' in csv_path:
        csv_list = sorted(glob.glob(csv_path))
    elif os.path.isdir(csv_path):
        csv_list = sorted(glob.glob(os.path.join(csv_path, 'higgs_signal_events_data_mumu_BR_*.csv')))
    else:
        # If using default csv and per-BR mumu CSVs exist, prefer those for overlay
        if os.path.abspath(csv_path) == os.path.abspath(default_csv):
            pb_pattern = os.path.join(os.path.dirname(__file__), 'Plots', 'higgs_signal_events_data_mumu_BR_*.csv')
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

    # No explicit label positions are embedded — labels are added in post-production.

    if len(csv_list) > 1:
        print(f'Found multiple CSVs: {csv_list} -> overlaying contours on same plot')
        # Attempt to read coupling metadata from the first CSV for title context
        czh_text = ''
        try:
            df_full = pd.read_csv(csv_list[0])
            if 'C_Zh_eff' in df_full.columns:
                unique_vals = pd.unique(df_full['C_Zh_eff'])
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
            pass

        title_base = 'Expected Signal Events for Fermion-Coupled ALPs, $pp\\to H \\to Z a$\n'
        title = title_base + czh_text + ' — includes MATHUSLA & LHC limits'
        # Draw only the 4-event overlay level for comparison
        levels_overlay = [4.0]
        plot_sensitivity_contours_overlay(csv_list, levels_overlay, output_path, title=title,
                                          use_log_scale=use_log, smooth_sigma=sigma)
    else:
        mass_vals, caphi_vals, heat = prepare_grid_from_csv(csv_list[0], 'N_signal')
        print(f'Loaded {len(mass_vals)} mass points × {len(caphi_vals)} coupling points from {csv_list[0]}')
        print(f'Plotting contours at levels: {levels} -> saving to {output_path}')

        # Try to extract the Higgs coupling `C_Zh_eff` from the CSV and append to title
        czh_text = ''
        try:
            df_full = pd.read_csv(csv_list[0])
            if 'C_Zh_eff' in df_full.columns:
                unique_vals = pd.unique(df_full['C_Zh_eff'])
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

        title_base = 'Expected Signal Events for Fermion-Coupled ALPs, $pp\\to H \\to Z a$'
        title = title_base + czh_text + ' — includes MATHUSLA & LHC limits'

        plot_sensitivity_contours(mass_vals, caphi_vals, heat, levels, output_path,
              title=title,
              use_log_scale=use_log, smooth_sigma=sigma)


if __name__ == '__main__':
    main()