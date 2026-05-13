"""
Compare MATHUSLA40 converted-data contours with simulated MATHUSLA40 contours.

This small helper imports the existing `11_sensitivity_contours_with_mathusla_data.py`
helper functions and calls `plot_sensitivity_contours_overlay` on the simulated
MATHUSLA40 grid files. The converted experimental polylines (files named
`MATHUSLA40_BR_*.csv` or `MATHUSLA40.csv`) are discovered automatically by the
plotting module and overlaid as limits.

Usage:
	python 10_compare_MATHUSLA40_converted_data_and_simulated_data.py \
		--simulated "Plots/Simulated_MATHUSLA40_higgs_signal_events_data_mumu_BR_*.csv" \
		--output Plots/compare_MATHUSLA40_converted_vs_simulated.png
"""

from __future__ import annotations

import os
import glob
import argparse
import importlib.util
from pathlib import Path
import sys
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


def _load_plot_module():
	base = os.path.dirname(__file__)
	mod_path = os.path.join(base, '11_sensitivity_contours_with_mathusla_data.py')
	if not os.path.exists(mod_path):
		raise FileNotFoundError(f'Required module not found: {mod_path}')
	spec = importlib.util.spec_from_file_location('s10', mod_path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def main():
	parser = argparse.ArgumentParser(description='Overlay Simulated MATHUSLA40 contours over converted-data polylines')
	default_sim = os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA40_higgs_signal_events_data.csv')
	parser.add_argument('--simulated', type=str, default=default_sim,
						help='Path, glob, or folder for simulated MATHUSLA40 CSV(s)')
	default_out = os.path.join(os.path.dirname(__file__), 'Plots', 'compare_MATHUSLA40_converted_vs_simulated.png')
	parser.add_argument('--output', type=str, default=default_out, help='Output PNG path')
	parser.add_argument('--levels', type=float, nargs='+', default=[4.0], help='Contour levels to draw')
	parser.add_argument('--smooth_sigma', type=float, default=20.0, help='Gaussian smoothing sigma (log-grid units)')
	parser.add_argument('--draw_heatmap', action='store_true', help='Draw combined heatmap under contours')
	parser.add_argument('--keep-temp', action='store_true', help='Keep temporary sanitized files')

	args = parser.parse_args()

	# Expand simulated CSV list
	sim_list = []
	sim_arg = args.simulated
	if '*' in sim_arg or '?' in sim_arg:
		sim_list = sorted(glob.glob(sim_arg))
	elif os.path.isdir(sim_arg):
		sim_list = sorted(glob.glob(os.path.join(sim_arg, 'Simulated_MATHUSLA40_higgs_signal_events_data_mumu_BR_*.csv')))
	else:
		# If using default, prefer per-BR files in Plots/ if present
		if os.path.abspath(sim_arg) == os.path.abspath(default_sim):
			pb_pattern = os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA40_higgs_signal_events_data_mumu_BR_*.csv')
			pb_files = sorted(glob.glob(pb_pattern))
			if pb_files:
				sim_list = pb_files
			elif os.path.exists(sim_arg):
				sim_list = [sim_arg]
			else:
				# fallback to any matching Simulated_MATHUSLA40 files
				sim_list = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'Plots', 'Simulated_MATHUSLA40*')))
		else:
			if os.path.exists(sim_arg):
				sim_list = [sim_arg]
			else:
				sim_list = sorted(glob.glob(sim_arg))

	sim_list = [p for p in sim_list if p and os.path.exists(p)]

	if not sim_list:
		raise FileNotFoundError(f'No simulated MATHUSLA40 CSV files found for pattern: {args.simulated}')

	plotmod = _load_plot_module()

	out_dir = Path(args.output).parent
	out_dir.mkdir(parents=True, exist_ok=True)

	title = 'MATHUSLA40: Own simulated contours over converted MATHUSLA40 data limits'

	print('Simulated CSVs to plot (overlay):')
	for p in sim_list:
		print('  ', p)

	# --- New approach: draw converted experimental polylines first (LHC, MATHUSLA200, MATHUSLA40)
	fig, ax = plt.subplots(figsize=(8, 12))
	# patterns: (displayname, globpattern, color_idx)
	patterns = [
		('LHC', 'LHC_BR_*.csv', 0),
		('MATHUSLA200', 'MATHUSLA_BR_*.csv', 1),
		('MATHUSLA40', 'MATHUSLA40_BR_*.csv', 1),
	]

	# Use the plotting module helper to draw experimental polylines onto our axes
	try:
		exp_handles = plotmod._collect_experiment_contour_handles(ax, base_dir=os.path.dirname(__file__), patterns=patterns)
	except Exception:
		exp_handles = []

	# Rename any MATHUSLA40 experimental legend entries to indicate converted data
	try:
		for h in list(exp_handles):
			try:
				lab = h.get_label()
				if isinstance(lab, str) and 'mathusla40' in lab.lower():
					h.set_label('MATHUSLA40 (converted data)')
			except Exception:
				continue
	except Exception:
		pass

	# Helper: interpolate simulated grid CSV -> (GX, GY, GZ) using LinearNDInterpolator
	def _compute_interpolated_grid(csv_path, nx_grid=800, ny_grid=800, smooth_sigma=None):
		mass_vals, caphi_vals, heat = plotmod.prepare_grid_from_csv(csv_path, 'N_signal')
		x = np.asarray(mass_vals, dtype=float)
		y = np.asarray(caphi_vals, dtype=float)
		Z2 = np.array(heat, dtype=float)

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

		xlog = np.log10(x)
		ylog = np.log10(y)
		XX, YY = np.meshgrid(xlog, ylog)
		pts = np.column_stack((XX.ravel(), YY.ravel()))
		vals = zlog.ravel()

		finite_mask = np.isfinite(vals)
		pts_f = pts[finite_mask]
		vals_f = vals[finite_mask]

		if pts_f.shape[0] < 3:
			return None, None, None

		if plotmod.LinearNDInterpolator is None:
			raise RuntimeError('scipy.interpolate.LinearNDInterpolator is required; please install scipy.')

		interp = plotmod.LinearNDInterpolator(pts_f, vals_f)

		LOGX = np.linspace(xlog.min(), xlog.max(), nx_grid)
		LOGY = np.linspace(ylog.min(), ylog.max(), ny_grid)
		LOGGX, LOGGY = np.meshgrid(LOGX, LOGY)
		eval_pts = np.column_stack((LOGGX.ravel(), LOGGY.ravel()))
		GZ_log = interp(eval_pts).reshape(LOGGX.shape)

		# Mask outside convex hull
		try:
			if plotmod.ConvexHull is not None and pts_f.shape[0] >= 3:
				hull = plotmod.ConvexHull(pts_f)
				hull_path = plotmod.MplPath(pts_f[hull.vertices])
				inside = hull_path.contains_points(eval_pts)
				GZ_log_flat = GZ_log.ravel()
				GZ_log_flat[~inside] = np.nan
				GZ_log = GZ_log_flat.reshape(GZ_log.shape)
		except Exception:
			pass

		# Optional smoothing in log-space
		if plotmod.gaussian_filter is not None and smooth_sigma is not None and smooth_sigma > 0:
			try:
				med = np.nanmedian(GZ_log)
				if np.isfinite(med):
					filled = np.where(np.isfinite(GZ_log), GZ_log, med)
				else:
					filled = np.where(np.isfinite(GZ_log), GZ_log, 0.0)
				smoothed = plotmod.gaussian_filter(filled, sigma=smooth_sigma, mode='nearest')
				smoothed[~np.isfinite(GZ_log)] = np.nan
				GZ_log = smoothed
			except Exception:
				pass

		GZ = np.where(np.isfinite(GZ_log), 10.0 ** GZ_log, np.nan)
		GX, GY = np.meshgrid(10.0 ** LOGX, 10.0 ** LOGY)

		# Clip smoothed grid to original data range
		if smooth_sigma is not None and smooth_sigma > 0 and np.any(positive_mask):
			try:
				GZ = np.clip(GZ, eps, max_pos)
			except Exception:
				pass

		return GX, GY, GZ

	# Draw simulated MATHUSLA40 contours on top of experimental polylines
	# Use a single legend entry for all simulated files (grouped)
	sim_color = 'purple'
	sim_any = False
	# store per-sim interpolated grids so we can compute BR envelopes / fills
	sim_gz_items = []
	for idx, sim_path in enumerate(sim_list):
		try:
			GX, GY, GZ = _compute_interpolated_grid(sim_path, nx_grid=800, ny_grid=800, smooth_sigma=args.smooth_sigma)
		except Exception as e:
			print(f'Warning: failed to interpolate {sim_path}: {e}')
			continue
		if GX is None:
			continue

		# choose color and linestyle (use unified sim color for visibility)
		color = sim_color
		br_val = None
		try:
			br_val = plotmod._extract_br_value(sim_path, df=None)
		except Exception:
			br_val = None

		# store the interpolated grid for later BR-envelope / fill computation
		try:
			sim_gz_items.append({'path': sim_path, 'GX': GX, 'GY': GY, 'GZ': GZ, 'br_val': br_val, 'idx': idx})
		except Exception:
			# non-fatal; continue without envelope capability for this file
			pass
		# mark that we've processed at least one simulated dataset (defer drawing until after BR-envelope)
		sim_any = True

	# Compose legend: experimental handles first, then a single simulated group handle
	all_handles = []
	try:
		all_handles.extend(exp_handles)
	except Exception:
		pass

	# --- Set x/y limits based on union of simulated grids (match behaviour in 11_sensitivity_contours_with_mathusla_data.py)
	try:
		mass_lists = []
		caphi_lists = []
		for p in sim_list:
			try:
				mv, cv, _ = plotmod.prepare_grid_from_csv(p, 'N_signal')
				mass_lists.append(np.asarray(mv, dtype=float))
				caphi_lists.append(np.asarray(cv, dtype=float))
			except Exception:
				continue
		if mass_lists:
			union_mass = np.sort(np.unique(np.concatenate(mass_lists)))
			try:
				if union_mass.size > 1 and np.all(union_mass > 0):
					gx = np.sqrt(union_mass[:-1] * union_mass[1:])
					x_edges = np.concatenate(([union_mass[0] ** 2 / gx[0]], gx, [union_mass[-1] ** 2 / gx[-1]]))
					ax.set_xlim(x_edges[0], x_edges[-1])
				else:
					ax.set_xlim(float(union_mass.min()) * 0.9, float(union_mass.max()) * 1.1)
			except Exception:
				pass
		# Y-limits: attempt to mimic the overlay behaviour (but keep default enforced later)
		if caphi_lists:
			union_caphi = np.sort(np.unique(np.concatenate(caphi_lists)))
			try:
				if union_caphi.size > 1 and np.all(union_caphi > 0):
					gy = np.sqrt(union_caphi[:-1] * union_caphi[1:])
					y_edges = np.concatenate(([union_caphi[0] ** 2 / gy[0]], gy, [union_caphi[-1] ** 2 / gy[-1]]))
					ax.set_ylim(y_edges[0], y_edges[-1])
				else:
					ax.set_ylim(float(union_caphi.min()) * 0.9, float(union_caphi.max()) * 1.1)
			except Exception:
				pass
	except Exception:
		pass
	if sim_any:
		try:
			sim_group_handle = Line2D([0], [0], color=sim_color, lw=2.8, linestyle='-', label='MATHUSLA40 (simulated)')
			all_handles.append(sim_group_handle)
		except Exception:
			pass

	# --- Apply BR-envelope like 7.5 and draw per-BR contours, then unified fills ---
	group_masks = {}
	group_colors = {}
	group_zords = {}
	try:
		if sim_gz_items:
			# Build processed items with metadata
			processed_items = []
			for it in sim_gz_items:
				grp = os.path.basename(it['path']).split('_BR_')[0]
				processed_items.append({'idx': it['idx'], 'path': it['path'], 'GX': it['GX'], 'GY': it['GY'], 'GZ': it['GZ'], 'grp': grp, 'br_val': it['br_val']})

			# Group and apply running envelope (ascending BR)
			groups = {}
			for item in processed_items:
				groups.setdefault(item['grp'], []).append(item)
			for gname, items in groups.items():
				items.sort(key=lambda x: float(x['br_val']) if x.get('br_val') is not None else 0.0)
				running_gz = None
				for item in items:
					if item['GZ'] is None:
						continue
					if running_gz is None:
						running_gz = item['GZ'].copy()
						item['GZ'] = running_gz.copy()
					else:
						running_gz = np.fmax(running_gz, item['GZ'])
						item['GZ'] = running_gz.copy()

			# Restore input order
			processed_items.sort(key=lambda x: x['idx'])

			# Draw contours for each processed item and accumulate 4-event masks
			for item in processed_items:
				GX = item['GX']; GY = item['GY']; GZ = item['GZ']
				if GZ is None:
					continue
				br_val = item['br_val']
				linestyle = ':' if (br_val is not None and not np.isclose(br_val, 1.0)) else '-'
				levels_non4 = [lv for lv in args.levels if not np.isclose(lv, 4.0)]
				has_4 = any(np.isclose(lv, 4.0) for lv in args.levels)

				try:
					if levels_non4:
						ax.contour(GX, GY, GZ, levels=levels_non4, colors=[sim_color], linewidths=1.8, linestyles=linestyle, zorder=300+item['idx'])
				except Exception as e:
					print(f"Warning: failed to draw contours for {item['path']}: {e}")

				if has_4:
					try:
						ax.contour(GX, GY, GZ, levels=[4.0], colors=[sim_color], linewidths=3.2, linestyles=linestyle, zorder=320+item['idx'])
						mask = np.isfinite(GZ) & (GZ >= 4.0)
						if np.any(mask):
							grp = item['grp']
							if grp not in group_masks:
								group_masks[grp] = mask
								group_colors[grp] = sim_color
								group_zords[grp] = 320 + item['idx'] - 5
							else:
								group_masks[grp] = group_masks[grp] | mask
					except Exception as e:
						print(f"Warning: failed to draw 4-event contour for {item['path']}: {e}")
	except Exception:
		pass

	# Draw unified fills for simulated groups (using group's first available grid)
	for grp_fill, mask_fill in group_masks.items():
		try:
			mask_float = np.where(mask_fill, 1.0, np.nan)
			# find a representative grid for this group
			try:
				grid_item = next(it for it in processed_items if it['grp'] == grp_fill and it['GZ'] is not None)
				GX_ref = grid_item['GX']; GY_ref = grid_item['GY']
			except StopIteration:
				GX_ref = GX; GY_ref = GY
			ax.contourf(GX_ref, GY_ref, mask_float, levels=[0.5, 1.5], colors=[group_colors.get(grp_fill, sim_color)], alpha=0.15, zorder=group_zords.get(grp_fill, 230))
		except Exception as e:
			print(f"Warning: failed to draw filled contour for group {grp_fill}: {e}")

	# finalize axes styling similar to other plots
	try:
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.axvspan(0.72, ax.get_xlim()[1], color='grey', alpha=0.3, zorder=5, label=r'$C_{Zh} > 0.72$')
		ax.set_ylim(1e-7, 100.0)
		ax.set_xlabel('Effective $C_{Zh}$', fontsize=16)
		ax.set_ylabel(r'Coupling $C_{a\Phi}$', fontsize=16)
		ax.set_title(title, fontsize=18)
		ax.tick_params(axis='both', which='major', labelsize=14)
		ax.tick_params(axis='both', which='minor', labelsize=12)
		ax.minorticks_on()
		ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=999))
		ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=999))
		ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=999))
		ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=999))
		ax.xaxis.set_minor_formatter(mticker.NullFormatter())
		ax.yaxis.set_minor_formatter(mticker.NullFormatter())
		ax.tick_params(axis='both', which='minor', length=3, width=0.6)
		ax.grid(False)
		# manual gridlines
		xlim = ax.get_xlim(); ylim = ax.get_ylim()
		for x in ax.get_xticks(minor=False):
			ax.axvline(x, color='grey', alpha=0.45, linestyle='--', linewidth=0.6, zorder=1.5)
		for x in ax.get_xticks(minor=True):
			ax.axvline(x, color='grey', alpha=0.25, linestyle=':', linewidth=0.4, zorder=1.5)
		for y in ax.get_yticks(minor=False):
			ax.axhline(y, color='grey', alpha=0.45, linestyle='--', linewidth=0.6, zorder=1.5)
		for y in ax.get_yticks(minor=True):
			ax.axhline(y, color='grey', alpha=0.25, linestyle=':', linewidth=0.4, zorder=1.5)
		ax.set_xlim(xlim); ax.set_ylim(ylim)
		ax.set_axisbelow(False)
		for spine in ax.spines.values():
			spine.set_zorder(10000)
		ax.xaxis.set_zorder(10000); ax.yaxis.set_zorder(10000)
		ax.tick_params(axis='both', which='both', zorder=10000)
		plt.tight_layout()
		# draw legend if handles exist
		if all_handles:
			legend = ax.legend(handles=all_handles, loc='lower left', title='Limits / Simulated', framealpha=0.9, edgecolor='grey', prop={'size':12}, title_fontsize=12)
			legend.set_zorder(100)
	except Exception:
		pass

	out_path = Path(args.output)
	os.makedirs(out_path.parent or '.', exist_ok=True)
	plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
	try:
		pdf_path = out_path.with_suffix('.pdf')
		plt.savefig(str(pdf_path), bbox_inches='tight')
	except Exception:
		pass
	plt.close()

	print('Saved overlay plot to', args.output)


if __name__ == '__main__':
	main()

