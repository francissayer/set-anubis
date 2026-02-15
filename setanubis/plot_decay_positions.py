import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from SetAnubis.core.Selection.domain.SelectionPipeline import SelectionPipelineBuilder
from SetAnubis.core.Selection.domain.SelectionManager import SelectionManager
from SetAnubis.core.Selection.domain.DatasetSource import EventsBundleSource
from SetAnubis.core.Selection.domain.SelectionEngine import SelectionConfig, RunConfig, MinThresholds, MinDR
from SetAnubis.core.Selection.adapters.input.SelectionGeometryAdapter import SelectionGeometryAdapter
from SetAnubis.core.Geometry.adapters.selection_adapter import GeometrySelectionAdapter
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern


def build_selection(sel_mode: str = "standard"):
    cav = ATLASCavern()
    geom_adapter = GeometrySelectionAdapter(cav)
    sel_geo = SelectionGeometryAdapter(geom_adapter)

    sel_cfg = SelectionConfig(
        geometry=sel_geo,
        minMET=30.0,
        minP=MinThresholds(LLP=0.1, chargedTrack=0.1, neutralTrack=0.1, jet=0.1),
        minPt=MinThresholds(LLP=0.0, chargedTrack=5.0, neutralTrack=5.0, jet=15.0),
        minDR=MinDR(jet=0.4, chargedTrack=0.4, neutralTrack=0.4),
        nStations=2, nIntersections=2, nTracks=1,
    )
    run_cfg = RunConfig(reweightLifetime=False, plotTrajectory=False)

    pipeline = (
        SelectionPipelineBuilder()
        .set_options(add_jets=True, compute_isolation=True, selection_mode=sel_mode)
        .build()
    )

    return pipeline, sel_cfg, run_cfg


def extract_decay_positions(df):
    """Extract x, y, z decay positions from dataframe."""
    x_vals, y_vals, z_vals = [], [], []
    
    if df is None or df.empty:
        return x_vals, y_vals, z_vals
    
    # Try different column name variations
    vertex_cols = [col for col in df.columns if 'decay' in col.lower() and 'vertex' in col.lower()]
    
    if not vertex_cols:
        print(f"Available columns: {list(df.columns)}")
        return x_vals, y_vals, z_vals
    
    vertex_col = vertex_cols[0]
    print(f"Using column: {vertex_col}")
    
    for idx, row in df.iterrows():
        try:
            vertex = row[vertex_col]
            if isinstance(vertex, (list, tuple)) and len(vertex) >= 3:
                # Vertex is (x, y, z, t) or (x, y, z) in mm, convert to meters
                x_vals.append(vertex[0] / 1000.0)
                y_vals.append(vertex[1] / 1000.0)
                z_vals.append(vertex[2] / 1000.0)
        except Exception as e:
            continue
    
    return x_vals, y_vals, z_vals


def analyze_decay_positions(cavern, x_vals, y_vals, z_vals):
    """Analyze decay positions relative to detector geometry."""
    r_3d = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_vals, y_vals, z_vals)]
    r_transverse = [np.sqrt(x**2 + y**2) for x, y in zip(x_vals, y_vals)]
    
    # Count decays in different regions
    stats = {
        'total': len(x_vals),
        'r_3d': r_3d,
        'r_transverse': r_transverse,
        'in_atlas': sum(1 for r_t in r_transverse if r_t <= cavern.radiusATLAS),
        'in_cavern': sum(1 for x, y, z in zip(x_vals, y_vals, z_vals)
                        if (cavern.CavernX[0] <= x <= cavern.CavernX[1] and
                            cavern.CavernY[0] <= y <= cavern.CavernY[1] and
                            cavern.CavernZ[0] <= z <= cavern.CavernZ[1])),
        'between_25_100m': sum(1 for r in r_3d if 25 <= r <= 100),
    }
    
    if hasattr(cavern, 'radiusATLAStracking'):
        stats['in_tracking'] = sum(1 for r_t in r_transverse if r_t <= cavern.radiusATLAStracking)
    
    return stats


def plot_decay_positions_with_cavern(cavern, df, sample_name, out_dir):
    """Plot decay positions overlaid with ATLAS cavern geometry."""
    x_vals, y_vals, z_vals = extract_decay_positions(df)
    
    if not x_vals:
        print(f"No decay positions found for {sample_name}")
        return
    
    print(f"Found {len(x_vals)} decay positions")
    
    # Analyze positions
    stats = analyze_decay_positions(cavern, x_vals, y_vals, z_vals)
    x_arr, y_arr, z_arr = np.array(x_vals), np.array(y_vals), np.array(z_vals)
    r_arr = np.array(stats['r_3d'])
    
    # XY view (top-down) with acceptance
    fig, ax = plt.subplots(figsize=(12, 10))
    cavern.plotCavernXY(ax, plotATLAS=True, plotAcceptance=True)
    scatter = ax.scatter(x_vals, y_vals, c=r_arr, cmap='viridis', s=20, alpha=0.7, 
                        edgecolors='darkred', linewidths=0.5, label='Decay positions')
    cbar = plt.colorbar(scatter, ax=ax, label='Radial Distance from IP [m]')
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'Decay Positions - XY View: {sample_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'decay_positions_XY_{sample_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: decay_positions_XY_{sample_name}.png")
    
    # XZ view (side view along beam)
    fig, ax = plt.subplots(figsize=(14, 8))
    cavern.plotCavernXZ(ax, plotATLAS=True)
    ax.scatter(x_vals, z_vals, c='red', s=20, alpha=0.7, label='Decay positions', edgecolors='darkred', linewidths=0.5)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title(f'Decay Positions - XZ View: {sample_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'decay_positions_XZ_{sample_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: decay_positions_XZ_{sample_name}.png")
    
    # ZY view (side view perpendicular to beam) with acceptance
    fig, ax = plt.subplots(figsize=(14, 8))
    cavern.plotCavernZY(ax, plotATLAS=True, plotAcceptance=True)
    scatter = ax.scatter(z_vals, y_vals, c=r_arr, cmap='viridis', s=20, alpha=0.7, 
                        edgecolors='darkred', linewidths=0.5, label='Decay positions')
    cbar = plt.colorbar(scatter, ax=ax, label='Radial Distance from IP [m]')
    ax.set_xlabel('Z [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'Decay Positions - ZY View: {sample_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'decay_positions_ZY_{sample_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: decay_positions_ZY_{sample_name}.png")
    
    # Cavern ceiling coordinates (unrolled view)
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        cavern.plotCavernCeilingCoords(ax)
        
        # Convert x positions to local ceiling coordinates
        local_x = []
        for x, y in zip(x_vals, y_vals):
            heights = y - cavern.centreOfCurvature["y"]
            if abs(heights) > 0.01:  # Avoid division by very small numbers
                local = cavern.archRadius * np.arctan2(x, heights)
                local_x.append(local)
            else:
                local_x.append(0.0)
        
        scatter = ax.scatter(local_x, z_vals, c=r_arr, cmap='viridis', s=20, alpha=0.7,
                           edgecolors='darkred', linewidths=0.5, label='Decay positions')
        cbar = plt.colorbar(scatter, ax=ax, label='Radial Distance from IP [m]')
        ax.set_xlabel('Local Ceiling Coordinate [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.set_title(f'Decay Positions - Unrolled Ceiling View: {sample_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-cavern.arcLength/2 - 2, cavern.arcLength/2 + 2)
        ax.set_ylim(-32, 32)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'decay_positions_ceiling_{sample_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: decay_positions_ceiling_{sample_name}.png")
    except Exception as e:
        print(f"Could not create ceiling coordinates plot: {e}")
    
    # Radial distribution histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3D radial distance
    ax1.hist(stats['r_3d'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(25, color='red', linestyle='--', linewidth=2, label='25m')
    ax1.axvline(100, color='red', linestyle='--', linewidth=2, label='100m')
    ax1.set_xlabel('3D Radial Distance from IP [m]', fontsize=12)
    ax1.set_ylabel('Number of Decays', fontsize=12)
    ax1.set_title('3D Radial Distance Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Transverse distance
    ax2.hist(stats['r_transverse'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(cavern.radiusATLAS, color='purple', linestyle='--', linewidth=2, 
               label=f'ATLAS radius ({cavern.radiusATLAS}m)')
    if hasattr(cavern, 'radiusATLAStracking'):
        ax2.axvline(cavern.radiusATLAStracking, color='orange', linestyle=':', linewidth=2,
                   label=f'Tracking radius ({cavern.radiusATLAStracking}m)')
    ax2.set_xlabel('Transverse Distance from IP [m]', fontsize=12)
    ax2.set_ylabel('Number of Decays', fontsize=12)
    ax2.set_title('Transverse (XY) Distance Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'decay_radial_dist_{sample_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: decay_radial_dist_{sample_name}.png")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Decay Position Analysis for {sample_name}")
    print(f"{'='*60}")
    print(f"  Total decay positions: {stats['total']}")
    print(f"  Decays between 25-100m from IP: {stats['between_25_100m']} "
          f"({100*stats['between_25_100m']/stats['total']:.1f}%)")
    print(f"  Decays within ATLAS detector (r_T < {cavern.radiusATLAS}m): {stats['in_atlas']} "
          f"({100*stats['in_atlas']/stats['total']:.1f}%)")
    if 'in_tracking' in stats:
        print(f"  Decays within tracking volume (r_T < {cavern.radiusATLAStracking}m): {stats['in_tracking']} "
              f"({100*stats['in_tracking']/stats['total']:.1f}%)")
    print(f"  Decays within cavern bounds: {stats['in_cavern']} "
          f"({100*stats['in_cavern']/stats['total']:.1f}%)")
    print(f"\n  Position ranges:")
    print(f"    X: [{min(x_vals):.2f}, {max(x_vals):.2f}] m")
    print(f"    Y: [{min(y_vals):.2f}, {max(y_vals):.2f}] m")
    print(f"    Z: [{min(z_vals):.2f}, {max(z_vals):.2f}] m")
    print(f"    R (3D): [{min(stats['r_3d']):.2f}, {max(stats['r_3d']):.2f}] m")
    print(f"    R_T (transverse): [{min(stats['r_transverse']):.2f}, {max(stats['r_transverse']):.2f}] m")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="Plot decay positions overlaid with ATLAS cavern geometry")
    ap.add_argument("--bundle", default="/usera/fs568/set-anubis/ALP_Z_Runs/ALP_Z_sampledfs_Scan_2_Run_3.pkl.gz", 
                    help="Path to gzipped bundle pickle")
    ap.add_argument("--sel-mode", default="standard", help="Selection mode (standard/2dv if available)")
    ap.add_argument("--name", default="sample", help="Sample name label for plots")
    ap.add_argument("--outdir", default="/usera/fs568/set-anubis/setanubis", 
                    help="Directory to save plots")
    ap.add_argument("--run-selection", action="store_true", 
                    help="Run full selection pipeline (otherwise use pre-selected data if available)")
    args = ap.parse_args()

    bundle_path = os.path.abspath(args.bundle)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    cav = ATLASCavern()

    if args.run_selection:
        print("Running selection pipeline...")
        pipeline, sel_cfg, run_cfg = build_selection(sel_mode=args.sel_mode)
        source = EventsBundleSource.from_bundle_file(bundle_path)
        
        mgr = SelectionManager(pipeline)
        combined = mgr.run_many(named_sources=[(args.name, source)], sel_cfg=sel_cfg, run_cfg=run_cfg)
        
        sample = combined.per_sample[0]
        df = sample.finalDF
        print(f"Selection complete. {len(df)} events passed.")
    else:
        print("Loading data from bundle...")
        source = EventsBundleSource.from_bundle_file(bundle_path)
        # Get raw data without running selection
        bundle = source.materialize()
        # Use LLPs dataframe which contains decay vertex information
        df = bundle.get("LLPs", bundle.get("llps", None))
        if df is None:
            raise ValueError("Could not find LLPs dataframe in bundle. Available keys: " + str(list(bundle.keys())))
        print(f"Loaded {len(df)} LLP events from bundle.")

    # Plot decay positions
    plot_decay_positions_with_cavern(cav, df, args.name, out_dir)
    
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
