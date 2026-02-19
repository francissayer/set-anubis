#!/usr/bin/env python3
"""
Script to extract selection cut data from bundle files and save to CSV.

This script processes sample dataframes, runs the selection pipeline on each,
and extracts all cutflow information along with the number of surviving LLPs.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

from SetAnubis.core.Selection.domain.SelectionPipeline import SelectionPipelineBuilder
from SetAnubis.core.Selection.domain.SelectionManager import SelectionManager
from SetAnubis.core.Selection.domain.DatasetSource import EventsBundleSource
from SetAnubis.core.Selection.domain.SelectionEngine import SelectionConfig, RunConfig, MinThresholds, MinDR
from SetAnubis.core.Selection.adapters.input.SelectionGeometryAdapter import SelectionGeometryAdapter
from SetAnubis.core.Geometry.adapters.selection_adapter import GeometrySelectionAdapter
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern


def build_selection(sel_mode: str = "standard"):
    """Build the selection pipeline and configuration."""
    cav = ATLASCavern()
    geom_adapter = GeometrySelectionAdapter(cav)
    sel_geo = SelectionGeometryAdapter(geom_adapter)
    
    # Create RPCs for ANUBIS station intersection
    cav.createSimpleRPCs(
        [cav.archRadius-0.2, cav.archRadius-0.6, cav.archRadius-1.2], 
        RPCthickness=0.06
    )

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


def extract_scan_run_from_filename(filename: str) -> tuple:
    """
    Extract scan number and run number from filename.
    
    Example: 'ALP_Z_sampledfs_Scan_3_Run_4.pkl.gz' -> (3, 4)
    """
    match = re.search(r'Scan_(\d+)_Run_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_mass_and_coupling(bundle_path: str, scan: int, run: int) -> tuple:
    """
    Extract mass and CaPhi coupling from bundle file and MadGraph scan_run file.
    
    Returns:
        tuple: (mass, CaPhi) or (None, None) if not found
    """
    import pickle
    import gzip
    
    mass = None
    caphi = None
    
    # Extract mass from bundle pickle file (most reliable source)
    try:
        with gzip.open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
            if 'LLPs' in bundle and len(bundle['LLPs']) > 0:
                mass = bundle['LLPs']['mass'].iloc[0]
    except Exception as e:
        print(f"  Warning: Could not extract mass from bundle: {e}")
    
    # Extract CaPhi from MadGraph scan_run file (contains actual parameter values used)
    try:
        scan_dir = Path(bundle_path).parent / f"ALP_axZ_scan_{scan}"
        if scan_dir.exists():
            # Look for scan_run file which contains the actual parameter values for each run
            events_dir = scan_dir / "Events"
            
            if events_dir.exists():
                # Find any scan_run file
                scan_run_files = list(events_dir.glob("scan_run_*.txt"))
                
                if scan_run_files:
                    # Parse the scan_run file (it's a table with headers)
                    with open(scan_run_files[0], 'r') as f:
                        lines = f.readlines()
                        
                    if len(lines) >= 2:
                        # First line is header
                        header = lines[0].split()
                        
                        # Find column indices
                        mass_col_idx = None
                        caphi_col_idx = None
                        
                        for i, col in enumerate(header):
                            if 'mass#9000005' in col:
                                mass_col_idx = i
                            if 'alppars#5' in col:  # CaPhi is typically alppars#5
                                caphi_col_idx = i
                        
                        # Find the row for this run (run_XX format)
                        run_name = f"run_{run:02d}"
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) > 0 and parts[0] == run_name:
                                # Extract mass if not already found in bundle
                                if mass is None and mass_col_idx is not None and mass_col_idx < len(parts):
                                    mass = float(parts[mass_col_idx])
                                
                                # Extract CaPhi
                                if caphi_col_idx is not None and caphi_col_idx < len(parts):
                                    caphi = float(parts[caphi_col_idx])
                                break
    except Exception as e:
        print(f"  Warning: Could not extract parameters from scan_run file: {e}")
    
    return mass, caphi


def process_bundle_file(bundle_path: str, pipeline, sel_cfg, run_cfg) -> Dict:
    """
    Run selection pipeline on a single bundle file and extract all cut data.
    
    Returns:
        Dict containing file info, cutflow data, and number of surviving LLPs
    """
    filename = Path(bundle_path).name
    scan, run = extract_scan_run_from_filename(filename)
    
    # Extract mass and coupling
    mass, caphi = extract_mass_and_coupling(bundle_path, scan, run)
    
    result = {
        'filename': filename,
        'filepath': bundle_path,
        'scan': scan,
        'run': run,
        'mass': mass,
        'CaPhi': caphi,
    }
    
    try:
        source = EventsBundleSource.from_bundle_file(bundle_path)
        mgr = SelectionManager(pipeline)
        combined = mgr.run_many(
            named_sources=[("sample", source)], 
            sel_cfg=sel_cfg, 
            run_cfg=run_cfg
        )
        
        # Extract cutflow information
        cutflow = combined.cutflow_sum
        for cut_name, count in cutflow.items():
            result[cut_name] = count
        
        # Get number of surviving LLPs from final dataframe
        if len(combined.per_sample) > 0:
            final_df = combined.per_sample[0].finalDF
            result['n_surviving_llps'] = len(final_df)
        else:
            result['n_surviving_llps'] = 0
            
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ERROR: {e}")
    
    return result


def find_bundle_files(directory: str, pattern: str = "ALP_Z_sampledfs_*.pkl.gz") -> List[str]:
    """Find all bundle files matching the pattern in the directory."""
    bundle_files = []
    path = Path(directory)
    for file in path.glob(pattern):
        bundle_files.append(str(file))
    return sorted(bundle_files)


def main():
    ap = argparse.ArgumentParser(
        description="Extract selection cut data from bundle files to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all bundles in default directory
  python extract_selection_data.py
  
  # Specify different directory
  python extract_selection_data.py --bundle-dir /path/to/ALP_Z_Runs
  
  # Process specific files
  python extract_selection_data.py --bundle-files file1.pkl.gz file2.pkl.gz
  
  # Specify output file
  python extract_selection_data.py --output my_selection_data.csv
        """
    )
    
    ap.add_argument(
        "--bundle-dir", 
        default="/usera/fs568/set-anubis/ALP_Z_Runs",
        help="Directory containing bundle files (default: /usera/fs568/set-anubis/ALP_Z_Runs)"
    )
    ap.add_argument(
        "--bundle-files",
        nargs="+",
        help="Specific bundle files to process (overrides --bundle-dir)"
    )
    ap.add_argument(
        "--pattern",
        default="ALP_Z_sampledfs_*.pkl.gz",
        help="File pattern to match (default: ALP_Z_sampledfs_*.pkl.gz)"
    )
    ap.add_argument(
        "--sel-mode",
        default="standard",
        help="Selection mode (default: standard)"
    )
    ap.add_argument(
        "--output",
        default="selection_cutflow_data.csv",
        help="Output CSV filename (default: selection_cutflow_data.csv)"
    )
    ap.add_argument(
        "--outdir",
        default="/usera/fs568/set-anubis/setanubis",
        help="Output directory (default: /usera/fs568/set-anubis/setanubis)"
    )
    
    args = ap.parse_args()
    
    # Get list of bundle files to process
    if args.bundle_files:
        bundle_files = [os.path.abspath(f) for f in args.bundle_files]
    else:
        bundle_files = find_bundle_files(args.bundle_dir, args.pattern)
    
    if not bundle_files:
        print("No bundle files found!")
        return
    
    # Check for existing CSV and skip already-processed files
    output_path = os.path.join(args.outdir, args.output)
    already_processed = set()
    
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        already_processed = set(existing_df['filename'].values)
        print(f"Found existing CSV with {len(existing_df)} entries")
        print(f"Will skip {len(already_processed)} already-processed files")
    
    # Filter out already-processed files
    bundle_files_to_process = [
        bf for bf in bundle_files 
        if Path(bf).name not in already_processed
    ]
    
    print("="*70)
    print(f"SELECTION DATA EXTRACTION")
    print("="*70)
    print(f"Found {len(bundle_files)} bundle files")
    print(f"Already processed: {len(already_processed)}")
    print(f"To process: {len(bundle_files_to_process)}")
    print(f"Output directory: {args.outdir}")
    print(f"Output file: {args.output}")
    print("="*70)
    
    if len(bundle_files_to_process) == 0:
        print("\nAll files already processed!")
        return
    
    # Build selection pipeline once
    print("\nBuilding selection pipeline...")
    pipeline, sel_cfg, run_cfg = build_selection(sel_mode=args.sel_mode)
    print("Pipeline built successfully!\n")
    
    # Process all files
    results = []
    for i, bundle_path in enumerate(bundle_files_to_process, 1):
        print(f"[{i}/{len(bundle_files_to_process)}] Processing: {Path(bundle_path).name}")
        result = process_bundle_file(bundle_path, pipeline, sel_cfg, run_cfg)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"  ✓ Scan: {result['scan']}, Run: {result['run']}, Surviving LLPs: {result['n_surviving_llps']}")
        else:
            print(f"  ✗ Failed")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Count successes and failures
    n_success = (df['status'] == 'success').sum()
    n_failed = (df['status'] == 'failed').sum()
    
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(df)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    
    if n_success > 0:
        successful_df = df[df['status'] == 'success']
        print(f"\nTotal surviving LLPs (all successful runs): {successful_df['n_surviving_llps'].sum()}")
        print(f"Mean surviving LLPs per run: {successful_df['n_surviving_llps'].mean():.2f}")
        print(f"Std: {successful_df['n_surviving_llps'].std():.2f}")
        print(f"Min: {successful_df['n_surviving_llps'].min()}")
        print(f"Max: {successful_df['n_surviving_llps'].max()}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save to CSV (append mode if file exists)
    if os.path.exists(output_path):
        # Load existing data
        existing_df = pd.read_csv(output_path)
        
        # Append new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Appended {len(df)} new rows")
        print(f"  Total rows in CSV: {len(combined_df)}")
    else:
        # Create new file
        df.to_csv(output_path, index=False)
        print(f"\n✓ Created new CSV with {len(df)} rows")
    
    print("\n" + "="*70)
    print(f"✓ Data saved to: {output_path}")
    print("="*70)
    
    # Show column names
    print("\nColumns in output CSV:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Show a preview of the data
    print("\nPreview of first few rows:")
    print(df.head().to_string())
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
