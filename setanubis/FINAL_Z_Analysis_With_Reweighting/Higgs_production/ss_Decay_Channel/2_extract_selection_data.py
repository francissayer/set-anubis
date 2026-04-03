#!/usr/bin/env python3
r"""extract_selection_data
=================================

Utilities to run the SetAnubis selection pipeline on EventsBundle sample
files and write per-run selection summaries to CSV for the ``ss`` decay
channel.

This module performs the following, in order:

- Discover MadGraph-generated EventsBundle files (pickled sample bundles)
    in a configured directory (or from explicit file arguments).
- For each bundle: determine the ALP mass from the bundle (or from the
    MadGraph ``scan_run`` metadata), compute an ALP lifetime from a
    target coupling ``C_{a\Phi}`` using simple analytic fermionic
    partial-width formulae, reweight the events to that lifetime, and
    execute the selection pipeline.
- Aggregate per-run cutflow counters and the number of surviving LLPs
    into an output CSV.

Outputs
-------
The script writes a CSV (default: ``selection_cutflow_ss_decay_channel.csv``)
with one row per processed bundle. Each row contains metadata (filename,
scan/run, mass, CaPhi), per-cut counters from the pipeline, the
``n_surviving_llps`` integer and a ``status``/``error`` field on failure.

Notes
-----
The script uses formula-based widths (no external UFO evaluators by
default) to keep execution deterministic and suitable for offline
processing.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from SetAnubis.core.Selection.domain.SelectionPipeline import SelectionPipelineBuilder
from SetAnubis.core.Selection.domain.SelectionManager import SelectionManager
from SetAnubis.core.Selection.domain.DatasetSource import EventsBundleSource
from SetAnubis.core.Selection.domain.SelectionEngine import SelectionConfig, RunConfig, MinThresholds, MinDR
from SetAnubis.core.Selection.adapters.input.SelectionGeometryAdapter import SelectionGeometryAdapter
from SetAnubis.core.Geometry.adapters.selection_adapter import GeometrySelectionAdapter
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern
import math

# ------------------
# Configuration (edit these defaults)
# ------------------
# Default directories and filenames
DEFAULT_BUNDLE_DIR = "/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/ss_Decay_Channel"
DEFAULT_OUTDIR = "/usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting/Higgs_production/ss_Decay_Channel"
DEFAULT_OUTPUT = "selection_cutflow_ss_decay_channel.csv"

# Selection / file matching defaults
DEFAULT_PATTERN = "*.pkl.gz"
DEFAULT_SEL_MODE = "standard"

# Reweighting defaults
DEFAULT_REWEIGHT_LIFETIME = 1.0e-10
DEFAULT_REWEIGHT_LLP_PID = 9000005
DEFAULT_REWEIGHT_SEED = 42

# Decay constant f_a in GeV (user provided)
DEFAULT_FA_GEV = 1000.0

# MadGraph-style deterministic seed composition constants
# These mirror the approach used in the MadGraph driver to produce
# reproducible per-job seeds.
PROCESS_INDEX = 1
DECAY_CHANNEL_INDEX = 6

# Default CaPhi coupling scan used when --target-couplings is not provided
DEFAULT_TARGET_COUPLINGS = [1.0,0.316,0.1,0.0316,0.01,0.00316,0.001,0.000316,0.0001,0.0000316,0.00001,0.00000316,0.000001,0.000000316,0.0000001,0.0000000316]

# hbar in GeV*s used for lifetime conversion
HBAR_GEV_S = 6.582119569e-25


def build_selection(sel_mode: str = "standard", lifetime_s: Optional[float] = None, llp_pid: Optional[int] = None, seed: Optional[int] = None):
    """Build and configure a SelectionPipeline for processing bundles.

    Parameters
    ----------
    sel_mode : str, optional
        Selection mode passed to the pipeline builder (default ``"standard"``).
    lifetime_s : float or None, optional
        Target LLP lifetime in seconds to use for the reweighter. If ``None``
        a sensible default is applied.
    llp_pid : int or None, optional
        PDG id of the LLP to which reweighting is applied (default: 9000005).
    seed : int or None, optional
        Integer RNG seed for the reweighter to ensure reproducible runs.

    Returns
    -------
    pipeline, sel_cfg, run_cfg
        The constructed pipeline object and its associated selection and run
        configuration objects ready to be passed to the selection manager.
    """
    cav = ATLASCavern()
    geom_adapter = GeometrySelectionAdapter(cav)
    sel_geo = SelectionGeometryAdapter(geom_adapter)
    
    # Create RPCs for ANUBIS station intersection
    # cav.createSimpleRPCs(
    #     [cav.archRadius-0.2, cav.archRadius-0.6, cav.archRadius-1.2], 
    #     RPCthickness=0.06
    # )
    cav.createSimpleRPCs([cav.archRadius-0.2, cav.archRadius-1.2], RPCthickness=0.06)                       # Paul's team decided that the ANUBIS geometry we use for the sensitivity studies does not include the central RPC layer.
                                                                                                            # The reasoning behind this is that the central singlet is useful for tracking and vertex reconstruction. However, practically if we build the detector
                                                                                                            # and want to access the interior of it having ~1m clearance to move around in would be far better than ~0.5m

    sel_cfg = SelectionConfig(
        geometry=sel_geo,
        minMET=30.0,
        minP=MinThresholds(LLP=0.1, chargedTrack=0.1, neutralTrack=0.1, jet=0.1),
        minPt=MinThresholds(LLP=0.0, chargedTrack=5.0, neutralTrack=5.0, jet=15.0),
        minDR=MinDR(jet=0.5, chargedTrack=0.5, neutralTrack=0.5),
        nStations=2, nIntersections=2, nTracks=2,
    )
    
    # Reweighting is enabled for these sensitivity runs.
    run_cfg = RunConfig(reweightLifetime=True, plotTrajectory=False)

    builder = (
        SelectionPipelineBuilder()
        .set_options(add_jets=True, compute_isolation=True, selection_mode=sel_mode)
    )

    # Configure the pipeline reweighter with the provided target lifetime
    # Use sensible defaults if values not provided
    if lifetime_s is None:
        lifetime_s = 1.0e-10
    if llp_pid is None:
        llp_pid = 9000005
    if seed is None:
        seed = 42

    try:
        builder = builder.set_reweighter(lifetime_s=lifetime_s, llp_pid=llp_pid, seed=seed)
    except Exception as e:
        print(f"  Warning: failed to set reweighter on pipeline builder: {e}")

    pipeline = builder.build()

    return pipeline, sel_cfg, run_cfg





def compute_lifetime_from_coupling(coupling: float, mass_GeV: Optional[float] = None) -> Optional[float]:
    """Compute an approximate ALP lifetime from analytic fermionic widths.

    The routine evaluates the tree-level fermionic partial widths using the
    simple Yukawa-suppressed formulae used by the ALP model and sums the
    open channels to obtain a total width::

        Gamma_total = sum_f Gamma(a -> f fbar)
        tau = hbar / Gamma_total

    Parameters
    ----------
    coupling : float
        The effective `C_{a\Phi}` coupling used in the width expressions.
    mass_GeV : float or None
        ALP mass in GeV. If ``None`` the function returns ``None``.

    Returns
    -------
    float or None
        Lifetime in seconds, or ``None`` if the width cannot be computed
        (e.g. below kinematic thresholds or on error).

    Notes
    -----
    This is a deterministic, offline-friendly approximation and intentionally
    does not call external UFO evaluators. It includes the light quark and
    charged-lepton channels with simple color factors.
    """
    try:
        if mass_GeV is None:
            print("  ERROR: mass_GeV must be provided from the bundle to compute lifetime.")
            return None

        # Constants
        vgev = 246.22056907348585  # Higgs vev in GeV from ALP_linear_UFO_WIDTH model parameters
        Ca = float(coupling)
        fa = float(DEFAULT_FA_GEV)
        M_a = float(mass_GeV)

        # SM fermion masses (GeV) taken from the ALP_linear_UFO_WIDTH model parameters
        sm_masses = {
            'd': 0.00504,
            'u': 0.00255,
            's': 0.101,
            'c': 1.27,
            'b': 4.7,
            't': 172,
            'e': 0.000511,
            'mu': 0.10566,
            'tau': 1.777,
        }

        # Channels to include in total width (name, mass, color_factor)
        channels = [
            ('b', sm_masses['b'], 3),
            ('c', sm_masses['c'], 3),
            ('d', sm_masses['d'], 3),
            ('u', sm_masses['u'], 3),
            ('s', sm_masses['s'], 3),
            ('t', sm_masses['t'], 3),
            ('e', sm_masses['e'], 1),
            ('mu', sm_masses['mu'], 1),
            ('tau', sm_masses['tau'], 1),
        ]

        total_width = 0.0
        for pname, m_f, color in channels:
            # Kinematic threshold
            if M_a <= 2.0 * m_f:
                continue

            # Yukawa (SM relation)
            y_f = math.sqrt(2.0) * m_f / vgev

            sqrt_term = math.sqrt(max(0.0, M_a * M_a - 4.0 * m_f * m_f))

            prefactor = ( (3.0 if color == 3 else 1.0) * (Ca * Ca) * (vgev * vgev) * (y_f * y_f) )
            gamma_f = prefactor * sqrt_term / (16.0 * math.pi * (fa * fa))
            total_width += gamma_f

        if total_width <= 0.0:
            print(f"  Warning: computed total formula width is zero for coupling={coupling}, mass={M_a}")
            return None

        lifetime_s = HBAR_GEV_S / float(total_width)
        return lifetime_s
    except Exception as e:
        print(f"  Warning: failed to compute lifetime from coupling {coupling}, mass {mass_GeV}: {e}")
        return None


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
    """Determine ALP mass and CaPhi coupling for a given bundle file.

    Strategy (best-effort):
    1. Try to read the pickled EventsBundle and obtain the ALP mass from the
       ``LLPs`` table if present (most reliable).
    2. If not found, locate the corresponding MadGraph ``scan_run_*.txt`` in
       the scan's ``Events`` directory and parse the table header to extract
       the mass and the alppars column that maps to ``CaPhi``.

    Parameters
    ----------
    bundle_path : str
        Path to the pickled bundle file (gzip compressed pickle).
    scan : int
        Scan index inferred from the bundle filename (used to find MadGraph
        scan directories).
    run : int
        Run index within the scan (used to match the scan_run table row).

    Returns
    -------
    (mass_GeV, CaPhi)
        Tuple containing the ALP mass (float) and the CaPhi coupling (float),
        either or both may be ``None`` if not discoverable.
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
        scan_dir = Path(bundle_path).parent / f"Higgs_to_ALP_axZ_scan_{scan}"
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


def process_bundle_file(bundle_path: str, pipeline, sel_cfg, run_cfg, reweight_lifetime: Optional[float] = None) -> Dict:
    """Process one EventsBundle: run selection and collect cutflow results.

    Steps performed:
    - Determine scan/run, mass and CaPhi for the bundle.
    - Wrap the bundle into an ``EventsBundleSource`` and pass it to the
        ``SelectionManager`` to execute the pipeline with the supplied
        ``sel_cfg`` and ``run_cfg``.
    - Collect per-cut counters from ``combined.cutflow_sum`` and the final
        surviving LLP count from the pipeline output.

    The returned dictionary always contains metadata keys (``filename``,
    ``scan``, ``run``, ``mass``, ``CaPhi``) and either ``status: 'success'``
    with cutflow keys, or ``status: 'failed'`` and an ``error`` message.
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
    # record the target reweight lifetime used for this run
    result['reweight_lifetime'] = reweight_lifetime
    # Extract Generated_Events index if present in the path (e.g. Generated_Events_1)
    gen_idx = None
    m = re.search(r'Generated_Events_(\d+)', str(bundle_path))
    if m:
        try:
            gen_idx = int(m.group(1))
        except Exception:
            gen_idx = None
    result['generated_events_index'] = gen_idx
    
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

        # We store the selection results (cutflow and surviving counts)
        # in the main CSV only. Do not write separate reweighted CSV files.
        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ERROR: {e}")

    return result


def find_bundle_files(directory: str, pattern: str = "Higgs_to_ALP_Z_sampledfs_*.pkl.gz") -> List[str]:
    """Recursively find EventsBundle files matching ``pattern`` under ``directory``.

    If the provided ``directory`` points to a ``Generated_Events_N`` subfolder
    the function searches the parent directory so that sibling
    ``Generated_Events_*`` folders are included. This makes the discovery
    robust when a user points to a single generated-events folder.

    Parameters
    ----------
    directory : str
        Root directory to search under.
    pattern : str
        Glob pattern to match bundle filenames (default
        ``Higgs_to_ALP_Z_sampledfs_*.pkl.gz``).

    Returns
    -------
    list[str]
        Sorted list of absolute file paths that matched the pattern.
    """
    bundle_files = []
    path = Path(directory)
    # If user pointed to a Generated_Events_N folder, search its parent so we include siblings
    if 'Generated_Events_' in path.name or 'Generated_Events_' in str(path):
        path = path.parent

    # Search recursively so we pick up files in all Generated_Events_* subfolders
    for file in path.rglob(pattern):
        bundle_files.append(str(file))
    # Debug: number found
    # (caller prints counts too, but a quick message here helps diagnose missing files)
    print(f"Searching bundles under: {path} -> found {len(bundle_files)} files matching '{pattern}'")
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
        default=DEFAULT_BUNDLE_DIR,
        help=f"Directory containing Generated_Events_* folders (default: {DEFAULT_BUNDLE_DIR})"
    )
    ap.add_argument(
        "--bundle-files",
        nargs="+",
        help="Specific bundle files to process (overrides --bundle-dir)"
    )
    ap.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"File pattern to match (default: {DEFAULT_PATTERN})"
    )
    ap.add_argument(
        "--sel-mode",
        default=DEFAULT_SEL_MODE,
        help=f"Selection mode (default: {DEFAULT_SEL_MODE})"
    )
    # Lifetime reweighting is enabled by default.
    ap.add_argument(
        "--reweight-lifetime",
        type=float,
        default=DEFAULT_REWEIGHT_LIFETIME,
        help=f"Target lifetime in seconds for reweighting (default: {DEFAULT_REWEIGHT_LIFETIME})."
    )
    ap.add_argument(
        "--target-couplings",
        nargs="+",
        type=float,
        help="List of target CaPhi coupling values to compute selection cuts for (space-separated)."
    )
    ap.add_argument(
        "--reweight-llp-pid",
        type=int,
        default=DEFAULT_REWEIGHT_LLP_PID,
        help=f"PDG id of the LLP to reweight (default: {DEFAULT_REWEIGHT_LLP_PID})."
    )
    ap.add_argument(
        "--reweight-seed",
        type=int,
        default=DEFAULT_REWEIGHT_SEED,
        help=f"Random seed for the reweighter (default: {DEFAULT_REWEIGHT_SEED})."
    )
    # No UFO coupling parameter required; lifetimes computed from bundle mass
    ap.add_argument(
        "--reweighted-outdir",
        default=None,
        help="Directory where detected reweighted event CSVs will be saved (default: <outdir>/reweighted_events)."
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV filename (default: {DEFAULT_OUTPUT})"
    )
    ap.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})"
    )
    
    args, unknown = ap.parse_known_args()
    
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
    processed_by_filepath = False

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # Prefer deduplication using generated_events_index + scan + run + target coupling
        if {'generated_events_index', 'scan', 'run', 'target_coupling'}.issubset(existing_df.columns):
            already_processed = set(
                (
                    int(row['generated_events_index']) if pd.notna(row['generated_events_index']) else None,
                    int(row['scan']) if pd.notna(row['scan']) else None,
                    int(row['run']) if pd.notna(row['run']) else None,
                    str(row['target_coupling']) if pd.notna(row['target_coupling']) else None,
                )
                for _, row in existing_df.iterrows()
            )
            processed_by_filepath = False
        # Fallback: if historical CSV only has reweight_lifetime, keep previous behavior
        elif {'generated_events_index', 'scan', 'run', 'reweight_lifetime'}.issubset(existing_df.columns):
            already_processed = set(
                (
                    int(row['generated_events_index']) if pd.notna(row['generated_events_index']) else None,
                    int(row['scan']) if pd.notna(row['scan']) else None,
                    int(row['run']) if pd.notna(row['run']) else None,
                    float(row['reweight_lifetime']) if pd.notna(row['reweight_lifetime']) else None,
                )
                for _, row in existing_df.iterrows()
            )
            processed_by_filepath = False
        # Fallback: use filepath + target_coupling if present
        elif {'filepath', 'target_coupling'}.issubset(existing_df.columns):
            already_processed = set(
                (row['filepath'], str(row['target_coupling']) if pd.notna(row['target_coupling']) else None)
                for _, row in existing_df.iterrows()
            )
            processed_by_filepath = True
        # Fallback: use filename + target_coupling if present
        elif {'filename', 'target_coupling'}.issubset(existing_df.columns):
            already_processed = set(
                (row['filename'], str(row['target_coupling']) if pd.notna(row['target_coupling']) else None)
                for _, row in existing_df.iterrows()
            )
            processed_by_filepath = False
        else:
            # Existing CSV doesn't include coupling/lifetime info — assume no prior
            # reweight computations to avoid false positives.
            already_processed = set()
            processed_by_filepath = False

        print(f"Found existing CSV with {len(existing_df)} entries")
        print(f"Will skip {len(already_processed)} already-processed (scan/run/gen_idx/lifetime) entries")

    # We will iterate per target coupling (or once by lifetime) below; do not filter
    # the bundle list here globally since deduplication is coupling-specific.
    bundle_files_to_process = bundle_files
    
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
    
    # Determine reweighted outdir default
    if args.reweighted_outdir is None:
        args.reweighted_outdir = os.path.join(args.outdir, 'reweighted_events')

    # Determine which target couplings to run. If none provided, fall back to
    # using the default coupling scan defined in the script.
    target_couplings = args.target_couplings if args.target_couplings else DEFAULT_TARGET_COUPLINGS

    all_results = []
    for coup_pos, target_coup in enumerate(target_couplings, start=1):
        # Build per-coupling list of bundles to process (deduplicated)
        per_coupling_to_process = []
        for bf in bundle_files_to_process:
            fname = Path(bf).name
            scan, run = extract_scan_run_from_filename(fname)
            m = re.search(r'Generated_Events_(\d+)', str(bf))
            gen_idx = int(m.group(1)) if m else None

            if len(already_processed) == 0:
                per_coupling_to_process.append(bf)
                continue

            key_primary = (gen_idx, scan, run, str(target_coup) if target_coup is not None else None)
            if key_primary in already_processed:
                continue

            key_fp = (bf, str(target_coup) if target_coup is not None else None)
            if key_fp in already_processed:
                continue

            key_fn = (fname, str(target_coup) if target_coup is not None else None)
            if key_fn in already_processed:
                continue

            per_coupling_to_process.append(bf)

        if len(per_coupling_to_process) == 0:
            print(f"\nAll files already processed for coupling={target_coup}!\n")
            continue

        print(f"Processing {len(per_coupling_to_process)} bundles for coupling={target_coup} (pos={coup_pos})...")
        for i, bundle_path in enumerate(per_coupling_to_process, 1):
            print(f"[{i}/{len(per_coupling_to_process)}] Processing: {Path(bundle_path).name}")

            # Extract ALP mass from the bundle and compute lifetime for that mass
            fname = Path(bundle_path).name
            scan, run = extract_scan_run_from_filename(fname)
            mass, _ = extract_mass_and_coupling(bundle_path, scan, run)

            lifetime = compute_lifetime_from_coupling(target_coup, mass_GeV=mass)
            if lifetime is None:
                print(f"  ERROR: could not compute lifetime for coupling={target_coup} mass={mass}. Skipping this bundle.")
                continue

            # Build selection pipeline with this bundle-specific lifetime
            # Compose a simple integer seed from generation index, process
            # index, run index, coupling position, and base seed so different
            # couplings produce different seeds.
            m = re.search(r'Generated_Events_(\d+)', str(bundle_path))
            gen_idx = int(m.group(1)) if m else 0
            run_idx = int(run) if run is not None else 0

            seed = (
                gen_idx * 1_000_000
                + PROCESS_INDEX * 100_000
                + DECAY_CHANNEL_INDEX * 10_000
                + run_idx * 100
                + int(coup_pos)
            ) % 2147483647
            if seed == 0:
                seed = 1

            pipeline, sel_cfg, run_cfg = build_selection(
                sel_mode=args.sel_mode,
                lifetime_s=lifetime,
                llp_pid=args.reweight_llp_pid,
                seed=seed,
            )

            result = process_bundle_file(bundle_path, pipeline, sel_cfg, run_cfg, reweight_lifetime=lifetime)
            # annotate which target coupling produced these results
            result['target_coupling'] = target_coup
            result['reweight_lifetime'] = lifetime
            # Record CaPhi as the target coupling used for the reweight
            result['CaPhi'] = target_coup
            all_results.append(result)

            if result['status'] == 'success':
                print(f"  ✓ Scan: {result['scan']}, Run: {result['run']}, Surviving LLPs: {result['n_surviving_llps']}")
            else:
                print(f"  ✗ Failed")
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Count successes and failures (handle empty results or missing 'status')
    if 'status' in df.columns:
        n_success = int((df['status'] == 'success').sum())
        n_failed = int((df['status'] == 'failed').sum())
    else:
        n_success = 0
        n_failed = 0
        print("Warning: no 'status' column in results (no bundles processed?)")
    
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
