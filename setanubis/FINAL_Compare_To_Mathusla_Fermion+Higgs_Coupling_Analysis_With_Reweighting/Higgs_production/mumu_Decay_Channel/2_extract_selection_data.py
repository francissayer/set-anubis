#!/usr/bin/env python3
r"""extract_selection_data
=================================

Utilities to run the SetAnubis selection pipeline on EventsBundle sample
files and write per-run selection summaries to CSV for the ``mumu`` decay
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
The script writes a CSV (default: ``selection_cutflow_mumu_decay_channel.csv``)
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
from SetAnubis.core.Selection.adapters.input.ATLASCavernSelectionGeometryAdapter import ATLASCavernSelectionGeometryAdapter
from SetAnubis.core.Geometry.adapters.ATLASCavernGeometry import ATLASCavernGeometry, ATLASCavernGeometryConfig, GeometryRegion
import math

# ------------------
# Configuration (edit these defaults)
# ------------------
# Default directories and filenames
DEFAULT_BUNDLE_DIR = "/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel"
DEFAULT_OUTDIR = "/usera/fs568/set-anubis/setanubis/FINAL_Compare_To_Mathusla_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/mumu_Decay_Channel"
DEFAULT_OUTPUT = "selection_cutflow_mumu_decay_channel.csv"

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
DECAY_CHANNEL_INDEX = 2

# Default CaPhi coupling scan used when --target-couplings is not provided
DEFAULT_TARGET_COUPLINGS = [0.00631,0.00501,0.00398,0.00316,0.00251,0.002,0.00158,0.00126,0.001,0.000794,0.000631,0.000501,0.000398,0.000316,0.000251,0.0002,0.000158,0.000126,0.0001,0.0000794,0.0000631,0.0000501,0.0000398,0.0000316,0.0000251,0.00002,0.0000158,0.0000126,0.00001,0.00000794,0.00000631,0.00000501,0.00000398,0.00000316,0.00000251,0.000002,0.00000158,0.00000126,0.000001]

# hbar in GeV*s used for lifetime conversion
HBAR_GEV_S = 6.582119569e-25
# Default branching ratio(s) for ALP -> mu+ mu- used when inferring
# the total width from the muon partial width. This is a configuration
# tuple — include multiple values to scan over different BR assumptions.
DEFAULT_BR_MU_TUPLE = (1.0,0.1,0.01)


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
    # Create an ATLASCavernGeometry instance (v2-compatible) using explicit
    # fields to mirror the example in compare_old_geo_with_new.py
    base_cfg = ATLASCavernGeometryConfig(
        mode="ceiling",
        origin="IP",
        rpc_eff=1.0,
        n_rpcs_per_layer=1,
        use_cache=False,
        cache_file="atlas_cavern.pkl",
    )
    geometry = ATLASCavernGeometry.create(base_cfg)

    # Configure legacy-style simple RPC radii similar to historical behavior
    legacy_cfg = ATLASCavernGeometryConfig(
        mode=base_cfg.mode,
        origin=base_cfg.origin,
        rpc_eff=base_cfg.rpc_eff,
        n_rpcs_per_layer=base_cfg.n_rpcs_per_layer,
        use_cache=base_cfg.use_cache,
        cache_file=base_cfg.cache_file,
        simple_rpc_radii=(
            geometry._cavern.archRadius - 0.2,
            geometry._cavern.archRadius - 1.2,
        ),                                                                  # Paul's team decided that the ANUBIS geometry we use for the sensitivity studies does not include the central RPC layer.
                                                                            # The reasoning behind this is that the central singlet is useful for tracking and vertex reconstruction. However, practically if we build the detector
                                                                            # and want to access the interior of it having ~1m clearance to move around in would be far better than ~0.5m
        simple_rpc_thickness=0.06,
        rpc_max_radius=geometry._cavern.archRadius - 1.2 - 0.5,
    )

    geometry.reconfigure(legacy_cfg)

    sel_geo = ATLASCavernSelectionGeometryAdapter(geometry, default_decay_region=GeometryRegion.FIDUCIAL)

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





def compute_lifetime_from_coupling(coupling: float, mass_GeV: Optional[float] = None, BR_mu: float = 1.0) -> Optional[float]:
    """Compute ALP lifetime (s) from the muon partial width and muon BR.

    The function computes the partial width to muons using a simple
    formulaic expression and infers the total width as
    Gamma_total = Gamma(mu+mu-) / BR_mu. The lifetime is tau = hbar / Gamma_total.
    """
    try:
        if mass_GeV is None:
            print("  ERROR: mass_GeV must be provided from the bundle to compute lifetime.")
            return None
        if BR_mu is None or BR_mu <= 0.0:
            print(f"  ERROR: invalid BR_mu={BR_mu}")
            return None

        # Constants
        vgev = 246.22056907348585  # Higgs vev in GeV from ALP_linear_UFO_WIDTH model parameters
        Ca = float(coupling)
        fa = float(DEFAULT_FA_GEV)
        M_a = float(mass_GeV)

        # SM muon mass (GeV) taken from the ALP_linear_UFO_WIDTH model parameters
        mu = 0.10566

        # Kinematic threshold
        if M_a <= 2.0 * mu:
            print(f"  Warning: mass {M_a} GeV below 2*m_mu; cannot decay to muons")
            return None

        # Yukawa (SM relation)
        y_mu = math.sqrt(2.0) * mu / vgev

        sqrt_term = math.sqrt(max(0.0, M_a * M_a - 4.0 * mu * mu))

        prefactor = (Ca * Ca) * (vgev * vgev) * (y_mu * y_mu)
        gamma_mu = prefactor * sqrt_term / (16.0 * math.pi * (fa * fa))

        total_width = gamma_mu / float(BR_mu)

        if total_width <= 0.0:
            print(f"  Warning: computed total formula width is zero for coupling={coupling}, mass={M_a}, BR_mu={BR_mu}")
            return None

        lifetime_s = HBAR_GEV_S / float(total_width)
        return lifetime_s
    except Exception as e:
        print(f"  Warning: failed to compute lifetime from coupling {coupling}, mass {mass_GeV}, BR_mu {BR_mu}: {e}")
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


# Create a global cache to avoid checking the same scan/run twice
_PARAM_CACHE = {}

def extract_mass_and_coupling(bundle_path: str, scan: int, run: int) -> tuple:
    """Determine ALP mass and CaPhi coupling for a given bundle file.
    Optimized to use memory cache first, then tiny txt files, and finally massive pickles.
    """
    import pickle
    import gzip
    
    # 1. FASTEST: Check memory cache
    cache_key = (scan, run)
    if cache_key in _PARAM_CACHE:
        return _PARAM_CACHE[cache_key]

    mass = None
    caphi = None
    
    # 2. FAST: Extract CaPhi and Mass from MadGraph scan_run file first (tiny plain text file)
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
                        
                        mass_col_idx = next((i for i, col in enumerate(header) if 'mass#9000005' in col), None)
                        caphi_col_idx = next((i for i, col in enumerate(header) if 'alppars#5' in col), None)
                        
                        # Find the row for this run (run_XX format)
                        run_name = f"run_{run:02d}"
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) > 0 and parts[0] == run_name:
                                if mass_col_idx is not None and mass_col_idx < len(parts):
                                    mass = float(parts[mass_col_idx])
                                
                                # Extract CaPhi
                                if caphi_col_idx is not None and caphi_col_idx < len(parts):
                                    caphi = float(parts[caphi_col_idx])
                                break
    except Exception as e:
        print(f"  Warning: Could not extract parameters from scan_run file: {e}")

    # 3. SLOWEST: Fallback to unzipping and unpickling the bundle if text file parsing failed
    if mass is None:
        try:
            with gzip.open(bundle_path, 'rb') as f:
                bundle = pickle.load(f)
                if 'LLPs' in bundle and len(bundle['LLPs']) > 0:
                    mass = float(bundle['LLPs']['mass'].iloc[0])
        except Exception as e:
            print(f"  Warning: Could not extract mass from bundle: {e}")

    # Save to cache so future files with this scan/run load instantly
    _PARAM_CACHE[cache_key] = (mass, caphi)
    
    return mass, caphi


def _norm_float_str(val):
    """Normalize numeric-like values to a stable string representation.

    Returns None for None/NaN, otherwise returns a compact consistent
    string using 12 significant digits to avoid formatting mismatches.
    """
    try:
        if val is None:
            return None
        f = float(val)
        return format(f, '.12g')
    except Exception:
        try:
            s = str(val)
            return s if s != '' else None
        except Exception:
            return None


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
        default=None,
        help="Random seed for the reweighter (if provided, overrides the computed per-bundle seed)."
    )
    ap.add_argument(
        "--BR_mu",
        type=float,
        default=None,
        help="If provided, only process this muon branching ratio (overrides DEFAULT_BR_MU_TUPLE)."
    )
    ap.add_argument(
        "--force-coup-pos",
        dest='force_coup_pos',
        type=int,
        default=None,
        help="If provided, use this 1-based coupling position when composing deterministic seed (useful for single-coupling jobs)."
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
    ap.add_argument(
        "--mass-filter",
        type=float,
        default=1.0,
        help="Only process bundles with ALP mass (GeV) equal to this value. Set <= 0 to disable filtering (default: 1.0)."
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
        # Build lookup sets for de-duplication. We normalise rows into three
        # lookup sets so older CSV formats (filename/filepath based) remain supported.
        processed_by_gen_run = set()
        processed_by_filepath = set()
        processed_by_filename = set()

        for _, row in existing_df.iterrows():
            gen_idx = int(row['generated_events_index']) if 'generated_events_index' in existing_df.columns and pd.notna(row['generated_events_index']) else None
            scan_val = int(row['scan']) if 'scan' in existing_df.columns and pd.notna(row['scan']) else None
            run_val = int(row['run']) if 'run' in existing_df.columns and pd.notna(row['run']) else None
            target_coup_norm = _norm_float_str(row['target_coupling']) if 'target_coupling' in existing_df.columns and pd.notna(row['target_coupling']) else None
            br_mu_norm = _norm_float_str(row['BR_mu']) if 'BR_mu' in existing_df.columns and pd.notna(row['BR_mu']) else None
            
            mass_val = float(row['mass']) if 'mass' in existing_df.columns and pd.notna(row['mass']) else None
            mass_val_str = _norm_float_str(mass_val)

            processed_by_gen_run.add((gen_idx, scan_val, run_val, target_coup_norm, br_mu_norm, mass_val_str))

            if 'filepath' in existing_df.columns and pd.notna(row.get('filepath')):
                processed_by_filepath.add((row['filepath'], target_coup_norm, br_mu_norm, mass_val_str))
            if 'filename' in existing_df.columns and pd.notna(row.get('filename')):
                processed_by_filename.add((row['filename'], target_coup_norm, br_mu_norm, mass_val_str))

        print(f"Found existing CSV with {len(existing_df)} entries")
        print(f"Will skip {len(processed_by_gen_run) + len(processed_by_filepath) + len(processed_by_filename)} already-processed entries (by various keys)")
    else:
        processed_by_gen_run = set()
        processed_by_filepath = set()
        processed_by_filename = set()

    # We will iterate per target coupling (or once by lifetime) below; do not filter
    # the bundle list here globally since deduplication is coupling-specific.
    bundle_files_to_process = bundle_files
    # Filter bundles by ALP mass if requested (default: 1.0 GeV)
    mass_filter = getattr(args, 'mass_filter', None)
    if mass_filter is not None and mass_filter > 0:
        print(f"Filtering bundles to mass = {mass_filter} GeV (this may read bundle files)...")
        filtered_bundles = []
        for bf in bundle_files_to_process:
            fname = Path(bf).name
            scan, run = extract_scan_run_from_filename(fname)
            mass, _ = extract_mass_and_coupling(bf, scan, run)
            if mass is None:
                print(f"  Skipping {fname}: could not determine mass")
                continue
            try:
                if math.isclose(float(mass), float(mass_filter), rel_tol=1e-6, abs_tol=1e-8):
                    filtered_bundles.append(bf)
            except Exception:
                # Fallback: try exact match
                if float(mass) == float(mass_filter):
                    filtered_bundles.append(bf)
        bundle_files_to_process = filtered_bundles
    
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
    # Branching ratio(s) to muons to iterate over (configuration tuple)
    if getattr(args, 'BR_mu', None) is not None:
        br_mu_list = (args.BR_mu,)
    else:
        br_mu_list = DEFAULT_BR_MU_TUPLE

    all_results = []
    for br_pos, br_mu in enumerate(br_mu_list, start=1):
        for coup_pos, target_coup in enumerate(target_couplings, start=1):
            # Build per-(coupling,BR) list of bundles to process (deduplicated)
            per_coupling_to_process = []
            for bf in bundle_files_to_process:
                fname = Path(bf).name
                scan, run = extract_scan_run_from_filename(fname)
                m = re.search(r'Generated_Events_(\d+)', str(bf))
                gen_idx = int(m.group(1)) if m else None
                
                mass, _ = extract_mass_and_coupling(bf, scan, run)
                mass_val_str = _norm_float_str(mass)

                # If no existing processed entries, accept all
                if not processed_by_gen_run and not processed_by_filepath and not processed_by_filename:
                    per_coupling_to_process.append(bf)
                    continue

                # If no existing processed entries, accept all
                if not processed_by_gen_run and not processed_by_filepath and not processed_by_filename:
                    per_coupling_to_process.append(bf)
                    continue

                target_coup_norm = _norm_float_str(target_coup) if target_coup is not None else None
                br_mu_norm = _norm_float_str(br_mu) if br_mu is not None else None

                key_primary = (gen_idx, scan, run, target_coup_norm, br_mu_norm, mass_val_str)
                if key_primary in processed_by_gen_run:
                    continue

                key_fp = (bf, target_coup_norm, br_mu_norm, mass_val_str)
                if key_fp in processed_by_filepath:
                    continue

                key_fn = (fname, target_coup_norm, br_mu_norm, mass_val_str)
                if key_fn in processed_by_filename:
                    continue

                per_coupling_to_process.append(bf)

            if len(per_coupling_to_process) == 0:
                print(f"\nAll files already processed for coupling={target_coup}, BR_mu={br_mu}!\n")
                continue

            print(f"Processing {len(per_coupling_to_process)} bundles for coupling={target_coup} (pos={coup_pos}), BR_mu={br_mu} (pos={br_pos})...")
            for i, bundle_path in enumerate(per_coupling_to_process, 1):
                print(f"[{i}/{len(per_coupling_to_process)}] Processing: {Path(bundle_path).name}")

                # Extract ALP mass from the bundle and compute lifetime for that mass
                fname = Path(bundle_path).name
                scan, run = extract_scan_run_from_filename(fname)
                mass, _ = extract_mass_and_coupling(bundle_path, scan, run)

                lifetime = compute_lifetime_from_coupling(target_coup, mass_GeV=mass, BR_mu=br_mu)
                if lifetime is None:
                    print(f"  ERROR: could not compute lifetime for coupling={target_coup} mass={mass} BR_mu={br_mu}. Skipping this bundle.")
                    continue

                # Build selection pipeline with this bundle-specific lifetime
                # Compose a simple integer seed from generation index, process
                # index, run index, coupling position, BR position and base seed so different
                # couplings/BRs produce different seeds.
                m = re.search(r'Generated_Events_(\d+)', str(bundle_path))
                gen_idx = int(m.group(1)) if m else 0
                run_idx = int(run) if run is not None else 0

                # Allow overriding of the coupling position or the full seed
                if getattr(args, 'reweight_seed', None) is not None:
                    seed = int(args.reweight_seed) % 2147483647
                else:
                    coup_pos_used = int(args.force_coup_pos) if getattr(args, 'force_coup_pos', None) is not None else int(coup_pos)
                    seed = (
                        gen_idx * 1_000_000
                        + PROCESS_INDEX * 100_000
                        + DECAY_CHANNEL_INDEX * 10_000
                        + run_idx * 100
                        + coup_pos_used
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
                # annotate which target coupling and BR produced these results
                result['target_coupling'] = target_coup
                result['reweight_lifetime'] = lifetime
                # Record CaPhi as the target coupling used for the reweight
                result['CaPhi'] = target_coup
                # Record the muon branching ratio used for the inferred lifetime
                result['BR_mu'] = br_mu
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