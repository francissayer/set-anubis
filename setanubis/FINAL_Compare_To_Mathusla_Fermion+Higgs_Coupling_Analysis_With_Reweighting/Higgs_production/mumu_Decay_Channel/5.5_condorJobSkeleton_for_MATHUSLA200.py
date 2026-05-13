import sys, os
from glob import glob
import argparse
import ast
import re
from pathlib import Path
import pandas as pd
import math
import pickle
import gzip


def _parse_int_like(val):
    """Parse integer-like values robustly from various CSV representations.

    Returns an int or None if value is missing/unparseable.
    Accepts ints, floats, and strings like '3', '3.0', ' 3 '.
    """
    try:
        if val is None:
            return None
        # Handle pandas NaN floats
        try:
            if isinstance(val, float) and math.isnan(val):
                return None
        except Exception:
            pass

        if isinstance(val, int):
            return int(val)
        if isinstance(val, float):
            return int(val)

        s = str(val).strip()
        if s == '':
            return None
        f = float(s)
        if math.isnan(f):
            return None
        return int(f)
    except Exception:
        return None

parser = argparse.ArgumentParser()
parser.add_argument("--jobscriptDir", type=str, default="")
parser.add_argument("--memory", default = "6G")
parser.add_argument("--dryrun", action="store_true", help="Run without submitting, but produces the jobscripts and submission files which can be checked.")
parser.add_argument("--testArg", action="store_true", help="Just a proxy argument for reference")
parser.add_argument("--mass-filter", type=float, default=1.0, help="Only process bundles with ALP mass (GeV) equal to this value. Set <= 0 to disable filtering.")
# Rebuild-missing should be the default when running interactively (e.g. VSCode play button).
parser.add_argument("--rebuild-missing", dest='rebuild_missing', action="store_true", help="Enumerate expected bundle x coupling x BR keys and prepare jobs for any missing entries in the detector CSV.")
parser.add_argument("--no-rebuild-missing", dest='rebuild_missing', action="store_false", help="Disable rebuilding missing entries (useful when running bulk non-rebuild flows).")
parser.set_defaults(rebuild_missing=True)
args = parser.parse_args()


# Base bash header executed in each job
bashHeader = """#!/bin/bash

cd /usera/fs568/set-anubis
source /usera/fs568/set-anubis/.venv/bin/activate
"""

# Path to the extract-selection script we want each job to run
EXTRACT_SCRIPT = "/usera/fs568/set-anubis/setanubis/FINAL_Compare_To_Mathusla_Fermion+Higgs_Coupling_Analysis_With_Reweighting/Higgs_production/mumu_Decay_Channel/5_extract_selection_data_for_MATHUSLA200.py"


def load_default_lists_from_extract(path):
    """Parse several DEFAULT_* constants from the extract script without executing it.

    Returns: (target_couplings, br_mu_tuple, bundle_dir, pattern, outdir, output)
    """
    with open(path, 'r') as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}")
        return [], (), None, None, None, None

    target_couplings = []
    br_tuple = ()
    bundle_dir = None
    pattern = None
    outdir = None
    output = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, 'id', None)
                if not name:
                    continue
                try:
                    if name == 'DEFAULT_TARGET_COUPLINGS':
                        target_couplings = ast.literal_eval(node.value)
                    elif name == 'DEFAULT_BR_MU_TUPLE':
                        br_tuple = ast.literal_eval(node.value)
                    elif name == 'DEFAULT_BUNDLE_DIR':
                        bundle_dir = ast.literal_eval(node.value)
                    elif name == 'DEFAULT_PATTERN':
                        pattern = ast.literal_eval(node.value)
                    elif name == 'DEFAULT_OUTDIR':
                        outdir = ast.literal_eval(node.value)
                    elif name == 'DEFAULT_OUTPUT':
                        output = ast.literal_eval(node.value)
                except Exception:
                    # best-effort parsing; ignore individual failures
                    pass

    if target_couplings is None:
        target_couplings = []
    if br_tuple is None:
        br_tuple = ()
    return list(target_couplings), tuple(br_tuple), bundle_dir, pattern, outdir, output

def extract_scan_run_from_filename(filename: str) -> tuple:
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
            events_dir = scan_dir / "Events"
            if events_dir.exists():
                scan_run_files = list(events_dir.glob("scan_run_*.txt"))
                
                if scan_run_files:
                    with open(scan_run_files[0], 'r') as f:
                        lines = f.readlines()
                        
                    if len(lines) >= 2:
                        header = lines[0].split()
                        
                        mass_col_idx = next((i for i, col in enumerate(header) if 'mass#9000005' in col), None)
                        caphi_col_idx = next((i for i, col in enumerate(header) if 'alppars#5' in col), None)
                        
                        run_name = f"run_{run:02d}"
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) > 0 and parts[0] == run_name:
                                if mass_col_idx is not None and mass_col_idx < len(parts):
                                    mass = float(parts[mass_col_idx])
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
    """Normalize a numeric-like value to a stable string representation.

    Returns None for None/NaN, otherwise returns a compact but consistent
    string using 12 significant digits (avoids minor repr/scientific differences).
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
# Load defaults from the extract script (couplings, BRs, bundle dir, patterns, outdir/output)
target_couplings, br_mu_tuple, bundle_dir, pattern, default_outdir, default_output = load_default_lists_from_extract(EXTRACT_SCRIPT)

if not target_couplings:
    print("No target couplings found in extract script; aborting.")
    sys.exit(1)

if not br_mu_tuple:
    print("No BR mu tuple found in extract script; aborting.")
    sys.exit(1)

print(f"Found {len(target_couplings)} default couplings and {len(br_mu_tuple)} BR values")

# Prepare jobscript directory
jobscriptDir = args.jobscriptDir if args.jobscriptDir else os.path.join('.', 'jobscripts')
if not os.path.exists(jobscriptDir):
    os.makedirs(jobscriptDir)

# Discover bundle files (mirror logic in extract script's find_bundle_files)
bundle_files = []
if bundle_dir is not None:
    path = Path(bundle_dir)
    if 'Generated_Events_' in path.name or 'Generated_Events_' in str(path):
        path = path.parent
    use_pattern = pattern if pattern is not None else "Higgs_to_ALP_Z_sampledfs_*.pkl.gz"
    for file in path.rglob(use_pattern):
        bundle_files.append(str(file))
    print(f"Discovered {len(bundle_files)} bundle files under: {path}")
else:
    print("Warning: no DEFAULT_BUNDLE_DIR found in extract script; cannot discover bundles automatically")

# Filter bundles by mass if requested (mirror logic in extract script)
if args.mass_filter > 0:
    print(f"Filtering bundles to mass = {args.mass_filter} GeV (this may read bundle files)...")
    filtered_bundles = []
    for bf in bundle_files:
        fname = Path(bf).name
        scan, run = extract_scan_run_from_filename(fname)
        mass, _ = extract_mass_and_coupling(bf, scan, run)
        if mass is None:
            continue
        try:
            if math.isclose(float(mass), float(args.mass_filter), rel_tol=1e-6, abs_tol=1e-8):
                filtered_bundles.append(bf)
        except Exception:
            if float(mass) == float(args.mass_filter):
                filtered_bundles.append(bf)
    bundle_files = filtered_bundles
    print(f"Remaining bundle files after mass filtering: {len(bundle_files)}")

# Read existing output CSV to determine already-processed (coupling,BR,bundle,mass) entries
processed_by_gen_run = set()
processed_by_filepath = set()
processed_by_filename = set()
output_csv = None
if default_outdir and default_output:
    # Look for detector-suffixed CSVs first (e.g. selection_cutflow_..._MATHUSLA200.csv)
    candidate = os.path.join(default_outdir, default_output)
    base = os.path.splitext(default_output)[0]
    pattern_glob = os.path.join(default_outdir, f"{base}*.csv")
    matches = glob(pattern_glob)
    output_csv = None
    if matches:
        # Prefer any match that mentions the detector name (case-insensitive)
        mathusla_matches = [m for m in matches if 'mathusla' in os.path.basename(m).lower()]
        if mathusla_matches:
            # If multiple, pick the newest
            output_csv = max(mathusla_matches, key=os.path.getmtime)
            print(f"Found detector-suffixed CSV: {output_csv}")
        else:
            # Pick the most recently modified matching CSV
            output_csv = max(matches, key=os.path.getmtime)
            print(f"Found existing CSV by pattern: {output_csv}")
    elif os.path.exists(candidate):
        output_csv = candidate
    else:
        output_csv = candidate

if output_csv and os.path.exists(output_csv):
    try:

        existing_df = pd.read_csv(output_csv)
        for _, row in existing_df.iterrows():
            gen_idx = _parse_int_like(row['generated_events_index']) if 'generated_events_index' in existing_df.columns and pd.notna(row['generated_events_index']) else None
            scan_val = _parse_int_like(row['scan']) if 'scan' in existing_df.columns and pd.notna(row['scan']) else None
            run_val = _parse_int_like(row['run']) if 'run' in existing_df.columns and pd.notna(row['run']) else None
            target_coup_norm = _norm_float_str(row['target_coupling']) if 'target_coupling' in existing_df.columns and pd.notna(row['target_coupling']) else None
            br_mu_norm = _norm_float_str(row['BR_mu']) if 'BR_mu' in existing_df.columns and pd.notna(row['BR_mu']) else None

            mass_val = float(row['mass']) if 'mass' in existing_df.columns and pd.notna(row['mass']) else None
            mass_val_str = _norm_float_str(mass_val)

            processed_by_gen_run.add((gen_idx, scan_val, run_val, target_coup_norm, br_mu_norm, mass_val_str))

            if 'filepath' in existing_df.columns and pd.notna(row.get('filepath')):
                processed_by_filepath.add((row['filepath'], target_coup_norm, br_mu_norm, mass_val_str))
            if 'filename' in existing_df.columns and pd.notna(row.get('filename')):
                processed_by_filename.add((row['filename'], target_coup_norm, br_mu_norm, mass_val_str))

        print(f"Found existing CSV with {len(existing_df)} entries; will skip already-processed bundles for matching coupling/BR/mass")
    except Exception as e:
        print(f"Warning: failed to read existing output CSV {output_csv}: {e}")
else:
    print("No existing output CSV found; will create jobs for all coupling/BR/mass combinations")

# Create a job for each (coupling, BR) pair that still has unprocessed bundles
job_counter = 0

if args.rebuild_missing:
    # Build expected keys from discovered bundle files x couplings x BRs
    expected_set = set()
    for bf in bundle_files:
        fname = Path(bf).name
        scan, run = extract_scan_run_from_filename(fname)
        mass, _ = extract_mass_and_coupling(bf, scan, run)
        mass_val_str = _norm_float_str(mass)
        m = re.search(r'Generated_Events_(\d+)', fname)
        gen_idx = int(m.group(1)) if m else None
        for t in target_couplings:
            for b in br_mu_tuple:
                expected_set.add((gen_idx, scan, run, _norm_float_str(t), _norm_float_str(b), mass_val_str))

    missing_set = expected_set - processed_by_gen_run
    print(f"Computed expected {len(expected_set)} keys; processed {len(processed_by_gen_run)} keys; missing {len(missing_set)} keys")

    if len(missing_set) == 0:
        print("No missing entries detected; nothing to schedule.")
    else:
        # Group missing keys by (coupling_norm, br_norm)
        missing_by_combo = {}
        for (gen_idx, scan, run, t_norm, b_norm, mass_str) in missing_set:
            missing_by_combo.setdefault((t_norm, b_norm), []).append((gen_idx, scan, run, mass_str))

        # Maps to get original float values for wrapper generation
        target_map = {_norm_float_str(t): t for t in target_couplings}
        br_map = {_norm_float_str(b): b for b in br_mu_tuple}

        for (t_norm, b_norm), items in sorted(missing_by_combo.items()):
            t_float = target_map.get(t_norm)
            b_float = br_map.get(b_norm)
            if t_float is None or b_float is None:
                print(f"Skipping unmapped combo: {t_norm}, {b_norm}")
                continue

            coup_pos = target_couplings.index(t_float) + 1
            br_pos = br_mu_tuple.index(b_float) + 1
            job_counter += 1
            jobDir = os.path.join(jobscriptDir, f"job{job_counter}_c{coup_pos}_b{br_pos}_M200")

            if not os.path.exists(jobDir):
                os.makedirs(jobDir)
            else:
                t = input(f"WARNING: A job with this ID ({job_counter}) already exists at {jobDir}, overwrite? [y/n]")
                if t.lower() == "y":
                    for entry in os.listdir(jobDir):
                        entry_path = os.path.join(jobDir, entry)
                        try:
                            if os.path.isdir(entry_path):
                                import shutil
                                shutil.rmtree(entry_path)
                            else:
                                os.remove(entry_path)
                        except Exception:
                            pass
                else:
                    print(f"Skipping this job ID: {job_counter}...")
                    continue

            wrapper_py = os.path.join(jobDir, "run_wrapper.py")
            wrapper_code = f"""import importlib.util
import sys, os

spec = importlib.util.spec_from_file_location("extract_mod", "{EXTRACT_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Override defaults so this run processes a single (coupling,BR) pair
mod.DEFAULT_TARGET_COUPLINGS = [{repr(t_float)}]
mod.DEFAULT_BR_MU_TUPLE = ({repr(b_float)},)
# Force args so seed computation uses the original coupling position
sys.argv = ["{EXTRACT_SCRIPT}", "--force-coup-pos", "{coup_pos}", "--BR_mu", "{b_float}", "--mass-filter", "{args.mass_filter}"]
# Call main
if hasattr(mod, 'main'):
    mod.main()
else:
    raise RuntimeError("extract module has no main()")
"""

            with open(wrapper_py, 'w') as wf:
                wf.write(wrapper_code)

            bashScriptName = os.path.join(jobDir, "runJob")
            with open(f"{bashScriptName}.sh","w") as f:
                f.write(bashHeader)
                f.write(f"/usera/fs568/set-anubis/.venv/bin/python {wrapper_py}\n")

            try:
                os.chmod(f"{bashScriptName}.sh", 0o755)
            except Exception:
                pass

            condorString = f"executable = {bashScriptName}.sh" + "\n"
            condorString+= f"output = {jobDir}/job{job_counter}_output.log" + "\n"
            condorString+= f"error =  {jobDir}/job{job_counter}_error.log" + "\n"
            condorString+= f"request_memory = {args.memory}" + "\n"
            condorString+= f"log =  {jobDir}/job{job_counter}_log.log" + "\n"
            condorString+= "copy_to_spool = true\n"
            condorString+= "should_transfer_files = YES\n"
            condorString+= "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
            condorString+= "Queue"

            condorSubmissionFile = os.path.join(jobDir, "condor_submit.job")
            with open(condorSubmissionFile, 'w') as c:
                c.write(condorString)

            print(f"Prepared job: coupling={t_float} (pos={coup_pos}), BR_mu={b_float} (pos={br_pos}) -> {condorSubmissionFile}")
            print(f"To submit: condor_submit {condorSubmissionFile}")
            if not args.dryrun:
                os.system(f"condor_submit {condorSubmissionFile}")

else:
    for br_pos, br_mu in enumerate(br_mu_tuple, start=1):
        for coup_pos, target_coup in enumerate(target_couplings, start=1):
            # Determine whether any bundle under bundle_files remains unprocessed for this (coupling,BR)
            per_coupling_to_process = []
            for bf in bundle_files:
                fname = Path(bf).name
                m_scan = re.search(r'Scan_(\d+)_Run_(\d+)', fname)
                scan = int(m_scan.group(1)) if m_scan else None
                run = int(m_scan.group(2)) if m_scan else None
                
                mass, _ = extract_mass_and_coupling(bf, scan, run)
                mass_val_str = _norm_float_str(mass)

                m = re.search(r'Generated_Events_(\d+)', str(bf))
                gen_idx = int(m.group(1)) if m else None

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
                print(f"Skipping coupling={target_coup}, BR_mu={br_mu}: no unprocessed bundles")
                continue
            job_counter += 1
            # Use positional naming to avoid problematic characters in paths
            jobDir = os.path.join(jobscriptDir, f"job{job_counter}_c{coup_pos}_b{br_pos}")

            if not os.path.exists(jobDir):
                os.makedirs(jobDir)
            else:
                t = input(f"WARNING: A job with this ID ({job_counter}) already exists at {jobDir}, overwrite? [y/n]")
                if t.lower() == "y":
                    # Remove contents but keep folder
                    for entry in os.listdir(jobDir):
                        entry_path = os.path.join(jobDir, entry)
                        try:
                            if os.path.isdir(entry_path):
                                import shutil
                                shutil.rmtree(entry_path)
                            else:
                                os.remove(entry_path)
                        except Exception:
                            pass
                else:
                    print(f"Skipping this job ID: {job_counter}...")
                    continue

            # Create a small wrapper python file that overrides the default lists in the
            # extract script and then calls its main(). This avoids modifying the
            # extract script and does not require executing it at import-time.
            wrapper_py = os.path.join(jobDir, "run_wrapper.py")
            wrapper_code = f"""import importlib.util
import sys, os

spec = importlib.util.spec_from_file_location("extract_mod", "{EXTRACT_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Override defaults so this run processes a single (coupling,BR) pair
mod.DEFAULT_TARGET_COUPLINGS = [{repr(target_coup)}]
mod.DEFAULT_BR_MU_TUPLE = ({repr(br_mu)},)
# Force args so seed computation uses the original coupling position
sys.argv = ["{EXTRACT_SCRIPT}", "--force-coup-pos", "{coup_pos}", "--BR_mu", "{br_mu}", "--mass-filter", "{args.mass_filter}"]
# Call main
if hasattr(mod, 'main'):
    mod.main()
else:
    raise RuntimeError("extract module has no main()")
"""

            with open(wrapper_py, 'w') as wf:
                wf.write(wrapper_code)

            # Create Bash script
            bashScriptName = os.path.join(jobDir, "runJob")
            with open(f"{bashScriptName}.sh","w") as f:
                f.write(bashHeader)
                # Run the wrapper with the virtualenv python
                f.write(f"/usera/fs568/set-anubis/.venv/bin/python {wrapper_py}\n")

            # Make the bash script executable
            try:
                os.chmod(f"{bashScriptName}.sh", 0o755)
            except Exception:
                pass

            # Create condor submission script
            condorString = f"executable = {bashScriptName}.sh" + "\n"
            condorString+= f"output = {jobDir}/job{job_counter}_output.log" + "\n"
            condorString+= f"error =  {jobDir}/job{job_counter}_error.log" + "\n"
            condorString+= f"request_memory = {args.memory}" + "\n"
            condorString+= f"log =  {jobDir}/job{job_counter}_log.log" + "\n"
            condorString+= "copy_to_spool = true\n"
            condorString+= "should_transfer_files = YES\n"
            condorString+= "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
            condorString+= "Queue"

            condorSubmissionFile = os.path.join(jobDir, "condor_submit.job")
            with open(condorSubmissionFile, 'w') as c:
                c.write(condorString)

            print(f"Prepared job: coupling={target_coup} (pos={coup_pos}), BR_mu={br_mu} (pos={br_pos}) -> {condorSubmissionFile}")
            print(f"To submit: condor_submit {condorSubmissionFile}")
            if not args.dryrun:
                os.system(f"condor_submit {condorSubmissionFile}")

print(f"\nTotal jobs prepared: {job_counter}")
