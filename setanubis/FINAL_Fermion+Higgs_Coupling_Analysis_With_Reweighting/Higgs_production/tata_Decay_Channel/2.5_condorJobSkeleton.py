import sys, os
from glob import glob
import argparse
import ast
import re
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--jobscriptDir", type=str, default="")
parser.add_argument("--memory", default = "6G")
parser.add_argument("--dryrun", action="store_true", help="Run without submitting, but produces the jobscripts and submission files which can be checked.")
parser.add_argument("--testArg", action="store_true", help="Just a proxy argument for reference")
args = parser.parse_args()


# Base bash header executed in each job
bashHeader = """#!/bin/bash

cd /usera/fs568/set-anubis
source /usera/fs568/set-anubis/.venv/bin/activate
"""

# Path to the extract-selection script we want each job to run
EXTRACT_SCRIPT = "/usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting/Higgs_production/tata_Decay_Channel/2_extract_selection_data.py"


def load_default_lists_from_extract(path):
    """Parse several DEFAULT_* constants from the extract script without executing it.

    Returns: (target_couplings, bundle_dir, pattern, outdir, output)
    """
    with open(path, 'r') as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}")
        return [], None, None, None, None

    target_couplings = []
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
    return list(target_couplings), bundle_dir, pattern, outdir, output


print("=============")
print(f"Target extract script: {EXTRACT_SCRIPT}")

if input("Confirm this is correct with y...").lower() != "y":
    raise Exception("Incorrect extract script path, please adjust your arguments")


if args.jobscriptDir=="":
    jobscriptDir = f"./jobscripts/"
else:
    jobscriptDir=args.jobscriptDir

if not os.path.exists(jobscriptDir):
    os.makedirs(jobscriptDir)


# Load default coupling list and other defaults from the extract script
target_couplings, bundle_dir, pattern, default_outdir, default_output = load_default_lists_from_extract(EXTRACT_SCRIPT)

if not target_couplings:
    print("No target couplings found in extract script; aborting.")
    sys.exit(1)

print(f"Found {len(target_couplings)} default couplings")

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

# Read existing output CSV to determine already-processed (coupling,bundle) entries
processed_by_gen_run = set()
processed_by_filepath = set()
processed_by_filename = set()
output_csv = None
if default_outdir and default_output:
    output_csv = os.path.join(default_outdir, default_output)

if output_csv and os.path.exists(output_csv):
    try:
        existing_df = pd.read_csv(output_csv)
        cols = set(existing_df.columns)

        # Mirror the extract script's precedence for deduplication keys
        if {'generated_events_index', 'scan', 'run', 'target_coupling'}.issubset(cols):
            for _, row in existing_df.iterrows():
                gen_idx = int(row['generated_events_index']) if pd.notna(row['generated_events_index']) else None
                scan_val = int(row['scan']) if pd.notna(row['scan']) else None
                run_val = int(row['run']) if pd.notna(row['run']) else None
                target_coup = str(row['target_coupling']) if pd.notna(row['target_coupling']) else None
                processed_by_gen_run.add((gen_idx, scan_val, run_val, target_coup))
            print(f"Found existing CSV with {len(existing_df)} entries; will skip already-processed bundles for matching coupling")

        elif {'generated_events_index', 'scan', 'run', 'reweight_lifetime'}.issubset(cols):
            for _, row in existing_df.iterrows():
                gen_idx = int(row['generated_events_index']) if pd.notna(row['generated_events_index']) else None
                scan_val = int(row['scan']) if pd.notna(row['scan']) else None
                run_val = int(row['run']) if pd.notna(row['run']) else None
                rewt = float(row['reweight_lifetime']) if pd.notna(row['reweight_lifetime']) else None
                processed_by_gen_run.add((gen_idx, scan_val, run_val, rewt))
            print(f"Found existing CSV with {len(existing_df)} entries; will skip already-processed bundles for matching lifetime")

        elif {'filepath', 'target_coupling'}.issubset(cols):
            for _, row in existing_df.iterrows():
                processed_by_filepath.add((row['filepath'], str(row['target_coupling']) if pd.notna(row['target_coupling']) else None))
            print(f"Found existing CSV with {len(existing_df)} entries; will skip already-processed bundles by filepath+coupling")

        elif {'filename', 'target_coupling'}.issubset(cols):
            for _, row in existing_df.iterrows():
                processed_by_filename.add((row['filename'], str(row['target_coupling']) if pd.notna(row['target_coupling']) else None))
            print(f"Found existing CSV with {len(existing_df)} entries; will skip already-processed bundles by filename+coupling")

        else:
            print("Existing CSV found but no usable dedup columns (coupling/lifetime/file). Will create jobs for all coupling combinations")

    except Exception as e:
        print(f"Warning: failed to read existing output CSV {output_csv}: {e}")
else:
    print("No existing output CSV found; will create jobs for all coupling combinations")

# Create a job for each target coupling that still has unprocessed bundles
job_counter = 0
for coup_pos, target_coup in enumerate(target_couplings, start=1):
    # Determine whether any bundle under bundle_files remains unprocessed for this coupling
    per_coupling_to_process = []
    for bf in bundle_files:
        fname = Path(bf).name
        m_scan = re.search(r'Scan_(\d+)_Run_(\d+)', fname)
        scan = int(m_scan.group(1)) if m_scan else None
        run = int(m_scan.group(2)) if m_scan else None
        m = re.search(r'Generated_Events_(\d+)', str(bf))
        gen_idx = int(m.group(1)) if m else None

        key_primary = (gen_idx, scan, run, str(target_coup) if target_coup is not None else None)
        if key_primary in processed_by_gen_run:
            continue

        key_fp = (bf, str(target_coup) if target_coup is not None else None)
        if key_fp in processed_by_filepath:
            continue

        key_fn = (fname, str(target_coup) if target_coup is not None else None)
        if key_fn in processed_by_filename:
            continue

        per_coupling_to_process.append(bf)

    if len(per_coupling_to_process) == 0:
        print(f"Skipping coupling={target_coup}: no unprocessed bundles")
        continue
    job_counter += 1
    # Use positional naming to avoid problematic characters in paths
    jobDir = os.path.join(jobscriptDir, f"job{job_counter}_c{coup_pos}_Higgs_tata")

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
    wrapper_code = f'''import importlib.util
import sys, os

spec = importlib.util.spec_from_file_location("extract_mod", "{EXTRACT_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Override defaults so this run processes a single coupling
mod.DEFAULT_TARGET_COUPLINGS = [{repr(target_coup)}]
# Force args so seed computation uses the original coupling position
sys.argv = ["{EXTRACT_SCRIPT}", "--force-coup-pos", "{coup_pos}"]
# Call main
if hasattr(mod, "main"):
    mod.main()
else:
    raise RuntimeError("extract module has no main()")
'''

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

    print(f"Prepared job: coupling={target_coup} (pos={coup_pos}) -> {condorSubmissionFile}")
    print(f"To submit: condor_submit {condorSubmissionFile}")
    if not args.dryrun:
        os.system(f"condor_submit {condorSubmissionFile}")

print(f"\nTotal jobs prepared: {job_counter}")
