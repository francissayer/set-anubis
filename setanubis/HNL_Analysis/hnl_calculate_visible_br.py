#!/usr/bin/env python3
"""
Calculate HNL visible branching ratios from MadGraph decay files.

This script extracts branching ratios from param_card files and identifies
which decay modes produce visible (charged lepton) final states.
"""

import os
import re
import argparse
from pathlib import Path

# ============================================================================
# DEFAULT PARAMETERS - Modify these to set default scan/run
# ============================================================================
DEFAULT_SCAN = 1          # Default scan number (set to None to require --scan)
DEFAULT_RUN = None        # Default run number (set to None for scan-level analysis)
DEFAULT_ALL_MASSES = False  # Set to True to analyze all masses by default
DEFAULT_RUNS_PATH = '/usera/fs568/set-anubis/HNL_Runs_test'
# ============================================================================


def parse_hnl_decay_table(param_card_path):
    """
    Parse HNL decay table from param_card.dat file.
    
    Returns:
        dict: Branching ratios categorized by decay type
    """
    with open(param_card_path, 'r') as f:
        lines = f.readlines()
    
    # Find the DECAY section for HNL (PDG 9900012)
    in_hnl_section = False
    br_data = []
    
    for i, line in enumerate(lines):
        if 'DECAY' in line and '9900012' in line:
            in_hnl_section = True
            continue
        
        if in_hnl_section:
            # Stop at next DECAY section or end
            if line.strip().startswith('DECAY') or line.strip().startswith('#') and 'PDG' in line:
                break
            
            # Parse BR line: "   BR   NDA  ID1 ID2 ID3 ..."
            parts = line.split()
            if len(parts) >= 4 and not line.strip().startswith('#'):
                try:
                    br = float(parts[0])
                    nda = int(parts[1])
                    daughters = [int(parts[i]) for i in range(2, 2 + nda)]
                    br_data.append({'BR': br, 'daughters': daughters})
                except (ValueError, IndexError):
                    continue
    
    return br_data


def classify_decay_visibility(daughters):
    """
    Classify decay mode as visible or invisible based on daughter particles.
    
    PDG codes:
        ±11: e±
        ±13: μ±
        ±15: τ±
        ±12, ±14, ±16: neutrinos (invisible)
        ±1,2,3,4,5,6: quarks (produce hadrons, some charged)
    
    Returns:
        str: 'leptonic' (2 charged leptons), 'semi-leptonic' (1 charged lepton), 
             'hadronic' (only quarks), or 'invisible'
    """
    charged_leptons = [11, 13, 15]  # e, μ, τ
    abs_daughters = [abs(d) for d in daughters]
    
    n_charged_leptons = sum(1 for d in abs_daughters if d in charged_leptons)
    has_quarks = any(d in [1, 2, 3, 4, 5, 6] for d in abs_daughters)
    
    if n_charged_leptons >= 2:
        return 'leptonic'
    elif n_charged_leptons == 1:
        if has_quarks:
            return 'semi-leptonic'
        else:
            return 'leptonic_single'  # e.g., ℓ + ν + ν
    elif has_quarks:
        return 'hadronic'
    else:
        return 'invisible'


def calculate_visible_branching_ratios(param_card_path):
    """
    Calculate visible branching ratios from HNL decay table.
    
    Returns:
        dict: Categorized branching ratios
    """
    br_data = parse_hnl_decay_table(param_card_path)
    
    categories = {
        'leptonic': 0.0,        # 2+ charged leptons (cleanest signature)
        'semi-leptonic': 0.0,   # 1 charged lepton + quarks
        'hadronic': 0.0,        # Only quarks
        'invisible': 0.0,       # Only neutrinos
        'total': 0.0
    }
    
    decay_details = {
        'leptonic': [],
        'semi-leptonic': [],
        'hadronic': [],
        'invisible': []
    }
    
    for entry in br_data:
        br = entry['BR']
        daughters = entry['daughters']
        category = classify_decay_visibility(daughters)
        
        if category in categories:
            categories[category] += br
            decay_details[category].append({
                'BR': br,
                'daughters': daughters
            })
        
        categories['total'] += br
    
    # Visible = leptonic + semi-leptonic (for track-based detectors like ANUBIS)
    categories['visible_total'] = categories['leptonic'] + categories['semi-leptonic']
    
    # For ANUBIS displaced vertex signature, leptonic is best
    categories['cleanest_visible'] = categories['leptonic']
    
    return categories, decay_details


def print_branching_ratios(categories, decay_details, mass, numixing):
    """Print formatted branching ratio results."""
    print("="*70)
    mass_str = f"{mass:.2f} GeV" if mass is not None else "Unknown"
    numixing_str = f"{numixing:.2e}" if numixing is not None else "Unknown"
    print(f"HNL BRANCHING RATIOS (Mass = {mass_str}, |V_eN| = {numixing_str})")
    print("="*70)
    
    print(f"\nLeptonic (2+ charged leptons):     {categories['leptonic']:.4f} ({categories['leptonic']*100:.2f}%)")
    print(f"  Number of decay modes: {len(decay_details['leptonic'])}")
    
    print(f"\nSemi-leptonic (1 lepton + quarks): {categories['semi-leptonic']:.4f} ({categories['semi-leptonic']*100:.2f}%)")
    print(f"  Number of decay modes: {len(decay_details['semi-leptonic'])}")
    
    print(f"\nHadronic (only quarks):             {categories['hadronic']:.4f} ({categories['hadronic']*100:.2f}%)")
    print(f"  Number of decay modes: {len(decay_details['hadronic'])}")
    
    print(f"\nInvisible (only neutrinos):         {categories['invisible']:.4f} ({categories['invisible']*100:.2f}%)")
    print(f"  Number of decay modes: {len(decay_details['invisible'])}")
    
    print("\n" + "-"*70)
    print(f"Total visible (leptonic + semi-leptonic): {categories['visible_total']:.4f} ({categories['visible_total']*100:.2f}%)")
    print(f"Cleanest signature (leptonic only):       {categories['cleanest_visible']:.4f} ({categories['cleanest_visible']*100:.2f}%)")
    print(f"Total BR (check):                          {categories['total']:.4f}")
    print("="*70)


def extract_mass_and_mixing_from_param_card(param_card_path):
    """Extract HNL mass and mixing angle from param_card."""
    mass = None
    numixing = None
    
    with open(param_card_path, 'r') as f:
        lines = f.readlines()
    
    # Extract mass from MASS block
    in_mass_block = False
    for line in lines:
        if line.strip().startswith('BLOCK MASS') or line.strip().startswith('###') and 'MASS' in line:
            in_mass_block = True
            continue
        
        if in_mass_block:
            if line.strip().startswith('BLOCK') or line.strip().startswith('###'):
                in_mass_block = False
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pdg = int(parts[0])
                    if pdg == 9900012:  # HNL N1
                        mass = float(parts[1])
                except (ValueError, IndexError):
                    pass
    
    # Extract mixing from NUMIXING block
    in_numixing_block = False
    for line in lines:
        if line.strip().startswith('BLOCK NUMIXING') or line.strip().startswith('###') and 'NUMIXING' in line:
            in_numixing_block = True
            continue
        
        if in_numixing_block:
            if line.strip().startswith('BLOCK') or line.strip().startswith('###'):
                in_numixing_block = False
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    if idx == 1:  # VeN1
                        numixing = float(parts[1])
                except (ValueError, IndexError):
                    pass
    
    return mass, numixing


def main():
    parser = argparse.ArgumentParser(
        description='Calculate HNL visible branching ratios from param_card or banner files'
    )
    parser.add_argument(
        '--param-card',
        type=str,
        help='Path to specific param_card.dat file'
    )
    parser.add_argument(
        '--scan',
        type=int,
        default=DEFAULT_SCAN,
        help=f'Scan number to analyze (default: {DEFAULT_SCAN})'
    )
    parser.add_argument(
        '--run',
        type=int,
        default=DEFAULT_RUN,
        help='Run number within scan (used with --scan)'
    )
    parser.add_argument(
        '--runs-path',
        type=str,
        default=DEFAULT_RUNS_PATH,
        help=f'Base path to HNL_Runs_test directory (default: {DEFAULT_RUNS_PATH})'
    )
    parser.add_argument(
        '--all-masses',
        action='store_true',
        default=DEFAULT_ALL_MASSES,
        help='Calculate BR for all mass points in scan by reading banner files from each run'
    )
    parser.add_argument(
        '--from-banner',
        type=str,
        help='Path to specific banner.txt file to extract BRs from'
    )
    
    args = parser.parse_args()
    
    # Determine which source to use
    if args.from_banner:
        # Extract from banner file
        mass, numixing = extract_mass_and_mixing_from_param_card(args.from_banner)
        print(f"\nAnalyzing banner: {args.from_banner}\n")
        categories, decay_details = calculate_visible_branching_ratios(args.from_banner)
        print_branching_ratios(categories, decay_details, mass, numixing)
        
    elif args.param_card:
        param_card_path = args.param_card
        mass, numixing = extract_mass_and_mixing_from_param_card(param_card_path)
        
        print(f"\nAnalyzing: {param_card_path}\n")
        categories, decay_details = calculate_visible_branching_ratios(param_card_path)
        print_branching_ratios(categories, decay_details, mass, numixing)
        
    elif args.scan:
        scan_dir = f'HNL_Condor_CCDY_qqe_Scan_{args.scan}'
        scan_path = os.path.join(args.runs_path, scan_dir)
        
        if args.all_masses:
            # Find all decayed run banner files (each has mass-specific BRs)
            events_dir = os.path.join(scan_path, 'Events')
            run_dirs = sorted([d for d in Path(events_dir).iterdir() 
                             if d.is_dir() and d.name.endswith('_decayed_1')])
            
            if not run_dirs:
                print(f"ERROR: No decayed run directories found in {events_dir}")
                print("Looking for directories ending in '_decayed_1'")
                return
            
            print(f"\n{'='*70}")
            print(f"Branching Ratios for All Mass Points (Scan {args.scan})")
            print(f"{'='*70}\n")
            
            for run_dir in run_dirs:
                # Find banner file
                banner_files = list(run_dir.glob('*_banner.txt'))
                if banner_files:
                    banner_path = str(banner_files[0])
                    mass, numixing = extract_mass_and_mixing_from_param_card(banner_path)
                    categories, _ = calculate_visible_branching_ratios(banner_path)
                    
                    run_name = run_dir.name.replace('_decayed_1', '')
                    run_num = int(run_name.split('_')[1])
                    
                    print(f"Run {run_num:2d}: Mass={mass:6.2f} GeV, |V_eN|={numixing:.2e}")
                    print(f"  Leptonic: {categories['leptonic']*100:5.2f}%, "
                          f"Semi-leptonic: {categories['semi-leptonic']*100:5.2f}%, "
                          f"Hadronic: {categories['hadronic']*100:5.2f}%, "
                          f"Invisible: {categories['invisible']*100:5.2f}%")
                    print(f"  Total visible: {categories['visible_total']*100:5.2f}%\n")
            return
        
        # Default: use single decay directory param_card
        decay_dir = os.path.join(scan_path, 'decay_9900012_0')
        
        if args.run:
            param_card_path = os.path.join(decay_dir, f'Events/run_{args.run:02d}/param_card.dat')
            
            if not os.path.exists(param_card_path):
                print(f"ERROR: param_card not found at {param_card_path}")
                return
            
            mass, numixing = extract_mass_and_mixing_from_param_card(param_card_path)
            print(f"\nAnalyzing Scan {args.scan}, Run {args.run}\n")
            categories, decay_details = calculate_visible_branching_ratios(param_card_path)
            print_branching_ratios(categories, decay_details, mass, numixing)
        else:
            # Use default param_card from Cards directory
            param_card_path = os.path.join(decay_dir, 'Cards/param_card.dat')
            
            if not os.path.exists(param_card_path):
                print(f"ERROR: param_card not found at {param_card_path}")
                return
            
            mass, numixing = extract_mass_and_mixing_from_param_card(param_card_path)
            print(f"\nAnalyzing Scan {args.scan} (default param_card)\n")
            categories, decay_details = calculate_visible_branching_ratios(param_card_path)
            print_branching_ratios(categories, decay_details, mass, numixing)
    
    else:
        print("ERROR: Must specify either --param-card, --from-banner, or --scan")
        parser.print_help()


if __name__ == '__main__':
    main()
