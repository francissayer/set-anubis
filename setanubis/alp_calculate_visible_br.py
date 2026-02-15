#!/usr/bin/env python3
"""
Calculate ALP visible branching ratios from MadGraph decay files.

This script extracts branching ratios from param_card/banner files and identifies
which decay modes produce visible final states for ANUBIS detector.
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
DEFAULT_SHOW_DETAILS = False  # Set to True to show detailed decay modes by default
DEFAULT_RUNS_PATH = '/usera/fs568/set-anubis/ALP_Z_Runs'
# ============================================================================


def parse_alp_decay_table(param_card_path):
    """
    Parse ALP decay table from param_card.dat or banner file.
    
    Returns:
        list: Branching ratios with daughter particle information
    """
    with open(param_card_path, 'r') as f:
        lines = f.readlines()
    
    # Find the DECAY section for ALP (PDG 9000005)
    in_alp_section = False
    br_data = []
    
    for i, line in enumerate(lines):
        if 'DECAY' in line and '9000005' in line:
            in_alp_section = True
            continue
        
        if in_alp_section:
            # Stop at next DECAY section or end
            if line.strip().startswith('DECAY') or (line.strip().startswith('#') and 'PDG' in line):
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


def classify_alp_decay_visibility(daughters):
    """
    Classify ALP decay mode visibility for ANUBIS detector.
    
    PDG codes:
        22: photon (typically invisible/escapes detector)
        21: gluon (produces hadronic jet)
        ±11: e±
        ±13: μ±
        ±15: τ±
        ±12, ±14, ±16: neutrinos (invisible)
        ±1,2,3,4,5,6: quarks (produce hadrons)
        23: Z boson
        24: W± boson
        25: Higgs
    
    Returns:
        str: Category of decay ('photons', 'leptons', 'quarks', 'bosons', 'mixed', 'invisible')
    """
    abs_daughters = [abs(d) for d in daughters]
    
    # Count different particle types
    n_photons = sum(1 for d in abs_daughters if d == 22)
    n_gluons = sum(1 for d in abs_daughters if d == 21)
    n_charged_leptons = sum(1 for d in abs_daughters if d in [11, 13, 15])
    n_neutrinos = sum(1 for d in abs_daughters if d in [12, 14, 16])
    n_quarks = sum(1 for d in abs_daughters if d in [1, 2, 3, 4, 5, 6])
    n_bosons = sum(1 for d in abs_daughters if d in [23, 24, 25])
    
    # Classification
    if n_photons >= 2 and len(abs_daughters) == 2:
        return 'photons'  # Pure photon decays (diphoton final state)
    elif n_charged_leptons >= 2:
        return 'leptons'  # Charged lepton pairs (visible)
    elif n_charged_leptons == 1:
        return 'semi-leptonic'  # One lepton + something else
    elif n_quarks >= 2 or n_gluons >= 2:
        return 'hadronic'  # Quarks/gluons (jets)
    elif n_bosons > 0:
        return 'bosons'  # Decays to W/Z/H
    elif n_photons > 0:
        return 'photonic'  # Photons mixed with other particles
    elif n_neutrinos > 0:
        return 'invisible'  # Only neutrinos
    else:
        return 'other'


def get_decay_description(daughters):
    """Get human-readable description of decay mode."""
    pdg_names = {
        22: 'γ', 21: 'g',
        11: 'e⁻', -11: 'e⁺', 13: 'μ⁻', -13: 'μ⁺', 15: 'τ⁻', -15: 'τ⁺',
        12: 'νe', -12: 'ν̄e', 14: 'νμ', -14: 'ν̄μ', 16: 'ντ', -16: 'ν̄τ',
        1: 'd', -1: 'd̄', 2: 'u', -2: 'ū', 3: 's', -3: 's̄',
        4: 'c', -4: 'c̄', 5: 'b', -5: 'b̄', 6: 't', -6: 't̄',
        23: 'Z', 24: 'W⁺', -24: 'W⁻', 25: 'H'
    }
    
    parts = [pdg_names.get(d, f'({d})') for d in daughters]
    return ' '.join(parts)


def calculate_visible_branching_ratios(param_card_path):
    """
    Calculate visible branching ratios from ALP decay table.
    
    Returns:
        dict: Categorized branching ratios
    """
    br_data = parse_alp_decay_table(param_card_path)
    
    categories = {
        'photons': 0.0,        # Diphoton (typically escapes)
        'leptons': 0.0,        # Charged lepton pairs (VISIBLE)
        'semi-leptonic': 0.0,  # One lepton + other
        'hadronic': 0.0,       # Quarks/gluons (jets, VISIBLE)
        'bosons': 0.0,         # W/Z/H
        'photonic': 0.0,       # Photons mixed with others
        'invisible': 0.0,      # Neutrinos only
        'other': 0.0,          # Other combinations
        'total': 0.0
    }
    
    decay_details = {cat: [] for cat in categories.keys()}
    
    for entry in br_data:
        br = entry['BR']
        daughters = entry['daughters']
        category = classify_alp_decay_visibility(daughters)
        
        if category in categories:
            categories[category] += br
            decay_details[category].append({
                'BR': br,
                'daughters': daughters,
                'description': get_decay_description(daughters)
            })
        
        categories['total'] += br
    
    # Define visibility for ANUBIS
    # Leptons: clearly visible (charged tracks)
    # Hadronic: visible (jets)
    # Bosons: decay to leptons/quarks (visible)
    # Photonic + semi-leptonic: partially visible
    categories['clearly_visible'] = categories['leptons'] + categories['hadronic']
    categories['visible_with_bosons'] = categories['clearly_visible'] + categories['bosons']
    categories['all_visible'] = (categories['leptons'] + categories['semi-leptonic'] + 
                                 categories['hadronic'] + categories['bosons'])
    
    return categories, decay_details


def print_branching_ratios(categories, decay_details, mass, caphi, show_details=False):
    """Print formatted branching ratio results."""
    print("="*70)
    mass_str = f"{mass:.2f} GeV" if mass is not None else "Unknown"
    caphi_str = f"{caphi:.2e}" if caphi is not None else "Unknown"
    print(f"ALP BRANCHING RATIOS (Mass = {mass_str}, CaPhi = {caphi_str})")
    print("="*70)
    
    print(f"\nPhotons (γγ):                      {categories['photons']:.4f} ({categories['photons']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['photons'])}")
    
    print(f"\nLeptons (ℓ⁺ℓ⁻):                     {categories['leptons']:.4f} ({categories['leptons']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['leptons'])}")
    if show_details and decay_details['leptons']:
        for d in decay_details['leptons'][:3]:
            print(f"    BR={d['BR']:.4f}: {d['description']}")
    
    print(f"\nSemi-leptonic (ℓ + X):             {categories['semi-leptonic']:.4f} ({categories['semi-leptonic']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['semi-leptonic'])}")
    
    print(f"\nHadronic (quarks/gluons):          {categories['hadronic']:.4f} ({categories['hadronic']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['hadronic'])}")
    if show_details and decay_details['hadronic']:
        for d in decay_details['hadronic'][:3]:
            print(f"    BR={d['BR']:.4f}: {d['description']}")
    
    print(f"\nBosons (W/Z/H):                    {categories['bosons']:.4f} ({categories['bosons']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['bosons'])}")
    
    print(f"\nPhotonic (γ + other):              {categories['photonic']:.4f} ({categories['photonic']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['photonic'])}")
    
    print(f"\nInvisible (neutrinos):             {categories['invisible']:.4f} ({categories['invisible']*100:.2f}%)")
    print(f"  Modes: {len(decay_details['invisible'])}")
    
    print("\n" + "-"*70)
    print(f"Clearly visible (ℓℓ + hadronic):       {categories['clearly_visible']:.4f} ({categories['clearly_visible']*100:.2f}%)")
    print(f"Visible including bosons:              {categories['visible_with_bosons']:.4f} ({categories['visible_with_bosons']*100:.2f}%)")
    print(f"All potentially visible:               {categories['all_visible']:.4f} ({categories['all_visible']*100:.2f}%)")
    print(f"Total BR (check):                      {categories['total']:.4f}")
    print("="*70)


def extract_mass_and_coupling_from_card(card_path):
    """Extract ALP mass and coupling (CaPhi) from param_card or banner file."""
    mass = None
    caphi = None
    
    with open(card_path, 'r') as f:
        lines = f.readlines()
    
    # Extract mass from MASS block (PDG 9000005)
    in_mass_block = False
    for line in lines:
        if line.strip().startswith('BLOCK MASS') or (line.strip().startswith('###') and 'MASS' in line):
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
                    if pdg == 9000005:  # ALP
                        mass = float(parts[1])
                except (ValueError, IndexError):
                    pass
    
    # Extract CaPhi from ALPPARS block (index 5)
    in_alppars_block = False
    for line in lines:
        if line.strip().startswith('BLOCK ALPPARS') or (line.strip().startswith('###') and 'ALPPARS' in line):
            in_alppars_block = True
            continue
        
        if in_alppars_block:
            if line.strip().startswith('BLOCK') or line.strip().startswith('###'):
                in_alppars_block = False
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    if idx == 5:  # CaPhi
                        caphi = float(parts[1])
                except (ValueError, IndexError):
                    pass
    
    return mass, caphi


def main():
    parser = argparse.ArgumentParser(
        description='Calculate ALP visible branching ratios from param_card or banner files'
    )
    parser.add_argument(
        '--param-card',
        type=str,
        help='Path to specific param_card.dat file'
    )
    parser.add_argument(
        '--from-banner',
        type=str,
        help='Path to specific banner.txt file to extract BRs from'
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
        help=f'Base path to ALP_Z_Runs directory (default: {DEFAULT_RUNS_PATH})'
    )
    parser.add_argument(
        '--all-masses',
        action='store_true',
        default=DEFAULT_ALL_MASSES,
        help='Calculate BR for all mass points in scan by reading banner files from each run'
    )
    parser.add_argument(
        '--details',
        action='store_true',
        default=DEFAULT_SHOW_DETAILS,
        help='Show detailed decay mode information'
    )
    
    args = parser.parse_args()
    
    # Determine which source to use
    if args.from_banner:
        # Extract from banner file
        mass, caphi = extract_mass_and_coupling_from_card(args.from_banner)
        print(f"\nAnalyzing banner: {args.from_banner}\n")
        categories, decay_details = calculate_visible_branching_ratios(args.from_banner)
        print_branching_ratios(categories, decay_details, mass, caphi, args.details)
        
    elif args.param_card:
        mass, caphi = extract_mass_and_coupling_from_card(args.param_card)
        print(f"\nAnalyzing: {args.param_card}\n")
        categories, decay_details = calculate_visible_branching_ratios(args.param_card)
        print_branching_ratios(categories, decay_details, mass, caphi, args.details)
        
    elif args.scan:
        scan_dir = f'ALP_axZ_scan_{args.scan}'
        scan_path = os.path.join(args.runs_path, scan_dir)
        
        if args.all_masses:
            # Find all decayed run banner files (each has mass-specific BRs)
            events_dir = os.path.join(scan_path, 'Events')
            
            if not os.path.exists(events_dir):
                print(f"ERROR: Events directory not found at {events_dir}")
                return
            
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
                    mass, caphi = extract_mass_and_coupling_from_card(banner_path)
                    categories, _ = calculate_visible_branching_ratios(banner_path)
                    
                    run_name = run_dir.name.replace('_decayed_1', '')
                    run_num = int(run_name.split('_')[1])
                    
                    print(f"Run {run_num:2d}: Mass={mass:6.2f} GeV, CaPhi={caphi:.2e}")
                    print(f"  Photons: {categories['photons']*100:5.2f}%, "
                          f"Leptons: {categories['leptons']*100:5.2f}%, "
                          f"Hadronic: {categories['hadronic']*100:5.2f}%")
                    print(f"  Clearly visible: {categories['clearly_visible']*100:5.2f}%\n")
            return
        
        # Default: use single decay directory
        decay_dirs = sorted([d for d in Path(scan_path).iterdir() 
                           if d.is_dir() and d.name.startswith('decay_9000005_')])
        
        if not decay_dirs:
            print(f"ERROR: No decay directories found in {scan_path}")
            return
        
        # Use first decay directory
        decay_dir = decay_dirs[0]
        param_card_path = decay_dir / 'Cards' / 'param_card.dat'
        
        if not param_card_path.exists():
            print(f"ERROR: param_card not found at {param_card_path}")
            return
        
        mass, caphi = extract_mass_and_coupling_from_card(str(param_card_path))
        print(f"\nAnalyzing Scan {args.scan} ({decay_dir.name})\n")
        categories, decay_details = calculate_visible_branching_ratios(str(param_card_path))
        print_branching_ratios(categories, decay_details, mass, caphi, args.details)
    
    else:
        print("ERROR: Must specify either --param-card, --from-banner, or --scan")
        parser.print_help()


if __name__ == '__main__':
    main()
