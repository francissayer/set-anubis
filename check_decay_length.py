#!/usr/bin/env python3
"""Check decay vertex distribution."""
import sys
sys.path.insert(0, '/usera/fs568/set-anubis/setanubis')
import numpy as np
from SetAnubis.core.Selection.adapters.output.WriteLoadSelectionDict import load_bundle

bundle = load_bundle("ALP_Z_test_sampledfs.pkl.gz")
llps = bundle["LLPs"]
decaying = llps[llps['status'] != 1]

print(f"Total LLPs: {len(llps)}, Decaying: {len(decaying)}")

# Check the theoretical proper decay length (ctau)
# Note: Bundle stores positions in millimeters
if 'ctau' in llps.columns:
    ctau_mm = llps['ctau'].values
    print(f"\nTheoretical proper lifetime (ctau):")
    print(f"  Min: {np.min(ctau_mm):.4f} mm = {np.min(ctau_mm)/1000:.6f} m")
    print(f"  Median: {np.median(ctau_mm):.4f} mm = {np.median(ctau_mm)/1000:.6f} m")
    print(f"  Max: {np.max(ctau_mm):.4f} mm = {np.max(ctau_mm)/1000:.6f} m")

if len(decaying) > 0:
    # Decay distances are stored in mm, convert to meters for display
    decay_dist_mm = decaying['decayVertexDist']
    decay_dist_m = decay_dist_mm / 1000.0
    
    print(f"\nDecay distances:")
    print(f"  Min: {decay_dist_mm.min():.2f} mm = {decay_dist_m.min():.6f} m")
    print(f"  P10: {np.percentile(decay_dist_mm, 10):.2f} mm = {np.percentile(decay_dist_m, 10):.6f} m")
    print(f"  Median: {np.median(decay_dist_mm):.2f} mm = {np.median(decay_dist_m):.6f} m")
    print(f"  Mean: {decay_dist_mm.mean():.2f} mm = {decay_dist_m.mean():.6f} m")
    print(f"  P90: {np.percentile(decay_dist_mm, 90):.2f} mm = {np.percentile(decay_dist_m, 90):.6f} m")
    print(f"  Max: {decay_dist_mm.max():.2f} mm = {decay_dist_m.max():.6f} m")
    
    # Categorize (in mm)
    n_prompt = (decay_dist_mm < 1000).sum()  # <1m
    n_atlas = ((decay_dist_mm >= 1000) & (decay_dist_mm < 25000)).sum()  # 1-25m
    n_anubis = ((decay_dist_mm >= 25000) & (decay_dist_mm < 100000)).sum()  # 25-100m
    n_beyond = (decay_dist_mm >= 100000).sum()  # >100m
    
    print(f"\nCategories:")
    print(f"  Prompt (<1m = <1000mm): {n_prompt} ({100*n_prompt/len(decaying):.1f}%)")
    print(f"  ATLAS (1-25m): {n_atlas} ({100*n_atlas/len(decaying):.1f}%)")
    print(f"  ANUBIS (25-100m): {n_anubis} ({100*n_anubis/len(decaying):.1f}%)")
    print(f"  Beyond (>100m): {n_beyond} ({100*n_beyond/len(decaying):.1f}%)")
