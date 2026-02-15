#!/usr/bin/env python3
"""Check if decays are actually hitting the cavern geometry."""
import sys
sys.path.insert(0, '/usera/fs568/set-anubis/setanubis')
import numpy as np
from SetAnubis.core.Selection.adapters.output.WriteLoadSelectionDict import load_bundle
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern

bundle = load_bundle("ALP_Z_test_sampledfs.pkl.gz")
llps = bundle["LLPs"]
cav = ATLASCavern()

print(f"Total LLPs: {len(llps)}")

# Check a few vertex values
print(f"\nFirst 5 decay vertices (raw):")
for i in range(min(5, len(llps))):
    print(f"  {llps.iloc[i]['decayVertex']}")

# Check scale
first_vertex = llps.iloc[0]['decayVertex']
print(f"\nFirst vertex scale check:")
print(f"  Raw values: {first_vertex}")
print(f"  Distance from origin: {np.linalg.norm(first_vertex[1:] if len(first_vertex)==4 else first_vertex):.2f}")

# Check how many are in the distance range
decay_dist = llps['decayVertexDist']
in_range = (decay_dist >= 1) & (decay_dist <= 50)
print(f"Decaying 1-50m from IP: {in_range.sum()}")

# Now check actual geometry
n_in_cavern = 0
n_in_shaft = 0

for idx, row in llps[in_range].iterrows():
    vertex = row['decayVertex']  # This should be (t, x, y, z) or (x, y, z)
    
    # Extract x, y, z (skip t if present)
    if len(vertex) == 4:
        x, y, z = vertex[1], vertex[2], vertex[3]  # Skip time component
    else:
        x, y, z = vertex[0], vertex[1], vertex[2]
    
    # Convert mm to m if needed (check scale)
    if abs(x) > 1000:  # Likely in mm
        x, y, z = x/1000, y/1000, z/1000
    
    # Shift from IP to cavern center coordinates
    x_cav = x - cav.IP["x"]
    y_cav = y - cav.IP["y"]
    z_cav = z - cav.IP["z"]
    
    if cav.inCavern(x_cav, y_cav, z_cav):
        n_in_cavern += 1
    if cav.inShaft(x_cav, y_cav, z_cav, shafts=["PX14"]):
        n_in_shaft += 1

print(f"\nActually in cavern: {n_in_cavern}")
print(f"Actually in PX14 shaft: {n_in_shaft}")
print(f"\nGeometric acceptance: {100*n_in_cavern/in_range.sum():.3f}%")
