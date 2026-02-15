"""
Simple script to check decay positions using ATLASCavern.inCavern() with verbose output.
"""
from SetAnubis.core.Selection.domain.DatasetSource import EventsBundleSource
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern

# Set your pickle file path here
PKL_FILE = "/usera/fs568/set-anubis/ALP_Z_Runs/ALP_Z_sampledfs_Scan_3_Run_4.pkl.gz"


def check_positions(pkl_file: str):
    """Check if decay positions are in ATLAS cavern using inCavern function."""
    # Load bundle and get LLPs
    bundle = EventsBundleSource.from_bundle_file(pkl_file).materialize()
    llps = bundle["LLPs"]
    
    # Get decay vertex column
    decay_col = [c for c in llps.columns if 'decay' in c.lower() and 'vertex' in c.lower()][0]
    
    # Initialize cavern
    cavern = ATLASCavern()
    print(f"Checking {len(llps)} decay positions...")
    
    # Check each position
    in_count = 0
    for idx, row in llps.iterrows():
        # Get coordinates from bundle (in mm)
        x_mm, y_mm, z_mm = row[decay_col][:3]
        
        # Convert to meters and transform to cavern coords
        x, y, z = cavern.coordsToOrigin(x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3)
        
        # Call inCavern function
        if cavern.inCavern(x, y, z, maxRadius="", verbose=True):
            in_count += 1
    
    print(f"\nResults: {in_count}/{len(llps)} ({100*in_count/len(llps):.1f}%) in cavern")


if __name__ == "__main__":
    check_positions(PKL_FILE)
