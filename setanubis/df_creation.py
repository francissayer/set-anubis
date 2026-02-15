import os
import pyhepmc

from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface
from SetAnubis.core.Selection.domain.HepMCFrameBuilder import HepmcFrameBuilder, HepmcFrameOptions


HEPMC_FILE = ("/usera/fs568/set-anubis/ALP_Z_Runs/ALP_axZ_scan_32/Events/run_17_decayed_1/tag_1_pythia8_events.hepmc.gz")


UFO_PATH = os.path.abspath("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH")

if __name__ == "__main__":
    neo = SetAnubisInterface(UFO_PATH)

    def on_progress(n: int):
        print(f"[build] {n} events")

    builder = HepmcFrameBuilder(
        neo_manager=neo,
        options=HepmcFrameOptions(progress_every=200, compute_met=True),
        progress_hook=on_progress,
    )

    with pyhepmc.open(HEPMC_FILE) as stream:
        df, unknown = builder.build_from_events(stream)
        
        df.to_pickle("ALP_Z_df_Scan_32_Run_17.pkl")